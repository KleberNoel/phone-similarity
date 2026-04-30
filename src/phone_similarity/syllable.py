"""
Syllable segmentation with Maximum Onset Principle.

Provides language-aware syllabification using a sonority hierarchy derived from
universal phonological features.  Dispatches to Cython when available.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import ClassVar, Protocol, Union

from phone_similarity._dispatch import HAS_CYTHON_SYLLABIFIER as _HAS_CYTHON_SYLLABLE
from phone_similarity._dispatch import cy_batch_syllabify as _cy_batch_syllabify
from phone_similarity._dispatch import cy_syllabify as _cy_syllabify
from phone_similarity.universal_features import UniversalFeatureEncoder

# Sonority ranks
RANK_STOP = 1
RANK_FRICATIVE = 2
RANK_NASAL = 3
RANK_LIQUID = 4
RANK_GLIDE = 5
RANK_VOWEL = 6
RANK_UNKNOWN = 0


# Syllable data class
@dataclasses.dataclass(frozen=True)
class Syllable:
    """A syllable decomposed into onset, nucleus, and coda (all tuples of IPA strings)."""

    onset: tuple[str, ...]
    nucleus: tuple[str, ...]
    coda: tuple[str, ...]
    stress: str | None = None

    @property
    def phonemes(self) -> tuple[str, ...]:
        """All phonemes in order: onset + nucleus + coda."""
        return self.onset + self.nucleus + self.coda

    @property
    def rime(self) -> tuple[str, ...]:
        """Nucleus + coda (the rhyming part of the syllable)."""
        return self.nucleus + self.coda

    @property
    def is_stressed(self) -> bool:
        """True if this syllable carries primary or secondary stress."""
        return self.stress is not None

    def __len__(self) -> int:
        return len(self.onset) + len(self.nucleus) + len(self.coda)


class SonorityScale:
    """Maps IPA phonemes to integer sonority ranks via universal features."""

    _cache: ClassVar[dict[str, int]] = {}

    def __init__(
        self,
        extra_ranks: dict[str, int] | None = None,
    ) -> None:
        self._extra = extra_ranks or {}

    # noinspection PyMethodMayBeStatic
    def rank(self, phoneme: str) -> int:
        """Return the sonority rank for *phoneme*."""
        if phoneme in self._extra:
            return self._extra[phoneme]
        cached = SonorityScale._cache.get(phoneme)
        if cached is not None:
            return cached
        r = self._compute_rank(phoneme)
        SonorityScale._cache[phoneme] = r
        return r

    @staticmethod
    def _compute_rank(phoneme: str) -> int:
        feats = UniversalFeatureEncoder.feature_dict(phoneme)
        if not feats:
            return RANK_UNKNOWN

        syl = feats.get("syl", 0)
        son = feats.get("son", 0)
        cons = feats.get("cons", 0)
        cont = feats.get("cont", 0)
        nas = feats.get("nas", 0)
        cor = feats.get("cor", 0)

        if syl == 1:
            return RANK_VOWEL
        if son == 1 and cons != 1:
            # Approximants: distinguish glides (j, w) from approximant
            # rhotics (ɹ) — rhotics are coronal, glides are not.
            if cor == 1:
                return RANK_LIQUID
            return RANK_GLIDE
        if son == 1 and nas == 1:
            return RANK_NASAL
        if son == 1:
            return RANK_LIQUID
        if cont == 1:
            return RANK_FRICATIVE
        return RANK_STOP

    def rank_tokens(self, tokens: Sequence[str]) -> list[int]:
        """Return sonority ranks for a sequence of phonemes."""
        return [self.rank(ph) for ph in tokens]

    def build_rank_map(self, phonemes: Sequence[str]) -> dict[str, int]:
        """Build a ``{phoneme: rank}`` dict for an inventory."""
        return {ph: self.rank(ph) for ph in phonemes}


class SyllabificationStrategy(Protocol):
    """Strategy interface for syllabification algorithms.

    Implementations receive pre-computed sonority ranks for each token
    position and the set of vowel phonemes.
    """

    def syllabify(
        self,
        tokens: Sequence[str],
        vowels: frozenset[str],
        sonority: Sequence[int],
    ) -> list[Syllable]: ...


class MaxOnsetSegmenter:
    """Syllabification using the Maximum Onset Principle (MOP) with SSP-compliant onset extension."""

    def __init__(self, sibilant_appendix: bool = True) -> None:
        self._sibilant_appendix = sibilant_appendix

    def syllabify(
        self,
        tokens: Sequence[str],
        vowels: frozenset[str],
        sonority: Sequence[int],
    ) -> list[Syllable]:
        """Segment *tokens* into syllables."""
        n = len(tokens)
        if n == 0:
            return []

        spans: list[tuple[int, int]] = []
        i = 0
        while i < n:
            if tokens[i] in vowels:
                start = i
                while i < n and tokens[i] in vowels:
                    i += 1
                spans.append((start, i))
            else:
                i += 1

        if not spans:
            # No vowels — treat everything as a single degenerate syllable
            return [Syllable(onset=tuple(tokens), nucleus=(), coda=())]

        syllables: list[Syllable] = []

        for idx, (v_start, v_end) in enumerate(spans):
            if idx == 0:
                onset = tuple(tokens[:v_start])
            else:
                prev_end = spans[idx - 1][1]
                cluster = tokens[prev_end:v_start]
                split = self._split_cluster(cluster, sonority[prev_end:v_start])
                if split > 0:
                    old = syllables[-1]
                    syllables[-1] = Syllable(
                        onset=old.onset,
                        nucleus=old.nucleus,
                        coda=old.coda + tuple(cluster[:split]),
                    )
                onset = tuple(cluster[split:])

            nucleus = tuple(tokens[v_start:v_end])

            if idx == len(spans) - 1:
                coda = tuple(tokens[v_end:])
            else:
                coda = ()  # will be filled when processing next span

            syllables.append(Syllable(onset=onset, nucleus=nucleus, coda=coda))

        return syllables

    def _split_cluster(
        self,
        cluster: Sequence[str],
        son_values: Sequence[int],
    ) -> int:
        """Return the index in *cluster* where the onset begins.

        Everything before this index belongs to the coda of the
        preceding syllable; everything from this index onward is the
        onset of the following syllable.
        """
        k = len(cluster)
        if k <= 1:
            return 0  # single consonant → all onset (MOP)

        # Scan from the right, extending onset leftward as long as
        # sonority is strictly rising towards the nucleus.
        onset_start = k - 1
        while onset_start > 0:
            if son_values[onset_start - 1] < son_values[onset_start]:
                onset_start -= 1
            else:
                break

        # Sibilant appendix: allow /s/-type (RANK_FRICATIVE) to attach
        # even though it has *higher* sonority than the following stop.
        if (
            self._sibilant_appendix
            and onset_start > 0
            and son_values[onset_start - 1] == RANK_FRICATIVE
            and son_values[onset_start] == RANK_STOP
        ):
            onset_start -= 1

        return onset_start


_DEFAULT_SCALE = SonorityScale()
_DEFAULT_SEGMENTER = MaxOnsetSegmenter()


def _assign_stress(
    syllables: list[Syllable],
    stress_marks: Sequence[tuple[int, str]],
) -> list[Syllable]:
    """Attach stress labels to syllables based on token-index marks.

    Each entry in *stress_marks* is ``(token_index, kind)`` where
    *token_index* is the position in the flat (stress-free) token list
    where the stress marker originally preceded, and *kind* is
    ``"primary"`` or ``"secondary"``.

    The marker attaches to whichever syllable *contains* that token
    index.  If multiple markers point to the same syllable, primary
    wins.
    """
    if not stress_marks:
        return syllables

    syl_stress: dict[int, str] = {}
    offset = 0
    for syl_idx, syl in enumerate(syllables):
        syl_len = len(syl)
        for tok_idx, kind in stress_marks:
            if offset <= tok_idx < offset + syl_len:
                existing = syl_stress.get(syl_idx)
                if existing != "primary":
                    syl_stress[syl_idx] = kind
        offset += syl_len

    if not syl_stress:
        return syllables

    return [
        dataclasses.replace(syl, stress=syl_stress.get(i)) if i in syl_stress else syl
        for i, syl in enumerate(syllables)
    ]


def syllabify(
    tokens: Sequence[str],
    vowels: Union[frozenset[str], set[str]],
    *,
    encoder: UniversalFeatureEncoder | None = None,
    strategy: SyllabificationStrategy | None = None,
    sonority_scale: SonorityScale | None = None,
    stress_marks: Sequence[tuple[int, str]] | None = None,
) -> list[Syllable]:
    """Segment IPA tokens into syllables using the configured strategy (default: MOP)."""
    scale = sonority_scale or _DEFAULT_SCALE
    seg = strategy or _DEFAULT_SEGMENTER
    vowel_fs = frozenset(vowels) if not isinstance(vowels, frozenset) else vowels

    # Cython fast path: bypass Python-level sonority lookup
    if _HAS_CYTHON_SYLLABLE and isinstance(seg, MaxOnsetSegmenter):
        son_map = scale.build_rank_map(set(tokens))
        raw = _cy_syllabify(
            list(tokens),
            vowel_fs,
            son_map,
            seg._sibilant_appendix,
        )
        result = [Syllable(onset=tuple(o), nucleus=tuple(n), coda=tuple(c)) for o, n, c in raw]
    else:
        son = scale.rank_tokens(tokens)
        result = seg.syllabify(tokens, vowel_fs, son)

    if stress_marks:
        result = _assign_stress(result, stress_marks)
    return result


def batch_syllabify(
    token_lists: Sequence[Sequence[str]],
    vowels: Union[frozenset[str], set[str]],
    *,
    sonority_scale: SonorityScale | None = None,
    strategy: SyllabificationStrategy | None = None,
    stress_marks_list: Sequence[Sequence[tuple[int, str]] | None] | None = None,
) -> list[list[Syllable]]:
    """Syllabify a batch of token lists, dispatching to Cython when available."""
    scale = sonority_scale or _DEFAULT_SCALE
    seg = strategy or _DEFAULT_SEGMENTER
    vowel_fs = frozenset(vowels) if not isinstance(vowels, frozenset) else vowels

    # Cython fast path
    if _HAS_CYTHON_SYLLABLE and isinstance(seg, MaxOnsetSegmenter):
        all_phones: set[str] = set()
        for tl in token_lists:
            all_phones.update(tl)
        son_map = scale.build_rank_map(all_phones)

        raw_batch = _cy_batch_syllabify(
            [list(tl) for tl in token_lists],
            vowel_fs,
            son_map,
            seg._sibilant_appendix,
        )
        results = [
            [Syllable(onset=tuple(o), nucleus=tuple(n), coda=tuple(c)) for o, n, c in word]
            for word in raw_batch
        ]
    else:
        # Pure-Python fallback
        results = [
            syllabify(tl, vowel_fs, sonority_scale=scale, strategy=seg) for tl in token_lists
        ]

    if stress_marks_list is not None:
        for i, marks in enumerate(stress_marks_list):
            if marks:
                results[i] = _assign_stress(results[i], marks)

    return results


# Stress query helpers
def stressed_syllable(
    syllables: Sequence[Syllable],
    kind: str = "primary",
) -> Syllable | None:
    """Return the first syllable with stress *kind* (``"primary"`` or ``"secondary"``), or None."""
    for syl in syllables:
        if syl.stress == kind:
            return syl
    return None


def stress_pattern(syllables: Sequence[Syllable]) -> str:
    """Return a numeric stress string: ``"1"`` primary, ``"2"`` secondary, ``"0"`` unstressed per syllable."""
    codes = {"primary": "1", "secondary": "2"}
    return "".join(codes.get(syl.stress, "0") for syl in syllables)  # type: ignore[arg-type]


def syllable_count(syllables: Sequence[Syllable]) -> int:
    """Return the number of syllables.

    A thin convenience wrapper for ``len(syllables)`` that reads more
    clearly in pipelines.
    """
    return len(syllables)
