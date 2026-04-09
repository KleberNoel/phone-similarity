"""
Syllable segmentation with Maximum Onset Principle.

Provides language-aware syllabification using a sonority hierarchy
derived from universal phonological features.  Follows the **Strategy**
pattern for extensibility — the default :class:`MaxOnsetSegmenter` can
be replaced with language-specific implementations.

Design
------
* **Single Responsibility**: each class has one job — ``SonorityScale``
  maps phonemes to sonority ranks, ``MaxOnsetSegmenter`` applies the
  splitting algorithm, and :func:`syllabify` is the high-level facade.
* **Open/Closed**: new segmentation strategies can be added by
  implementing :class:`SyllabificationStrategy` without modifying
  existing code.
* **Dependency Inversion**: the module depends on the abstract
  ``SyllabificationStrategy`` protocol, not on concrete segmenters.

Cython acceleration
-------------------
When the ``_core`` extension is available, :func:`syllabify` and
:func:`batch_syllabify` dispatch to C-level implementations that avoid
per-phoneme Python dict lookups.

Usage::

    from phone_similarity.syllable import syllabify, Syllable
    from phone_similarity.language import LANGUAGES

    lang = LANGUAGES["eng_us"]
    tokens = ["s", "t", "r", "ɪ", "ŋ", "z"]
    syllables = syllabify(tokens, vowels=lang.VOWELS_SET)
    # [Syllable(onset=('s', 't', 'r'), nucleus=('ɪ',), coda=('ŋ', 'z'))]
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, Sequence, Union

from phone_similarity.universal_features import (
    PANPHON_FEATURE_NAMES,
    UniversalFeatureEncoder,
)

# ---------------------------------------------------------------------------
# Cython dispatch
# ---------------------------------------------------------------------------
try:
    from phone_similarity._core import (
        cython_syllabify as _cy_syllabify,
    )
    from phone_similarity._core import (
        batch_cython_syllabify as _cy_batch_syllabify,
    )

    _HAS_CYTHON_SYLLABLE = True
except ImportError:
    _HAS_CYTHON_SYLLABLE = False


# ---------------------------------------------------------------------------
# Sonority ranks
# ---------------------------------------------------------------------------
RANK_STOP = 1
RANK_FRICATIVE = 2
RANK_NASAL = 3
RANK_LIQUID = 4
RANK_GLIDE = 5
RANK_VOWEL = 6
RANK_UNKNOWN = 0


# ---------------------------------------------------------------------------
# Syllable data class
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Syllable:
    """A syllable decomposed into onset, nucleus, and coda.

    All three fields are tuples of IPA phoneme strings.  An empty tuple
    means the component is absent (e.g. a vowel-initial syllable has an
    empty onset).
    """

    onset: tuple[str, ...]
    nucleus: tuple[str, ...]
    coda: tuple[str, ...]

    @property
    def phonemes(self) -> tuple[str, ...]:
        """All phonemes in order: onset + nucleus + coda."""
        return self.onset + self.nucleus + self.coda

    def __len__(self) -> int:
        return len(self.onset) + len(self.nucleus) + len(self.coda)


# ---------------------------------------------------------------------------
# Sonority scale
# ---------------------------------------------------------------------------
class SonorityScale:
    """Maps IPA phonemes to integer sonority ranks via universal features.

    Ranks (highest to lowest):
        6  vowel       (syl=+1)
        5  glide       (son=+1, cons=-1, syl=-1)
        4  liquid      (son=+1, cons=+1, nas=-1)
        3  nasal       (son=+1, nas=+1)
        2  fricative   (son=-1, cont=+1)
        1  stop/affr.  (son=-1, cont=-1)
        0  unknown

    Parameters
    ----------
    extra_ranks : dict, optional
        Manual overrides ``{phoneme: rank}``.  Useful for phonemes
        absent from the Panphon table or for language-specific tweaks.
    """

    _cache: dict[str, int] = {}

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


# ---------------------------------------------------------------------------
# Segmentation strategy protocol
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Maximum Onset segmenter (concrete strategy)
# ---------------------------------------------------------------------------
class MaxOnsetSegmenter:
    """Syllabification using the Maximum Onset Principle (MOP).

    For each inter-vocalic consonant cluster, assigns as many consonants
    as possible to the *onset* of the following syllable, subject to the
    Sonority Sequencing Principle (SSP): onsets must have rising
    sonority towards the nucleus.

    Parameters
    ----------
    sibilant_appendix : bool
        If *True* (default), allow a sibilant (rank 2) to attach to
        the left of an onset even when it would violate the SSP.  This
        covers the well-known /s/-exception in English ``str-``, ``sp-``,
        ``sk-`` clusters.
    """

    def __init__(self, sibilant_appendix: bool = True) -> None:
        self._sibilant_appendix = sibilant_appendix

    def syllabify(
        self,
        tokens: Sequence[str],
        vowels: frozenset[str],
        sonority: Sequence[int],
    ) -> list[Syllable]:
        """Segment *tokens* into syllables.

        Parameters
        ----------
        tokens : sequence of str
            IPA phoneme tokens (from an IPA tokeniser).
        vowels : frozenset of str
            The vowel inventory of the language.
        sonority : sequence of int
            Pre-computed sonority rank for each token position.

        Returns
        -------
        list of Syllable
        """
        n = len(tokens)
        if n == 0:
            return []

        # --- Phase 1: identify vowel-span boundaries -----------------------
        # spans = list of (start, end) index pairs for each vowel nucleus
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

        # --- Phase 2: split inter-vocalic clusters --------------------------
        syllables: list[Syllable] = []

        for idx, (v_start, v_end) in enumerate(spans):
            if idx == 0:
                # Leading consonants → onset of first syllable
                onset = tuple(tokens[:v_start])
            else:
                # Consonant cluster between previous nucleus end and this
                prev_end = spans[idx - 1][1]
                cluster = tokens[prev_end:v_start]
                split = self._split_cluster(cluster, sonority[prev_end:v_start])
                # Attach coda to previous syllable
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
                # Trailing consonants → coda of last syllable
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


# ---------------------------------------------------------------------------
# Module-level default instances
# ---------------------------------------------------------------------------
_DEFAULT_SCALE = SonorityScale()
_DEFAULT_SEGMENTER = MaxOnsetSegmenter()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def syllabify(
    tokens: Sequence[str],
    vowels: Union[frozenset[str], set[str]],
    *,
    encoder: UniversalFeatureEncoder | None = None,
    strategy: SyllabificationStrategy | None = None,
    sonority_scale: SonorityScale | None = None,
) -> list[Syllable]:
    """Segment IPA tokens into syllables.

    This is the main entry point.  It computes sonority ranks from
    universal features and delegates to the chosen strategy (default:
    :class:`MaxOnsetSegmenter`).

    When the Cython extension is available the hot path is dispatched to
    C for ~3-5x speedup on batch workloads.

    Parameters
    ----------
    tokens : sequence of str
        IPA phoneme tokens.
    vowels : set or frozenset of str
        The vowel inventory.
    encoder : UniversalFeatureEncoder, optional
        Feature encoder for sonority derivation.
    strategy : SyllabificationStrategy, optional
        Segmentation algorithm (default: MOP with sibilant appendix).
    sonority_scale : SonorityScale, optional
        Pre-built sonority scale.

    Returns
    -------
    list of Syllable
    """
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
        return [Syllable(onset=tuple(o), nucleus=tuple(n), coda=tuple(c)) for o, n, c in raw]

    son = scale.rank_tokens(tokens)
    return seg.syllabify(tokens, vowel_fs, son)


def batch_syllabify(
    token_lists: Sequence[Sequence[str]],
    vowels: Union[frozenset[str], set[str]],
    *,
    sonority_scale: SonorityScale | None = None,
    strategy: SyllabificationStrategy | None = None,
) -> list[list[Syllable]]:
    """Syllabify a batch of token lists.

    When the Cython extension is available the entire batch is processed
    in one C call, avoiding per-word Python overhead.

    Parameters
    ----------
    token_lists : sequence of sequence of str
        Each inner sequence is a tokenised IPA word.
    vowels : set or frozenset of str
        The vowel inventory.
    sonority_scale : SonorityScale, optional
    strategy : SyllabificationStrategy, optional

    Returns
    -------
    list of list of Syllable
    """
    scale = sonority_scale or _DEFAULT_SCALE
    seg = strategy or _DEFAULT_SEGMENTER
    vowel_fs = frozenset(vowels) if not isinstance(vowels, frozenset) else vowels

    # Cython fast path
    if _HAS_CYTHON_SYLLABLE and isinstance(seg, MaxOnsetSegmenter):
        # Build a single sonority map over the union of all inventories
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
        return [
            [Syllable(onset=tuple(o), nucleus=tuple(n), coda=tuple(c)) for o, n, c in word]
            for word in raw_batch
        ]

    # Pure-Python fallback
    return [syllabify(tl, vowel_fs, sonority_scale=scale, strategy=seg) for tl in token_lists]
