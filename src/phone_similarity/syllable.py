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
  implementing :class:`SyllabificationStrategy`` without modifying
  existing code.
* **Dependency Inversion**: the module depends on the abstract
  ``SyllabificationStrategy`` protocol, not on concrete segmenters.

Stress preservation
-------------------
:class:`Syllable` carries an optional ``stress`` field (``"primary"``,
``"secondary"``, or ``None``).  When *ipa_with_stress* is passed to
:func:`syllabify`, stress markers ``ˈ``/``ˌ`` are extracted before
segmentation and then assigned to the syllable whose onset or nucleus
they precede.  Helper functions :func:`stressed_syllable` and
:func:`stress_pattern` provide quick access to prosodic structure.

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

    # Stress-aware syllabification:
    syllables = syllabify(
        ["h", "ɛ", "l", "oʊ"],
        vowels=lang.VOWELS_SET,
        stress_marks=[(0, "primary")],  # stress on first syllable
    )
    # syllables[0].stress == "primary"
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import ClassVar, Protocol, Union

from phone_similarity._dispatch import (
    HAS_CYTHON_SYLLABIFIER as _HAS_CYTHON_SYLLABLE,
)
from phone_similarity._dispatch import (
    cy_batch_syllabify as _cy_batch_syllabify,
)
from phone_similarity._dispatch import (
    cy_syllabify as _cy_syllabify,
)
from phone_similarity.universal_features import (
    UniversalFeatureEncoder,
)

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

    All three structural fields are tuples of IPA phoneme strings.  An
    empty tuple means the component is absent (e.g. a vowel-initial
    syllable has an empty onset).

    Parameters
    ----------
    onset : tuple of str
        Consonant phonemes before the nucleus.
    nucleus : tuple of str
        Vowel phoneme(s) forming the syllable peak.
    coda : tuple of str
        Consonant phonemes after the nucleus.
    stress : str or None
        ``"primary"``, ``"secondary"``, or ``None``.  Populated when
        :func:`syllabify` is called with *stress_marks* or when the
        input IPA string contains stress markers and
        ``preserve_stress=True`` was used.
    """

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


# ---------------------------------------------------------------------------
# Module-level default instances
# ---------------------------------------------------------------------------
_DEFAULT_SCALE = SonorityScale()
_DEFAULT_SEGMENTER = MaxOnsetSegmenter()


# ---------------------------------------------------------------------------
# Stress assignment helper
# ---------------------------------------------------------------------------
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
    stress_marks: Sequence[tuple[int, str]] | None = None,
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
        IPA phoneme tokens (stress markers should already be stripped;
        use *stress_marks* to carry stress information through).
    vowels : set or frozenset of str
        The vowel inventory.
    encoder : UniversalFeatureEncoder, optional
        Feature encoder for sonority derivation.
    strategy : SyllabificationStrategy, optional
        Segmentation algorithm (default: MOP with sibilant appendix).
    sonority_scale : SonorityScale, optional
        Pre-built sonority scale.
    stress_marks : sequence of (int, str), optional
        Stress positions as ``(token_index, kind)`` pairs, where *kind*
        is ``"primary"`` or ``"secondary"``.  Obtained from
        :func:`~phone_similarity.clean_phones.extract_stress_marks` or
        manually specified.

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
    stress_marks_list : sequence of (marks or None), optional
        Per-word stress marks.  If supplied, must have the same length
        as *token_lists*.  ``None`` entries mean no stress for that word.

    Returns
    -------
    list of list of Syllable
    """
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


# ---------------------------------------------------------------------------
# Stress query helpers
# ---------------------------------------------------------------------------
def stressed_syllable(
    syllables: Sequence[Syllable],
    kind: str = "primary",
) -> Syllable | None:
    """Return the first syllable with the given stress kind.

    Parameters
    ----------
    syllables : sequence of Syllable
        Output of :func:`syllabify`.
    kind : str
        ``"primary"`` (default) or ``"secondary"``.

    Returns
    -------
    Syllable or None
        The stressed syllable, or ``None`` if no match.
    """
    for syl in syllables:
        if syl.stress == kind:
            return syl
    return None


def stress_pattern(syllables: Sequence[Syllable]) -> str:
    """Return a numeric stress pattern string.

    Each syllable is represented by one character:
    * ``"1"`` — primary stress
    * ``"2"`` — secondary stress
    * ``"0"`` — unstressed

    Parameters
    ----------
    syllables : sequence of Syllable
        Output of :func:`syllabify`.

    Returns
    -------
    str
        E.g. ``"100"`` for a three-syllable word with initial stress.

    Examples
    --------
    >>> stress_pattern([
    ...     Syllable(("b",), ("ɪ",), (), stress="secondary"),
    ...     Syllable(("n",), ("æ",), (), stress=None),
    ...     Syllable(("n",), ("ə",), (), stress="primary"),
    ... ])
    '201'
    """
    codes = {"primary": "1", "secondary": "2"}
    return "".join(codes.get(syl.stress, "0") for syl in syllables)  # type: ignore[arg-type]


def syllable_count(syllables: Sequence[Syllable]) -> int:
    """Return the number of syllables.

    A thin convenience wrapper for ``len(syllables)`` that reads more
    clearly in pipelines.
    """
    return len(syllables)
