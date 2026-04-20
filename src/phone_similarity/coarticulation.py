"""
Co-articulation model with controlled randomness for phonological distance.

In natural speech, phonemes are not produced in isolation.  Adjacent sounds
influence each other through **co-articulation**: anticipatory (look-ahead)
and carryover (perseverative) effects that shift articulatory features.
This module models those effects as feature perturbations on the 24-feature
Panphon vectors, with optional stochastic jitter.

Co-articulation effects modelled
---------------------------------
1. **Anticipatory vowel-to-consonant**: consonants shift features toward a
   following vowel (palatalization, lip rounding, backing).
2. **Carryover consonant-to-vowel**: vowels shift features due to a
   preceding consonant (nasalization, backing, rounding).
3. **Consonant cluster assimilation**: voicing and place assimilation in
   adjacent consonant sequences.
4. **Syllable boundary attenuation**: effects are stronger within a syllable
   than across syllable boundaries (configurable decay factor).
5. **Fricative-specific weighting**: :class:`FricativeConfig` controls how
   frication-related features (``cont``, ``strid``) are weighted in distance
   computation, and optionally models frication noise spreading to adjacent
   segments.

Design
------
* **Strategy pattern**: :class:`CoarticulationStrategy` is the abstract
  protocol; :class:`DefaultCoarticulationModel` is the concrete strategy.
  Custom models (e.g. language-specific) can be plugged in.
* **Single Responsibility**: this module handles only co-articulation
  perturbation.  Distance computation remains in :mod:`primitives`.
* **Open/Closed**: new co-articulation rules can be added by subclassing
  :class:`DefaultCoarticulationModel` without modifying existing code.

Usage::

    from phone_similarity.coarticulation import (
        DefaultCoarticulationModel,
        FricativeConfig,
        coarticulated_feature_edit_distance,
    )
    from phone_similarity.universal_features import UniversalFeatureEncoder

    model = DefaultCoarticulationModel(jitter=0.3, seed=42)
    tokens = ["k", "æ", "t"]
    perturbed = model.perturb_sequence(tokens)
    # perturbed[i] is a 24-float tuple representing the co-articulated features

    # Or use the high-level distance function directly:
    dist = coarticulated_feature_edit_distance(
        ["k", "æ", "t"], ["k", "æ", "b"],
        jitter=0.3,
    )

    # With custom fricative weighting (penalise frication differences 2x):
    fc = FricativeConfig(fricative_weight=2.0, sibilant_weight=1.5)
    model = DefaultCoarticulationModel(fricative_config=fc)
    dist = coarticulated_feature_edit_distance(
        ["s", "æ", "t"], ["θ", "æ", "t"],
        model=model,
    )
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Sequence
from typing import Protocol

# Cython dispatch (imported once; flag checked in hot functions)
from phone_similarity._dispatch import (
    HAS_CYTHON_COARTICULATION,
    cy_coarticulated_feature_edit_distance,
)
from phone_similarity.universal_features import (
    PANPHON_FEATURE_NAMES,
    UniversalFeatureEncoder,
)

# Constants
_NUM_FEATURES = len(PANPHON_FEATURE_NAMES)
_FEAT_IDX = {name: i for i, name in enumerate(PANPHON_FEATURE_NAMES)}

_SYL = _FEAT_IDX["syl"]
_SON = _FEAT_IDX["son"]
_CONS = _FEAT_IDX["cons"]
_CONT = _FEAT_IDX["cont"]
_NAS = _FEAT_IDX["nas"]
_VOI = _FEAT_IDX["voi"]
_ANT = _FEAT_IDX["ant"]
_COR = _FEAT_IDX["cor"]
_LAB = _FEAT_IDX["lab"]
_HI = _FEAT_IDX["hi"]
_LO = _FEAT_IDX["lo"]
_BACK = _FEAT_IDX["back"]
_ROUND = _FEAT_IDX["round"]
_STRID = _FEAT_IDX["strid"]
_DELREL = _FEAT_IDX["delrel"]

# Encode cache — avoids repeated NFD + diacritic-strip fallback per token
_encode = UniversalFeatureEncoder.encode
_encode_cache: dict[str, tuple[int, ...]] = {}


def _cached_encode(phoneme: str) -> tuple[int, ...]:
    """Cached wrapper around UniversalFeatureEncoder.encode()."""
    vec = _encode_cache.get(phoneme)
    if vec is None:
        vec = _encode(phoneme)
        _encode_cache[phoneme] = vec
    return vec


# FricativeConfig dataclass
@dataclasses.dataclass(frozen=True)
class FricativeConfig:
    """Configuration for fricative-specific weighting in distance computation.

    Fricatives are identified by ``[son=-1, cont=+1]``.  Sibilants (the
    *strident* subset — /s z ʃ ʒ/) additionally have ``[strid=+1]``, while
    non-sibilant fricatives (/f v θ ð/) have ``[strid=-1]``.

    This config lets the user amplify or attenuate the contribution of
    frication-related features (``cont`` and ``strid``) in the distance
    computation.  A ``fricative_weight`` of 2.0 means that when **either**
    phoneme in a comparison is a fricative, differences on ``cont`` and
    ``strid`` count twice as heavily as usual.

    Parameters
    ----------
    fricative_weight : float
        Multiplier applied to ``cont`` feature differences when at least one
        phoneme is a fricative.  Default 1.0 (no change).  Values >1.0
        penalise frication differences more heavily.
    sibilant_weight : float
        Multiplier applied to ``strid`` feature differences when at least one
        phoneme is a sibilant (strident fricative).  Default 1.0.  Separate
        from ``fricative_weight`` so users can independently weight the
        sibilant/non-sibilant distinction (e.g. /s/ vs /θ/).
    frication_spread : bool
        When True, enable co-articulation rules that model frication noise
        spreading to adjacent segments (partial devoicing of following
        vowels, partial frication bleed into neighbouring consonants).
    spread_magnitude : float
        Magnitude of frication spread effects in [0.0, 1.0].  Only used
        when ``frication_spread`` is True.  Default 0.30.
    """

    fricative_weight: float = 1.0
    sibilant_weight: float = 1.0
    frication_spread: bool = False
    spread_magnitude: float = 0.30

    def __post_init__(self) -> None:
        if self.fricative_weight < 0.0:
            raise ValueError(f"fricative_weight must be >= 0.0, got {self.fricative_weight}")
        if self.sibilant_weight < 0.0:
            raise ValueError(f"sibilant_weight must be >= 0.0, got {self.sibilant_weight}")
        if not 0.0 <= self.spread_magnitude <= 1.0:
            raise ValueError(
                f"spread_magnitude must be in [0.0, 1.0], got {self.spread_magnitude}"
            )


_DEFAULT_FRICATIVE_CONFIG = FricativeConfig()


# Co-articulation rule dataclass
@dataclasses.dataclass(frozen=True)
class CoarticulationRule:
    """A single co-articulation perturbation rule.

    Parameters
    ----------
    name : str
        Human-readable rule name (for debugging / logging).
    target_feature : int
        Index into the 24-feature vector to perturb.
    direction : float
        Target shift direction: +1.0 (toward +1) or -1.0 (toward -1).
    magnitude : float
        Maximum shift amount in [0.0, 1.0].  The actual perturbation is
        ``direction * magnitude * activation``.
    base_probability : float
        Probability that this rule fires at all (before jitter scaling).
    within_syllable_only : bool
        If True, the rule only fires when source and target are in the
        same syllable.
    """

    name: str
    target_feature: int
    direction: float
    magnitude: float
    base_probability: float = 0.8
    within_syllable_only: bool = False


# Strategy protocol
class CoarticulationStrategy(Protocol):
    """Abstract protocol for co-articulation models."""

    def perturb_sequence(
        self,
        tokens: Sequence[str],
        syllable_boundaries: Sequence[int] | None = None,
    ) -> list[tuple[float, ...]]:
        """Return perturbed 24-float feature vectors for each token."""
        ...


# Default co-articulation model
class DefaultCoarticulationModel:
    """Co-articulation model with deterministic rules and stochastic jitter.

    The model applies established phonological co-articulation effects to
    sequences of IPA phonemes, producing perturbed feature vectors that
    reflect how phonemes are actually realized in connected speech.

    Parameters
    ----------
    jitter : float
        Controls the amount of stochastic perturbation.  0.0 means
        deterministic (all rules fire at full ``base_probability``);
        1.0 means maximum randomness (rules fire probabilistically and
        magnitudes are randomly scaled).  Values in between interpolate.
    seed : int or None
        Random seed for reproducibility.  ``None`` means non-deterministic.
    cross_syllable_decay : float
        Multiplicative decay for co-articulation effects that cross a
        syllable boundary.  Default 0.4 means effects are 40% as strong
        across boundaries as within a syllable.
    """

    # Anticipatory rules: consonant features shift toward following vowel
    # Keyed by trigger condition for O(1) lookup instead of linear scan.
    _ANTICIPATORY_RULES: dict[str, list[CoarticulationRule]] = {
        "high_vowel": [
            CoarticulationRule("palatal_raise", _HI, +1.0, 0.35, 0.85),
        ],
        "round_vowel": [
            CoarticulationRule("lip_rounding", _ROUND, +1.0, 0.40, 0.80),
        ],
        "back_vowel": [
            CoarticulationRule("backing", _BACK, +1.0, 0.30, 0.75),
        ],
    }

    # Carryover rules: vowel features shift due to preceding consonant
    _CARRYOVER_RULES: dict[str, list[CoarticulationRule]] = {
        "nasal_consonant": [
            CoarticulationRule("nasalization", _NAS, +1.0, 0.50, 0.90),
        ],
        "velar_consonant": [
            CoarticulationRule("velar_backing", _BACK, +1.0, 0.25, 0.70),
            CoarticulationRule("velar_raising", _HI, +1.0, 0.20, 0.65),
        ],
        "labial_consonant": [
            CoarticulationRule("labial_rounding", _ROUND, +1.0, 0.30, 0.70),
        ],
    }

    # Assimilation rules: consonant-consonant cluster effects
    _ASSIMILATION_VOICING: list[CoarticulationRule] = [
        CoarticulationRule("voice_spread", _VOI, 0.0, 0.40, 0.75),
    ]

    _ASSIMILATION_PLACE: list[CoarticulationRule] = [
        CoarticulationRule("nasal_place_ant", _ANT, 0.0, 0.55, 0.85),
        CoarticulationRule("nasal_place_cor", _COR, 0.0, 0.55, 0.85),
        CoarticulationRule("nasal_place_lab", _LAB, 0.0, 0.55, 0.85),
    ]

    # Frication spread rules: fricative noise bleeds to adjacent segments
    # These only activate when FricativeConfig.frication_spread is True.
    # Note: vowels already have [cont=+1], so we model frication spread as:
    # 1. Partial devoicing of following vowels by voiceless fricatives
    # 2. Stridency spread from strident fricatives to adjacent segments
    _FRICATION_SPREAD_RULES: list[CoarticulationRule] = [
        # Voiceless fricative before vowel: partial devoicing
        CoarticulationRule("fric_devoice", _VOI, -1.0, 0.30, 0.75),
        # Sibilant spread: following segment picks up slight stridency
        CoarticulationRule("fric_spread_strid", _STRID, +1.0, 0.25, 0.60),
    ]

    def __init__(
        self,
        jitter: float = 0.0,
        seed: int | None = None,
        cross_syllable_decay: float = 0.4,
        fricative_config: FricativeConfig | None = None,
    ) -> None:
        if not 0.0 <= jitter <= 1.0:
            raise ValueError(f"jitter must be in [0.0, 1.0], got {jitter}")
        if not 0.0 <= cross_syllable_decay <= 1.0:
            raise ValueError(
                f"cross_syllable_decay must be in [0.0, 1.0], got {cross_syllable_decay}"
            )
        self._jitter = jitter
        self._rng = random.Random(seed)
        self._cross_syllable_decay = cross_syllable_decay
        self._fricative_config = fricative_config or _DEFAULT_FRICATIVE_CONFIG

    @property
    def fricative_config(self) -> FricativeConfig:
        """Current fricative configuration."""
        return self._fricative_config

    @fricative_config.setter
    def fricative_config(self, value: FricativeConfig) -> None:
        if not isinstance(value, FricativeConfig):
            raise TypeError(f"Expected FricativeConfig, got {type(value).__name__}")
        self._fricative_config = value

    @property
    def jitter(self) -> float:
        """Current jitter level."""
        return self._jitter

    @jitter.setter
    def jitter(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"jitter must be in [0.0, 1.0], got {value}")
        self._jitter = value

    @staticmethod
    def _is_vowel(feats: tuple[int, ...]) -> bool:
        return feats[_SYL] == 1

    @staticmethod
    def _is_consonant(feats: tuple[int, ...]) -> bool:
        return feats[_SYL] != 1

    @staticmethod
    def _is_high_vowel(feats: tuple[int, ...]) -> bool:
        return feats[_SYL] == 1 and feats[_HI] == 1

    @staticmethod
    def _is_round_vowel(feats: tuple[int, ...]) -> bool:
        return feats[_SYL] == 1 and feats[_ROUND] == 1

    @staticmethod
    def _is_back_vowel(feats: tuple[int, ...]) -> bool:
        return feats[_SYL] == 1 and feats[_BACK] == 1

    @staticmethod
    def _is_nasal(feats: tuple[int, ...]) -> bool:
        return feats[_NAS] == 1

    @staticmethod
    def _is_velar(feats: tuple[int, ...]) -> bool:
        """Velar: [+hi, +back, -cor] consonant (k, g, ŋ, x, ɣ)."""
        return feats[_SYL] != 1 and feats[_HI] == 1 and feats[_BACK] == 1

    @staticmethod
    def _is_labial(feats: tuple[int, ...]) -> bool:
        return feats[_LAB] == 1

    @staticmethod
    def _is_stop(feats: tuple[int, ...]) -> bool:
        return feats[_SON] == -1 and feats[_CONT] == -1

    @staticmethod
    def _is_fricative(feats: tuple[int, ...]) -> bool:
        """Fricative: [-sonorant, +continuant] (f, v, s, z, ʃ, ʒ, θ, ð, x, ɣ, ...)."""
        return feats[_SON] == -1 and feats[_CONT] == 1

    @staticmethod
    def _is_sibilant(feats: tuple[int, ...]) -> bool:
        """Sibilant: fricative with [+strident] (s, z, ʃ, ʒ, ...)."""
        return feats[_SON] == -1 and feats[_CONT] == 1 and feats[_STRID] == 1

    def _same_syllable(
        self,
        idx_a: int,
        idx_b: int,
        syl_boundaries: Sequence[int] | None,
    ) -> bool:
        """Check whether two token indices are in the same syllable."""
        if syl_boundaries is None:
            return True
        return syl_boundaries[idx_a] == syl_boundaries[idx_b]

    def _apply_shift(
        self,
        base: list[float],
        rule: CoarticulationRule,
        direction: float,
        same_syllable: bool,
    ) -> None:
        """Apply a single feature shift to a mutable feature vector.

        The shift amount is:
            direction * magnitude * boundary_factor * stochastic_scale

        where stochastic_scale depends on jitter.
        """
        prob = rule.base_probability
        boundary_factor = 1.0 if same_syllable else self._cross_syllable_decay

        if self._jitter > 0.0:
            fire_prob = prob * (1.0 - self._jitter) + prob * self._jitter * self._rng.random()
            if self._rng.random() > fire_prob:
                return
            mag = rule.magnitude * (1.0 - self._jitter + self._jitter * self._rng.random())
        else:
            mag = rule.magnitude

        shift = direction * mag * boundary_factor
        new_val = base[rule.target_feature] + shift
        base[rule.target_feature] = max(-1.0, min(1.0, new_val))

    def perturb_sequence(
        self,
        tokens: Sequence[str],
        syllable_boundaries: Sequence[int] | None = None,
    ) -> list[tuple[float, ...]]:
        """Compute co-articulated feature vectors for a phoneme sequence.

        Parameters
        ----------
        tokens : sequence of str
            IPA phoneme tokens.
        syllable_boundaries : sequence of int, optional
            Syllable index for each token position.  If provided, effects
            are attenuated across syllable boundaries.  Generate this from
            :func:`~phone_similarity.syllable.syllabify` output.

        Returns
        -------
        list of tuple of float
            Perturbed 24-float feature vectors, one per token.
        """
        n = len(tokens)
        if n == 0:
            return []

        raw_feats = [_cached_encode(tok) for tok in tokens]
        perturbed: list[list[float]] = [[float(v) for v in f] for f in raw_feats]

        _antic = self._ANTICIPATORY_RULES
        _carry = self._CARRYOVER_RULES
        _assim_voi = self._ASSIMILATION_VOICING
        _assim_place = self._ASSIMILATION_PLACE
        _fric_rules = self._FRICATION_SPREAD_RULES
        _fric_cfg = self._fricative_config
        _fric_spread = _fric_cfg.frication_spread
        _apply = self._apply_shift
        _same_syl = self._same_syllable

        for i in range(n):
            feats_i = raw_feats[i]

            if i + 1 < n:
                feats_next = raw_feats[i + 1]
                same_syl = _same_syl(i, i + 1, syllable_boundaries)

                if self._is_consonant(feats_i) and self._is_vowel(feats_next):
                    if self._is_high_vowel(feats_next):
                        for rule in _antic["high_vowel"]:
                            _apply(perturbed[i], rule, rule.direction, same_syl)
                    if self._is_round_vowel(feats_next):
                        for rule in _antic["round_vowel"]:
                            _apply(perturbed[i], rule, rule.direction, same_syl)
                    if self._is_back_vowel(feats_next):
                        for rule in _antic["back_vowel"]:
                            _apply(perturbed[i], rule, rule.direction, same_syl)

                if _fric_spread and self._is_fricative(feats_i) and self._is_vowel(feats_next):
                    mag_scale = _fric_cfg.spread_magnitude / 0.30
                    is_sib = self._is_sibilant(feats_i)
                    is_voiceless = feats_i[_VOI] == -1
                    for rule in _fric_rules:
                        if rule.target_feature == _STRID and not is_sib:
                            continue
                        if rule.target_feature == _VOI and not is_voiceless:
                            continue
                        scaled_rule = CoarticulationRule(
                            name=rule.name,
                            target_feature=rule.target_feature,
                            direction=rule.direction,
                            magnitude=rule.magnitude * mag_scale,
                            base_probability=rule.base_probability,
                            within_syllable_only=rule.within_syllable_only,
                        )
                        _apply(perturbed[i + 1], scaled_rule, rule.direction, same_syl)

                if self._is_consonant(feats_i) and self._is_consonant(feats_next):
                    voi_next = feats_next[_VOI]
                    if voi_next != 0:
                        direction = float(voi_next)
                        for rule in _assim_voi:
                            _apply(perturbed[i], rule, direction, same_syl)

                    if self._is_nasal(feats_i) and self._is_stop(feats_next):
                        for rule in _assim_place:
                            target_val = feats_next[rule.target_feature]
                            if target_val != 0:
                                _apply(
                                    perturbed[i],
                                    rule,
                                    float(target_val),
                                    same_syl,
                                )

            if i > 0:
                feats_prev = raw_feats[i - 1]
                same_syl = _same_syl(i - 1, i, syllable_boundaries)

                if self._is_vowel(feats_i) and self._is_consonant(feats_prev):
                    if self._is_nasal(feats_prev):
                        for rule in _carry["nasal_consonant"]:
                            _apply(perturbed[i], rule, rule.direction, same_syl)
                    if self._is_velar(feats_prev):
                        for rule in _carry["velar_consonant"]:
                            _apply(perturbed[i], rule, rule.direction, same_syl)
                    if self._is_labial(feats_prev):
                        for rule in _carry["labial_consonant"]:
                            _apply(perturbed[i], rule, rule.direction, same_syl)

                    if _fric_spread and self._is_fricative(feats_prev):
                        mag_scale = _fric_cfg.spread_magnitude / 0.30
                        is_sib_prev = self._is_sibilant(feats_prev)
                        is_voiceless_prev = feats_prev[_VOI] == -1
                        for rule in _fric_rules:
                            if rule.target_feature == _STRID and not is_sib_prev:
                                continue
                            if rule.target_feature == _VOI and not is_voiceless_prev:
                                continue
                            scaled_rule = CoarticulationRule(
                                name=rule.name + "_carry",
                                target_feature=rule.target_feature,
                                direction=rule.direction,
                                magnitude=rule.magnitude * mag_scale * 0.6,
                                base_probability=rule.base_probability,
                                within_syllable_only=rule.within_syllable_only,
                            )
                            _apply(perturbed[i], scaled_rule, rule.direction, same_syl)

        return [tuple(v) for v in perturbed]

    @staticmethod
    def syllable_boundary_map(
        syllables: Sequence[object],
    ) -> list[int]:
        """Convert a list of Syllable objects to a per-token syllable index.

        Parameters
        ----------
        syllables : sequence of Syllable
            Output from :func:`~phone_similarity.syllable.syllabify`.

        Returns
        -------
        list of int
            ``boundaries[i]`` is the syllable index that token ``i``
            belongs to.
        """
        boundaries: list[int] = []
        for syl_idx, syl in enumerate(syllables):
            boundaries.extend([syl_idx] * len(syl))
        return boundaries


# Module-level convenience instance
_DEFAULT_MODEL = DefaultCoarticulationModel(jitter=0.0)


def _is_fricative_vec(vec: tuple[float, ...]) -> bool:
    """Check if a float feature vector represents a fricative [son<0, cont>0]."""
    return vec[_SON] < -0.5 and vec[_CONT] > 0.5


def _is_sibilant_vec(vec: tuple[float, ...]) -> bool:
    """Check if a float vector is a sibilant [son<0, cont>0, strid>0]."""
    return vec[_SON] < -0.5 and vec[_CONT] > 0.5 and vec[_STRID] > 0.5


def coarticulated_phoneme_distance(
    vec_a: tuple[float, ...],
    vec_b: tuple[float, ...],
    fricative_config: FricativeConfig | None = None,
    *,
    _fric_flags: tuple[bool, bool] | None = None,
) -> float:
    """Distance between two co-articulated feature vectors, normalised to [0, 1].

    Uses the same logic as ``universal_phoneme_distance`` but operates on
    float vectors (because co-articulation produces continuous values).
    Features where either vector has value 0.0 are excluded.

    When a :class:`FricativeConfig` is provided and at least one of the
    phonemes is a fricative, the ``cont`` and ``strid`` feature differences
    are multiplied by the configured weights, giving the user fine-grained
    control over how much frication matters in comparisons.

    Parameters
    ----------
    vec_a, vec_b : tuple of float
        24-float co-articulated feature vectors.
    fricative_config : FricativeConfig, optional
        If provided, apply weighted distances for frication-related features
        when at least one phoneme is a fricative.
    _fric_flags : tuple of (bool, bool), optional
        Pre-computed ``(either_fricative, either_sibilant)`` flags.  Internal
        optimisation to avoid re-classifying vectors when the caller already
        knows the flags (e.g. the DP loop pre-computes them).

    Returns
    -------
    float
        Normalised distance in [0.0, 1.0] (may exceed 1.0 with very high
        weights, though in practice it stays close to [0, 1]).
    """
    fc = fricative_config or _DEFAULT_FRICATIVE_CONFIG
    if _fric_flags is not None:
        either_fric, either_sib = _fric_flags
    else:
        either_fric = _is_fricative_vec(vec_a) or _is_fricative_vec(vec_b)
        either_sib = _is_sibilant_vec(vec_a) or _is_sibilant_vec(vec_b)

    fric_w = fc.fricative_weight
    sib_w = fc.sibilant_weight

    comparable = 0
    total_diff = 0.0
    for idx in range(_NUM_FEATURES):
        a_val = vec_a[idx]
        b_val = vec_b[idx]
        if -0.01 < a_val < 0.01 and -0.01 < b_val < 0.01:
            continue
        comparable += 1
        diff = abs(a_val - b_val) * 0.5

        if either_fric and idx == _CONT:
            diff *= fric_w
        elif either_sib and idx == _STRID:
            diff *= sib_w

        total_diff += diff
    if comparable == 0:
        return 0.0
    return total_diff / comparable


def coarticulated_feature_edit_distance(
    seq_a: Sequence[str],
    seq_b: Sequence[str],
    *,
    model: DefaultCoarticulationModel | None = None,
    fricative_config: FricativeConfig | None = None,
    syl_boundaries_a: Sequence[int] | None = None,
    syl_boundaries_b: Sequence[int] | None = None,
    insert_cost: float = 1.0,
    delete_cost: float = 1.0,
) -> float:
    """Feature edit distance with co-articulation perturbation.

    Like :func:`~phone_similarity.primitives.feature_edit_distance`, but
    substitution costs are computed on co-articulated feature vectors
    that account for phonetic context.

    Parameters
    ----------
    seq_a, seq_b : sequence of str
        IPA phoneme sequences.
    model : DefaultCoarticulationModel, optional
        Co-articulation model.  Defaults to a deterministic model
        (jitter=0.0).
    fricative_config : FricativeConfig, optional
        Fricative-specific weighting.  If ``None``, uses the model's
        ``fricative_config`` attribute (which defaults to no weighting).
    syl_boundaries_a, syl_boundaries_b : sequence of int, optional
        Per-token syllable indices for each sequence.
    insert_cost, delete_cost : float
        Indel costs (default 1.0).

    Returns
    -------
    float
        Minimum-cost alignment distance.
    """
    m = model or _DEFAULT_MODEL
    fc = fricative_config or m.fricative_config
    vecs_a = m.perturb_sequence(seq_a, syl_boundaries_a)
    vecs_b = m.perturb_sequence(seq_b, syl_boundaries_b)

    if HAS_CYTHON_COARTICULATION:
        return cy_coarticulated_feature_edit_distance(
            vecs_a,
            vecs_b,
            fc.fricative_weight,
            fc.sibilant_weight,
            _SON,
            _CONT,
            _STRID,
            insert_cost,
            delete_cost,
        )

    len_a = len(vecs_a)
    len_b = len(vecs_b)

    fric_a = [_is_fricative_vec(v) for v in vecs_a]
    fric_b = [_is_fricative_vec(v) for v in vecs_b]
    sib_a = [_is_sibilant_vec(v) for v in vecs_a]
    sib_b = [_is_sibilant_vec(v) for v in vecs_b]

    prev = [insert_cost * j for j in range(len_b + 1)]
    curr = [0.0] * (len_b + 1)

    for i in range(1, len_a + 1):
        curr[0] = prev[0] + delete_cost
        va = vecs_a[i - 1]
        fa = fric_a[i - 1]
        sa = sib_a[i - 1]

        for j in range(1, len_b + 1):
            flags = (fa or fric_b[j - 1], sa or sib_b[j - 1])
            sub_cost = coarticulated_phoneme_distance(
                va,
                vecs_b[j - 1],
                fricative_config=fc,
                _fric_flags=flags,
            )
            d = prev[j] + delete_cost
            ins = curr[j - 1] + insert_cost
            sub = prev[j - 1] + sub_cost
            if ins < d:
                d = ins
            if sub < d:
                d = sub
            curr[j] = d

        prev, curr = curr, prev

    return prev[len_b]


def normalised_coarticulated_feature_edit_distance(
    seq_a: Sequence[str],
    seq_b: Sequence[str],
    *,
    model: DefaultCoarticulationModel | None = None,
    fricative_config: FricativeConfig | None = None,
    syl_boundaries_a: Sequence[int] | None = None,
    syl_boundaries_b: Sequence[int] | None = None,
    insert_cost: float = 1.0,
    delete_cost: float = 1.0,
) -> float:
    """Co-articulated feature edit distance normalised to [0, 1].

    Returns 0.0 when both sequences are empty.
    """
    max_len = max(len(seq_a), len(seq_b))
    if max_len == 0:
        return 0.0
    raw = coarticulated_feature_edit_distance(
        seq_a,
        seq_b,
        model=model,
        fricative_config=fricative_config,
        syl_boundaries_a=syl_boundaries_a,
        syl_boundaries_b=syl_boundaries_b,
        insert_cost=insert_cost,
        delete_cost=delete_cost,
    )
    return raw / max_len
