"""
Feature-vector inversion: reverse lookup from phonological features to phonemes.

Given a feature vector (e.g. ``{"voiced": True, "place": "alveolar"}``),
find the closest phonemes in a target language's inventory.  Also provides
:func:`invert_ipa` which maps every phoneme in an IPA string to its closest
target-language equivalents.
"""

from __future__ import annotations

from typing import Union

from phone_similarity.base_bit_array_specification import BaseBitArraySpecification
from phone_similarity.primitives import (
    _HAS_CYTHON_EXT,
    phoneme_feature_distance,
)

if _HAS_CYTHON_EXT:
    from phone_similarity.primitives import _c_invert_features


def invert_features(
    feature_vector: dict[str, Union[bool, str]],
    target_phoneme_features: dict[str, dict[str, Union[bool, str]]],
    *,
    top_n: int = 0,
    max_distance: float = 1.0,
) -> list[tuple[str, float]]:
    """Find phonemes in a target inventory closest to a feature vector.

    This is the *reverse* of the normal lookup: instead of
    ``phoneme -> features``, we go ``features -> ranked phonemes``.

    Parameters
    ----------
    feature_vector : dict
        A single phoneme's feature dictionary (e.g.
        ``{"voiced": True, "place": "alveolar", "manner": "plosive"}``).
    target_phoneme_features : dict
        The target language's ``PHONEME_FEATURES`` mapping
        ``{phoneme: {feature: value, ...}, ...}``.
    top_n : int, optional
        If > 0, return only the *top_n* closest phonemes.  0 (default)
        means return all phonemes within *max_distance*.
    max_distance : float, optional
        Exclude phonemes whose distance exceeds this value (default 1.0,
        i.e. keep everything).

    Returns
    -------
    list of (phoneme, distance)
        Phonemes sorted by ascending distance from *feature_vector*.
    """
    if _HAS_CYTHON_EXT:
        return _c_invert_features(feature_vector, target_phoneme_features, top_n, max_distance)
    ranked: list[tuple[str, float]] = []
    for phoneme, feats in target_phoneme_features.items():
        d = phoneme_feature_distance(feature_vector, feats)
        if d <= max_distance:
            ranked.append((phoneme, d))
    ranked.sort(key=lambda t: t[1])
    if top_n > 0:
        ranked = ranked[:top_n]
    return ranked


def invert_ipa(
    source_ipa: str,
    source_spec: BaseBitArraySpecification,
    source_phoneme_features: dict[str, dict[str, Union[bool, str]]],
    target_phoneme_features: dict[str, dict[str, Union[bool, str]]],
    *,
    top_n: int = 3,
    max_distance: float = 0.6,
) -> list[tuple[str, list[tuple[str, float]]]]:
    """Map each phoneme in a source IPA string to its closest target-language phonemes.

    Tokenises *source_ipa* using *source_spec*, looks up each token's
    features in *source_phoneme_features*, then calls
    :func:`invert_features` against the *target_phoneme_features* inventory.

    Parameters
    ----------
    source_ipa : str
        IPA transcription in the source language.
    source_spec : BaseBitArraySpecification
        Specification for tokenising *source_ipa*.
    source_phoneme_features : dict
        ``PHONEME_FEATURES`` of the source language.
    target_phoneme_features : dict
        ``PHONEME_FEATURES`` of the target language.
    top_n : int
        Max candidates per source phoneme (default 3).
    max_distance : float
        Ignore target phonemes further than this (default 0.6).

    Returns
    -------
    list of (source_phoneme, [(target_phoneme, distance), ...])
        One entry per token in *source_ipa*, each with a ranked list of
        target-language candidates.
    """
    tokens = source_spec.ipa_tokenizer(source_ipa)
    result: list[tuple[str, list[tuple[str, float]]]] = []
    for tok in tokens:
        feat_vec = source_phoneme_features.get(tok, {})
        candidates = invert_features(
            feat_vec,
            target_phoneme_features,
            top_n=top_n,
            max_distance=max_distance,
        )
        result.append((tok, candidates))
    return result
