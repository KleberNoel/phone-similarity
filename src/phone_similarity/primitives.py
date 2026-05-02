"""
Low-level phonological distance primitives.

Provides Hamming distance/similarity on bitarrays, phoneme-level feature
distance, feature-weighted edit distance (Levenshtein with gradient
substitution cost), and batch pairwise Hamming matrices.

Pure-Python implementations are transparently replaced by Cython-accelerated
versions when the ``_core`` extension is available.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

from bitarray import bitarray

from phone_similarity._dispatch import (
    HAS_CYTHON,
    HAS_CYTHON_EXT,
    HAS_PRANGE,
)
from phone_similarity._dispatch import (
    cy_batch_pairwise_hamming as _c_batch_pairwise_hamming,
)
from phone_similarity._dispatch import (
    cy_feature_edit_distance as _c_feature_edit_distance,
)
from phone_similarity._dispatch import (
    cy_hamming_similarity as _c_hamming_similarity,
)
from phone_similarity._dispatch import (
    cy_phoneme_feature_distance as _c_phoneme_feature_distance,
)

# Backward-compatible aliases for modules that import these flags
_HAS_CYTHON = HAS_CYTHON
_HAS_CYTHON_EXT = HAS_CYTHON_EXT
_HAS_PRANGE = HAS_PRANGE


def hamming_distance(a: bitarray, b: bitarray) -> int:
    """Count differing bits between two equal-length bitarrays."""
    if len(a) != len(b):
        raise ValueError(f"Bitarrays must have equal length (got {len(a)} vs {len(b)})")
    return (a ^ b).count()


def hamming_similarity(a: bitarray, b: bitarray) -> float:
    """Normalised bit-match ratio: ``1 - hamming_distance(a, b) / len(a)``. Returns 1.0 for empty arrays."""
    if HAS_CYTHON:
        return _c_hamming_similarity(a, b)
    n = len(a)
    if n == 0:
        return 1.0
    return 1.0 - hamming_distance(a, b) / n


def phoneme_feature_distance(
    features_a: dict[str, Union[bool, str]],
    features_b: dict[str, Union[bool, str]],
) -> float:
    """Normalised feature distance between two phonemes (fraction of disagreeing dimensions)."""
    if _HAS_CYTHON_EXT:
        return _c_phoneme_feature_distance(features_a, features_b)
    all_keys = set(features_a) | set(features_b)
    if not all_keys:
        return 0.0
    mismatches = sum(1 for k in all_keys if features_a.get(k) != features_b.get(k))
    return mismatches / len(all_keys)


def feature_edit_distance(
    seq_a: Sequence[str],
    seq_b: Sequence[str],
    phoneme_features: dict[str, dict[str, Union[bool, str]]],
    insert_cost: float = 1.0,
    delete_cost: float = 1.0,
) -> float:
    """Levenshtein edit distance with per-phoneme feature-distance substitution costs."""
    if _HAS_CYTHON:
        return _c_feature_edit_distance(seq_a, seq_b, phoneme_features, insert_cost, delete_cost)
    m, n = len(seq_a), len(seq_b)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + delete_cost
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + insert_cost

    for i in range(1, m + 1):
        fa = phoneme_features.get(seq_a[i - 1], {})
        for j in range(1, n + 1):
            fb = phoneme_features.get(seq_b[j - 1], {})
            sub_cost = phoneme_feature_distance(fa, fb)
            dp[i][j] = min(
                dp[i - 1][j] + delete_cost,
                dp[i][j - 1] + insert_cost,
                dp[i - 1][j - 1] + sub_cost,
            )

    return dp[m][n]


def normalised_feature_edit_distance(
    seq_a: Sequence[str],
    seq_b: Sequence[str],
    phoneme_features: dict[str, dict[str, Union[bool, str]]],
    insert_cost: float = 1.0,
    delete_cost: float = 1.0,
) -> float:
    """Feature edit distance normalised by the longer sequence length.

    Returns a value in ``[0.0, 1.0]``.  Returns 0.0 when both sequences
    are empty.
    """
    max_len = max(len(seq_a), len(seq_b))
    if max_len == 0:
        return 0.0
    raw = feature_edit_distance(seq_a, seq_b, phoneme_features, insert_cost, delete_cost)
    return raw / max_len


def batch_pairwise_hamming(
    arrays: Sequence[bitarray],
) -> list[list[float]]:
    """Symmetric N×N Hamming similarity matrix for a list of equal-length bitarrays."""
    if HAS_CYTHON:
        return _c_batch_pairwise_hamming(list(arrays))
    n = len(arrays)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        result[i][i] = 1.0
        for j in range(i + 1, n):
            sim = hamming_similarity(arrays[i], arrays[j])
            result[i][j] = sim
            result[j][i] = sim
    return result
