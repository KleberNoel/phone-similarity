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

# ---------------------------------------------------------------------------
# Cython dispatch flags
# ---------------------------------------------------------------------------
try:
    from phone_similarity._core import (  # noqa: F401
        batch_pairwise_hamming as _c_batch_pairwise_hamming,
    )
    from phone_similarity._core import feature_edit_distance as _c_feature_edit_distance
    from phone_similarity._core import (  # noqa: F401
        hamming_distance as _c_hamming_distance,
    )
    from phone_similarity._core import hamming_similarity as _c_hamming_similarity  # noqa: F401

    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False

try:
    from phone_similarity._core import (
        batch_dictionary_scan as _c_batch_dictionary_scan,  # noqa: F401
    )
    from phone_similarity._core import invert_features as _c_invert_features  # noqa: F401
    from phone_similarity._core import (
        phoneme_feature_distance as _c_phoneme_feature_distance,
    )

    _HAS_CYTHON_EXT = True
except ImportError:
    _HAS_CYTHON_EXT = False

try:
    from phone_similarity._core import (
        prange_batch_dictionary_scan as _c_prange_batch_dictionary_scan,  # noqa: F401
    )

    _HAS_PRANGE = True
except ImportError:
    _HAS_PRANGE = False


def hamming_distance(a: bitarray, b: bitarray) -> int:
    """Count the number of differing bits between two equal-length bitarrays.

    Parameters
    ----------
    a, b : bitarray
        Must be the same length.

    Returns
    -------
    int
        Number of positions where the bits differ.
    """
    if len(a) != len(b):
        raise ValueError(f"Bitarrays must have equal length (got {len(a)} vs {len(b)})")
    return (a ^ b).count()


def hamming_similarity(a: bitarray, b: bitarray) -> float:
    """Normalised bit-match ratio (1.0 = identical, 0.0 = maximally different).

    Parameters
    ----------
    a, b : bitarray
        Must be the same length.

    Returns
    -------
    float
        ``1 - hamming_distance(a, b) / len(a)``.  Returns 1.0 for empty
        arrays.
    """
    n = len(a)
    if n == 0:
        return 1.0
    return 1.0 - hamming_distance(a, b) / n


def phoneme_feature_distance(
    features_a: dict[str, Union[bool, str]],
    features_b: dict[str, Union[bool, str]],
) -> float:
    """Normalised feature distance between two phonemes.

    Computes the fraction of feature dimensions on which the two phonemes
    disagree.  Features present in one dict but absent in the other count
    as a mismatch.

    Delegates to Cython when the extended module is available.

    Parameters
    ----------
    features_a, features_b : dict
        Feature dictionaries as stored in ``PHONEME_FEATURES``.

    Returns
    -------
    float
        Value in ``[0.0, 1.0]``.  0 means identical feature sets.
    """
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
    """Feature-weighted edit distance between two phoneme sequences.

    Standard Levenshtein DP, but the *substitution cost* for replacing
    phoneme ``p`` with phoneme ``q`` is
    ``phoneme_feature_distance(features[p], features[q])`` -- a value in
    ``[0, 1]`` -- rather than a flat 1.0.  This makes the metric sensitive
    to *how different* two sounds are, not just *whether* they differ.

    Delegates to the Cython implementation when available (approx. 4x faster).

    Parameters
    ----------
    seq_a, seq_b : sequence of str
        Phoneme sequences (e.g. from an IPA tokeniser).
    phoneme_features : dict
        Mapping of phoneme -> feature dict.  Phonemes not found in this
        dict are treated as having an empty feature set (maximum distance
        from anything else).
    insert_cost : float
        Cost of inserting a phoneme (default 1.0).
    delete_cost : float
        Cost of deleting a phoneme (default 1.0).

    Returns
    -------
    float
        The minimum-cost alignment distance.
    """
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
    """Pairwise Hamming similarity matrix for a list of bitarrays.

    Parameters
    ----------
    arrays : sequence of bitarray
        All must be the same length.

    Returns
    -------
    list of list of float
        Symmetric ``N x N`` matrix where ``result[i][j]`` is the Hamming
        similarity between ``arrays[i]`` and ``arrays[j]``.
    """
    n = len(arrays)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        result[i][i] = 1.0
        for j in range(i + 1, n):
            sim = hamming_similarity(arrays[i], arrays[j])
            result[i][j] = sim
            result[j][i] = sim
    return result
