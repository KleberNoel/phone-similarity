# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: language_level=3
"""
Cython-accelerated core routines for phonological distance computation.

This module provides typed, optimised implementations of:

* ``hamming_distance``  -- popcount-based Hamming distance on raw byte buffers
* ``hamming_similarity`` -- normalised Hamming similarity
* ``feature_edit_distance`` -- DP edit distance with pre-computed float
  substitution-cost matrix
* ``batch_pairwise_hamming`` -- O(N^2/2) pairwise similarity matrix
* ``phoneme_feature_distance`` -- normalised feature-mismatch ratio
* ``invert_features`` -- feature vector → ranked phoneme lookup
* ``batch_dictionary_scan`` -- scan a pre-tokenized dictionary for closest
  matches to a source token sequence

All functions accept standard Python objects (``bitarray``, ``list``) on the
boundary and convert internally to typed memoryviews / C arrays for the
tight loops.
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset

from cython.parallel cimport prange, parallel

cdef extern from *:
    """
    #ifdef _OPENMP
    #include <omp.h>
    static int _get_max_threads(void) { return omp_get_max_threads(); }
    #else
    static int _get_max_threads(void) { return 1; }
    #endif
    """
    int _get_max_threads() nogil

import numpy as np
cimport numpy as cnp

cnp.import_array()



cpdef int hamming_distance(a, b) except -1:
    """Hamming distance between two bitarrays (number of differing bits).

    Uses ``bitarray`` XOR + count which is already C-accelerated, but
    this wrapper provides the Cython-module entry point so callers don't
    need to special-case.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Bitarrays must have equal length (got {len(a)} vs {len(b)})"
        )
    return (a ^ b).count()


cpdef double hamming_similarity(a, b):
    """Normalised bit-match ratio.  1.0 = identical, 0.0 = max different."""
    cdef int n = len(a)
    if n == 0:
        return 1.0
    cdef int dist = (a ^ b).count()
    return 1.0 - (<double>dist) / (<double>n)



cdef double _cdef_feature_edit_distance(
    object seq_a,
    Py_ssize_t m,
    object seq_b,
    Py_ssize_t n,
    dict phoneme_features,
    double insert_cost,
    double delete_cost,
):
    """Internal cdef DP — no Python call overhead when called from Cython."""
    cdef Py_ssize_t i, j
    cdef double sub_cost, del_val, ins_val, sub_val
    cdef dict fa, fb
    cdef set all_keys
    cdef Py_ssize_t num_keys, mismatches
    cdef str k

    cdef Py_ssize_t size = (m + 1) * (n + 1)
    cdef double* dp = <double*>malloc(size * sizeof(double))
    if dp == NULL:
        raise MemoryError("Could not allocate DP table")

    try:
        dp[0] = 0.0
        for i in range(1, m + 1):
            dp[i * (n + 1)] = dp[(i - 1) * (n + 1)] + delete_cost
        for j in range(1, n + 1):
            dp[j] = dp[j - 1] + insert_cost

        for i in range(1, m + 1):
            fa = phoneme_features.get(seq_a[i - 1], {})
            for j in range(1, n + 1):
                fb = phoneme_features.get(seq_b[j - 1], {})

                all_keys = set(fa) | set(fb)
                num_keys = len(all_keys)
                if num_keys == 0:
                    sub_cost = 0.0
                else:
                    mismatches = 0
                    for k in all_keys:
                        if fa.get(k) != fb.get(k):
                            mismatches += 1
                    sub_cost = (<double>mismatches) / (<double>num_keys)

                del_val = dp[(i - 1) * (n + 1) + j] + delete_cost
                ins_val = dp[i * (n + 1) + (j - 1)] + insert_cost
                sub_val = dp[(i - 1) * (n + 1) + (j - 1)] + sub_cost

                if del_val <= ins_val and del_val <= sub_val:
                    dp[i * (n + 1) + j] = del_val
                elif ins_val <= sub_val:
                    dp[i * (n + 1) + j] = ins_val
                else:
                    dp[i * (n + 1) + j] = sub_val

        return dp[m * (n + 1) + n]
    finally:
        free(dp)



cpdef double feature_edit_distance(
    object seq_a,
    object seq_b,
    dict phoneme_features,
    double insert_cost = 1.0,
    double delete_cost = 1.0,
):
    """Feature-weighted edit distance between two phoneme sequences.

    The substitution cost between phonemes *p* and *q* is the normalised
    feature distance (fraction of feature dimensions that disagree).

    Parameters
    ----------
    seq_a, seq_b : sequence of str (list, tuple, or any indexable)
    phoneme_features : dict mapping str -> dict
    insert_cost, delete_cost : float

    Returns
    -------
    float
    """
    return _cdef_feature_edit_distance(
        seq_a, len(seq_a), seq_b, len(seq_b),
        phoneme_features, insert_cost, delete_cost,
    )



cpdef list batch_pairwise_hamming(list arrays):
    """Pairwise Hamming similarity matrix for a list of bitarrays.

    Returns a list-of-lists (N x N) of floats.
    """
    cdef Py_ssize_t num = len(arrays)
    cdef Py_ssize_t i, j
    cdef double sim
    cdef int n_bits

    # Pre-allocate result matrix
    result = [[0.0] * num for _ in range(num)]

    if num == 0:
        return result

    n_bits = len(arrays[0])

    for i in range(num):
        result[i][i] = 1.0
        for j in range(i + 1, num):
            if n_bits == 0:
                sim = 1.0
            else:
                sim = 1.0 - (<double>(arrays[i] ^ arrays[j]).count()) / (<double>n_bits)
            result[i][j] = sim
            result[j][i] = sim

    return result



def phoneme_feature_distance(dict fa, dict fb) -> float:
    """Normalised feature-mismatch ratio between two phoneme feature dicts.

    Returns the fraction of the union of feature keys on which the two
    dicts disagree.  Returns 0.0 when both dicts are empty.
    """
    cdef set all_keys = set(fa) | set(fb)
    cdef Py_ssize_t num_keys = len(all_keys)
    if num_keys == 0:
        return 0.0
    cdef Py_ssize_t mismatches = 0
    cdef str k
    for k in all_keys:
        if fa.get(k) != fb.get(k):
            mismatches += 1
    return (<double>mismatches) / (<double>num_keys)



cdef void _fill_dist_matrix(
    double* dist_matrix,
    list all_phonemes,
    Py_ssize_t num_ph,
    Py_ssize_t matrix_dim,
    Py_ssize_t UNK_IDX,
    dict merged_feats,
):
    """Fill a pre-allocated flat distance matrix from merged_feats.

    Computes the symmetric pairwise phoneme feature distance for all
    phonemes in *all_phonemes*, plus an UNK row/col with distance 1.0
    to everything (0.0 to itself).
    """
    cdef Py_ssize_t ai, bi
    cdef dict fa, fb
    cdef set all_keys
    cdef Py_ssize_t num_keys, mismatches
    cdef str k
    cdef double d

    for ai in range(num_ph):
        fa = merged_feats.get(all_phonemes[ai], {})
        for bi in range(ai, num_ph):
            fb = merged_feats.get(all_phonemes[bi], {})
            all_keys = set(fa) | set(fb)
            num_keys = len(all_keys)
            if num_keys == 0:
                dist_matrix[ai * matrix_dim + bi] = 0.0
                dist_matrix[bi * matrix_dim + ai] = 0.0
            else:
                mismatches = 0
                for k in all_keys:
                    if fa.get(k) != fb.get(k):
                        mismatches += 1
                d = (<double>mismatches) / (<double>num_keys)
                dist_matrix[ai * matrix_dim + bi] = d
                dist_matrix[bi * matrix_dim + ai] = d
    # Unknown vs anything = 1.0
    for ai in range(matrix_dim):
        dist_matrix[ai * matrix_dim + UNK_IDX] = 1.0
        dist_matrix[UNK_IDX * matrix_dim + ai] = 1.0
    dist_matrix[UNK_IDX * matrix_dim + UNK_IDX] = 0.0


def build_phoneme_dist_matrix(dict merged_feats) -> tuple:
    """Build a pairwise phoneme distance matrix from merged feature dicts.

    Public entry point callable from Python (e.g. beam_search.py).

    Returns
    -------
    tuple of (ph_to_idx: dict, dist_flat: list[float], dim: int)
        * ph_to_idx: maps phoneme str -> int index
        * dist_flat: flat row-major float list (dim * dim)
        * dim: matrix dimension (num_phonemes + 1 for UNK)
    """
    cdef list all_phonemes = sorted(merged_feats)
    cdef Py_ssize_t num_ph = len(all_phonemes)
    cdef Py_ssize_t matrix_dim = num_ph + 1
    cdef Py_ssize_t UNK_IDX = num_ph
    cdef dict ph_to_idx = {}
    cdef Py_ssize_t pi

    for pi in range(num_ph):
        ph_to_idx[all_phonemes[pi]] = pi

    cdef double* dist_matrix = <double*>malloc(matrix_dim * matrix_dim * sizeof(double))
    if dist_matrix == NULL:
        raise MemoryError("Could not allocate distance matrix")

    cdef Py_ssize_t total = matrix_dim * matrix_dim
    cdef list dist_flat

    try:
        _fill_dist_matrix(dist_matrix, all_phonemes, num_ph, matrix_dim, UNK_IDX, merged_feats)

        # Convert to Python list
        dist_flat = [0.0] * total
        for pi in range(total):
            dist_flat[pi] = dist_matrix[pi]
    finally:
        free(dist_matrix)

    return (ph_to_idx, dist_flat, matrix_dim)



def invert_features(
    dict feature_vector,
    dict target_phoneme_features,
    int top_n = 0,
    double max_distance = 1.0,
) -> list:
    """Find phonemes in a target inventory closest to *feature_vector*.

    Parameters
    ----------
    feature_vector : dict
        A single phoneme's feature dictionary.
    target_phoneme_features : dict
        ``{phoneme: {feature: value}}`` for the target language.
    top_n : int
        If > 0, return only the *top_n* closest.
    max_distance : float
        Exclude phonemes further than this.

    Returns
    -------
    list of (phoneme, distance) tuples, sorted ascending.
    """
    cdef list ranked = []
    cdef str phoneme
    cdef dict feats
    cdef double d
    cdef set all_keys
    cdef Py_ssize_t num_keys, mismatches
    cdef str k

    for phoneme, feats in target_phoneme_features.items():
        # Inline phoneme_feature_distance for speed
        all_keys = set(feature_vector) | set(feats)
        num_keys = len(all_keys)
        if num_keys == 0:
            d = 0.0
        else:
            mismatches = 0
            for k in all_keys:
                if feature_vector.get(k) != feats.get(k):
                    mismatches += 1
            d = (<double>mismatches) / (<double>num_keys)

        if d <= max_distance:
            ranked.append((phoneme, d))

    ranked.sort(key=lambda t: t[1])
    if top_n > 0:
        return ranked[:top_n]
    return ranked



def cython_ipa_tokenizer(str ipa_str, frozenset phone_set, int max_phoneme_size):
    """Tokenize an IPA string using greedy longest-match against a phoneme set.

    This replaces the Python ``search_phonemes`` loop with a hash-set lookup:
    O(max_phoneme_size) per position instead of O(inventory_size).

    Parameters
    ----------
    ipa_str : str
        The IPA string to tokenize.
    phone_set : frozenset
        Set of all valid phonemes (for O(1) membership test).
    max_phoneme_size : int
        Maximum phoneme length in characters.

    Returns
    -------
    list[str]
        Tokenized phoneme list.
    """
    cdef list tokens = []
    cdef Py_ssize_t start = 0
    cdef Py_ssize_t str_len = len(ipa_str)
    cdef Py_ssize_t length
    cdef str candidate
    cdef bint found

    while start < str_len:
        found = False
        length = min(max_phoneme_size, str_len - start)
        while length > 0:
            candidate = ipa_str[start:start + length]
            if candidate in phone_set:
                tokens.append(candidate)
                start += length
                found = True
                break
            length -= 1
        if not found:
            start += 1  # skip unrecognised character

    return tokens


def batch_ipa_tokenize(
    list ipa_strings,
    frozenset phone_set,
    int max_phoneme_size,
    int min_tokens = 2,
):
    """Tokenize a batch of IPA strings in one Cython call.

    Parameters
    ----------
    ipa_strings : list[str]
        IPA strings to tokenize.
    phone_set : frozenset
        Set of all valid phonemes.
    max_phoneme_size : int
        Maximum phoneme length in characters.
    min_tokens : int
        Minimum number of tokens to include an entry (default 2).

    Returns
    -------
    list[list[str]]
        List of tokenized phoneme lists (entries below min_tokens are
        returned as empty lists).
    """
    cdef Py_ssize_t n = len(ipa_strings)
    cdef list result = []
    cdef Py_ssize_t i
    cdef str ipa_str
    cdef list tokens

    for i in range(n):
        ipa_str = ipa_strings[i]
        tokens = cython_ipa_tokenizer(ipa_str, phone_set, max_phoneme_size)
        if len(tokens) >= min_tokens:
            result.append(tokens)
        else:
            result.append([])

    return result



def batch_dictionary_scan(
    list source_tokens,
    Py_ssize_t source_len,
    object pre_tokenized,
    dict merged_feats,
    int top_n = 10,
    double max_distance = 0.50,
) -> list:
    """Scan a pre-tokenized dictionary for entries closest to *source_tokens*.

    Pre-computes a pairwise phoneme distance matrix from *merged_feats* so
    the DP inner loop is a single float lookup instead of set operations.
    Also applies length-ratio and phoneme-overlap pre-filters.

    When *pre_tokenized* is a ``PreTokenizedDictionary`` (with ``.inventory``,
    ``.token_indices``, ``.offsets``, ``.words``, ``.ipas`` attributes), the
    scan operates directly on the numpy index arrays, avoiding all Python
    list/set construction per entry.  This eliminates the ``__iter__()``
    bottleneck (previously 23.7% of CPU).

    For backward compatibility, a plain list of ``(word, ipa, tokens)`` tuples
    is still accepted and handled via the original iteration path.

    Parameters
    ----------
    source_tokens : list of str
        Tokenised IPA of the source phrase.
    source_len : int
        ``len(source_tokens)``.
    pre_tokenized : PreTokenizedDictionary or list of (word, ipa_str, tokens)
        Pre-tokenized dictionary entries.
    merged_feats : dict
        Merged phoneme-feature dict (source + target).
    top_n : int
        Return at most this many entries (default 10).
    max_distance : float
        Skip entries with normalised distance above this (default 0.50).

    Returns
    -------
    list of (word, ipa, distance) sorted by ascending distance.
    """
    # Dispatch: use fast numpy path when PreTokenizedDictionary is detected
    if (hasattr(pre_tokenized, 'token_indices') and
            hasattr(pre_tokenized, 'offsets') and
            hasattr(pre_tokenized, 'inventory')):
        return _batch_scan_ptd(
            source_tokens, source_len, pre_tokenized,
            merged_feats, top_n, max_distance,
        )
    return _batch_scan_generic(
        source_tokens, source_len, pre_tokenized,
        merged_feats, top_n, max_distance,
    )



cdef list _batch_scan_ptd(
    list source_tokens,
    Py_ssize_t source_len,
    object ptd,
    dict merged_feats,
    int top_n,
    double max_distance,
):
    """Scan a PreTokenizedDictionary without calling __iter__().

    Accesses .token_indices (int16[]), .offsets (int32[]), .inventory,
    .words, .ipas directly to avoid constructing Python token lists.
    """
    cdef list candidates = []
    cdef Py_ssize_t target_len
    cdef double ratio, d, raw
    cdef Py_ssize_t max_len

    cdef set all_phonemes_set
    cdef Py_ssize_t pi
    cdef list all_phonemes
    cdef dict ph_to_idx
    cdef Py_ssize_t num_ph, UNK_IDX, matrix_dim
    cdef double* dist_matrix
    cdef Py_ssize_t* src_idx
    cdef Py_ssize_t* tgt_idx
    cdef Py_ssize_t j
    cdef Py_ssize_t overlap, min_overlap

    # PTD-specific: read numpy arrays once
    cdef list ptd_words = ptd.words
    cdef list ptd_ipas = ptd.ipas
    cdef list ptd_inventory = ptd.inventory
    cdef Py_ssize_t ptd_inv_size = len(ptd_inventory)
    cdef Py_ssize_t n_entries = len(ptd_words)

    # Get raw numpy buffer pointers (avoids repeated Python indexing)
    ti_arr = ptd.token_indices   # int16 numpy array
    off_arr = ptd.offsets        # int32 numpy array
    cdef const cnp.int16_t[:] ti_view = ti_arr
    cdef const cnp.int32_t[:] off_view = off_arr

    # Translation table: ptd inventory index -> distance-matrix index
    cdef Py_ssize_t* inv_to_mat = NULL
    # Source phoneme set as bitmask over matrix indices for overlap filter
    cdef unsigned char* src_mask = NULL
    cdef unsigned char* tgt_mask = NULL
    cdef Py_ssize_t src_unique_count
    cdef Py_ssize_t tgt_unique_count
    cdef Py_ssize_t overlap_count
    cdef Py_ssize_t entry_idx, tgt_start, tgt_end

    # ── Build the phoneme universe ────────────────────────────────
    all_phonemes_set = set(source_tokens)
    for pi in range(ptd_inv_size):
        all_phonemes_set.add(ptd_inventory[pi])
    for ph in merged_feats:
        all_phonemes_set.add(ph)

    all_phonemes = sorted(all_phonemes_set)
    ph_to_idx = {}
    num_ph = len(all_phonemes)
    for pi in range(num_ph):
        ph_to_idx[all_phonemes[pi]] = pi

    UNK_IDX = num_ph
    matrix_dim = num_ph + 1

    # ── Allocate C arrays ─────────────────────────────────────────
    dist_matrix = <double*>malloc(matrix_dim * matrix_dim * sizeof(double))
    if dist_matrix == NULL:
        raise MemoryError("Could not allocate distance matrix")

    src_idx = <Py_ssize_t*>malloc(source_len * sizeof(Py_ssize_t))
    if src_idx == NULL:
        free(dist_matrix)
        raise MemoryError("Could not allocate source index array")

    inv_to_mat = <Py_ssize_t*>malloc(ptd_inv_size * sizeof(Py_ssize_t))
    if inv_to_mat == NULL:
        free(src_idx)
        free(dist_matrix)
        raise MemoryError("Could not allocate inventory translation array")

    src_mask = <unsigned char*>malloc(matrix_dim * sizeof(unsigned char))
    if src_mask == NULL:
        free(inv_to_mat)
        free(src_idx)
        free(dist_matrix)
        raise MemoryError("Could not allocate source mask")

    tgt_mask = <unsigned char*>malloc(matrix_dim * sizeof(unsigned char))
    if tgt_mask == NULL:
        free(src_mask)
        free(inv_to_mat)
        free(src_idx)
        free(dist_matrix)
        raise MemoryError("Could not allocate target mask")

    try:
        # ── Fill distance matrix (shared helper) ──────────────────
        _fill_dist_matrix(dist_matrix, all_phonemes, num_ph, matrix_dim, UNK_IDX, merged_feats)

        # ── Build inventory translation table ─────────────────────
        for pi in range(ptd_inv_size):
            inv_to_mat[pi] = ph_to_idx.get(ptd_inventory[pi], UNK_IDX)

        # ── Convert source tokens to matrix index array ───────────
        memset(src_mask, 0, matrix_dim * sizeof(unsigned char))
        src_unique_count = 0
        for pi in range(source_len):
            src_idx[pi] = ph_to_idx.get(source_tokens[pi], UNK_IDX)
            if src_mask[src_idx[pi]] == 0:
                src_mask[src_idx[pi]] = 1
                src_unique_count += 1

        # ── Scan entries (no Python iteration) ────────────────────
        for entry_idx in range(n_entries):
            tgt_start = off_view[entry_idx]
            tgt_end = off_view[entry_idx + 1]
            target_len = tgt_end - tgt_start

            if target_len == 0:
                continue

            # Length-ratio pre-filter
            if source_len >= target_len:
                ratio = (<double>source_len) / (<double>target_len)
            else:
                ratio = (<double>target_len) / (<double>source_len)
            if ratio > 2.0:
                continue

            # Phoneme-set overlap pre-filter (using bitmask, no Python sets)
            memset(tgt_mask, 0, matrix_dim * sizeof(unsigned char))
            tgt_unique_count = 0
            for pi in range(tgt_start, tgt_end):
                j = inv_to_mat[ti_view[pi]]
                if tgt_mask[j] == 0:
                    tgt_mask[j] = 1
                    tgt_unique_count += 1

            overlap_count = 0
            min_overlap = src_unique_count if src_unique_count < tgt_unique_count else tgt_unique_count
            if min_overlap > 0:
                for pi in range(matrix_dim):
                    if src_mask[pi] != 0 and tgt_mask[pi] != 0:
                        overlap_count += 1
                if (<double>overlap_count) / (<double>min_overlap) < 0.20:
                    continue

            # ── Build target index array via translation table ────
            tgt_idx = <Py_ssize_t*>malloc(target_len * sizeof(Py_ssize_t))
            if tgt_idx == NULL:
                raise MemoryError("Could not allocate target index array")
            for pi in range(target_len):
                tgt_idx[pi] = inv_to_mat[ti_view[tgt_start + pi]]

            # ── DP via shared nogil kernel ────────────────────────
            raw = _dp_edit_distance_nogil(
                src_idx, source_len,
                tgt_idx, target_len,
                dist_matrix, matrix_dim,
            )
            free(tgt_idx)

            if raw < 0.0:
                continue  # allocation failure inside kernel

            max_len = source_len if source_len >= target_len else target_len
            d = raw / (<double>max_len)

            if d <= max_distance:
                candidates.append((ptd_words[entry_idx], ptd_ipas[entry_idx], d))

        free(src_idx)
    finally:
        free(tgt_mask)
        free(src_mask)
        free(inv_to_mat)
        free(dist_matrix)

    candidates.sort(key=lambda t: t[2])
    return candidates[:top_n]



cdef list _batch_scan_generic(
    list source_tokens,
    Py_ssize_t source_len,
    object pre_tokenized,
    dict merged_feats,
    int top_n,
    double max_distance,
):
    """Original scan path for plain list-of-tuples pre_tokenized data."""
    cdef list candidates = []
    cdef Py_ssize_t target_len
    cdef double ratio, d, raw
    cdef str word, ipa
    cdef list target_tokens
    cdef Py_ssize_t max_len

    cdef set all_phonemes_set, source_set, target_set
    cdef Py_ssize_t pi
    cdef list all_phonemes
    cdef dict ph_to_idx
    cdef Py_ssize_t num_ph, UNK_IDX, matrix_dim
    cdef double* dist_matrix
    cdef Py_ssize_t* src_idx
    cdef Py_ssize_t* tgt_idx
    cdef Py_ssize_t overlap, min_overlap

    # ── Pre-compute phoneme distance matrix ────────────────────────
    all_phonemes_set = set(source_tokens)
    for entry in pre_tokenized:
        for ph in entry[2]:
            all_phonemes_set.add(ph)
    for ph in merged_feats:
        all_phonemes_set.add(ph)

    all_phonemes = sorted(all_phonemes_set)
    ph_to_idx = {}
    num_ph = len(all_phonemes)
    for pi in range(num_ph):
        ph_to_idx[all_phonemes[pi]] = pi

    UNK_IDX = num_ph
    matrix_dim = num_ph + 1

    dist_matrix = <double*>malloc(matrix_dim * matrix_dim * sizeof(double))
    if dist_matrix == NULL:
        raise MemoryError("Could not allocate distance matrix")

    src_idx = <Py_ssize_t*>malloc(source_len * sizeof(Py_ssize_t))
    if src_idx == NULL:
        free(dist_matrix)
        raise MemoryError("Could not allocate source index array")

    try:
        # Fill the matrix (shared helper)
        _fill_dist_matrix(dist_matrix, all_phonemes, num_ph, matrix_dim, UNK_IDX, merged_feats)

        # Convert source tokens to index array
        for pi in range(source_len):
            src_idx[pi] = ph_to_idx.get(source_tokens[pi], UNK_IDX)

        # Pre-compute source phoneme set for overlap filter
        source_set = set(source_tokens)

        # ── Scan entries ──────────────────────────────────────────

        for entry in pre_tokenized:
            word = entry[0]
            ipa = entry[1]
            target_tokens = entry[2]
            target_len = len(target_tokens)

            if target_len == 0:
                continue

            # Length-ratio pre-filter
            if source_len >= target_len:
                ratio = (<double>source_len) / (<double>target_len)
            else:
                ratio = (<double>target_len) / (<double>source_len)
            if ratio > 2.0:
                continue

            # Phoneme-set overlap pre-filter
            target_set = set(target_tokens)
            overlap = len(source_set & target_set)
            min_overlap = min(len(source_set), len(target_set))
            if min_overlap > 0 and (<double>overlap) / (<double>min_overlap) < 0.20:
                continue

            # Convert target tokens to index array
            tgt_idx = <Py_ssize_t*>malloc(target_len * sizeof(Py_ssize_t))
            if tgt_idx == NULL:
                raise MemoryError("Could not allocate target index array")
            for pi in range(target_len):
                tgt_idx[pi] = ph_to_idx.get(target_tokens[pi], UNK_IDX)

            # DP via shared nogil kernel
            raw = _dp_edit_distance_nogil(
                src_idx, source_len,
                tgt_idx, target_len,
                dist_matrix, matrix_dim,
            )
            free(tgt_idx)

            if raw < 0.0:
                continue  # allocation failure inside kernel

            max_len = source_len if source_len >= target_len else target_len
            d = raw / (<double>max_len)

            if d <= max_distance:
                candidates.append((word, ipa, d))

        free(src_idx)
    finally:
        free(dist_matrix)

    candidates.sort(key=lambda t: t[2])
    return candidates[:top_n]



cdef double _dp_edit_distance_nogil(
    const Py_ssize_t* src_idx,
    Py_ssize_t src_len,
    const Py_ssize_t* tgt_idx,
    Py_ssize_t tgt_len,
    const double* dist_matrix,
    Py_ssize_t matrix_dim,
) noexcept nogil:
    """Compute feature-weighted edit distance entirely without the GIL.

    Uses thread-local malloc/free for the DP table.  Returns raw (unnormalised)
    edit distance, or -1.0 on allocation failure.
    """
    cdef Py_ssize_t dp_size = (src_len + 1) * (tgt_len + 1)
    cdef double* dp = <double*>malloc(dp_size * sizeof(double))
    if dp == NULL:
        return -1.0  # signal allocation failure

    cdef Py_ssize_t i, j
    cdef Py_ssize_t stride = tgt_len + 1
    cdef double sub_cost, del_val, ins_val, sub_val

    dp[0] = 0.0
    for i in range(1, src_len + 1):
        dp[i * stride] = <double>i
    for j in range(1, tgt_len + 1):
        dp[j] = <double>j

    for i in range(1, src_len + 1):
        for j in range(1, tgt_len + 1):
            sub_cost = dist_matrix[src_idx[i - 1] * matrix_dim + tgt_idx[j - 1]]

            del_val = dp[(i - 1) * stride + j] + 1.0
            ins_val = dp[i * stride + (j - 1)] + 1.0
            sub_val = dp[(i - 1) * stride + (j - 1)] + sub_cost

            if del_val <= ins_val and del_val <= sub_val:
                dp[i * stride + j] = del_val
            elif ins_val <= sub_val:
                dp[i * stride + j] = ins_val
            else:
                dp[i * stride + j] = sub_val

    cdef double result = dp[src_len * stride + tgt_len]
    free(dp)
    return result



def prange_batch_dictionary_scan(
    list source_tokens,
    Py_ssize_t source_len,
    object pre_tokenized,
    dict merged_feats,
    int top_n = 10,
    double max_distance = 0.50,
    int num_threads = 0,
) -> list:
    """Parallel dictionary scan using OpenMP prange.

    Same semantics as ``batch_dictionary_scan`` but parallelises the
    per-entry DP computation across multiple OS threads using Cython's
    ``prange``.

    Parameters
    ----------
    source_tokens : list of str
        Tokenised IPA of the source phrase.
    source_len : int
        ``len(source_tokens)``.
    pre_tokenized : PreTokenizedDictionary
        Must have ``.token_indices``, ``.offsets``, ``.inventory``,
        ``.words``, ``.ipas`` attributes (numpy-backed).
    merged_feats : dict
        Merged phoneme-feature dict (source + target).
    top_n : int
        Return at most this many entries (default 10).
    max_distance : float
        Normalised distance threshold (default 0.50).
    num_threads : int
        Number of OpenMP threads.  0 means use OMP_NUM_THREADS or all cores.

    Returns
    -------
    list of (word, ipa, distance) sorted ascending by distance.
    """
    if not (hasattr(pre_tokenized, 'token_indices') and
            hasattr(pre_tokenized, 'offsets') and
            hasattr(pre_tokenized, 'inventory')):
        # Fall back to the sequential path for generic inputs
        return _batch_scan_generic(
            source_tokens, source_len, pre_tokenized,
            merged_feats, top_n, max_distance,
        )

    if source_len == 0:
        return []

    # PTD data
    cdef list ptd_words = pre_tokenized.words
    cdef list ptd_ipas = pre_tokenized.ipas
    cdef list ptd_inventory = pre_tokenized.inventory
    cdef Py_ssize_t ptd_inv_size = len(ptd_inventory)
    cdef Py_ssize_t n_entries = len(ptd_words)

    ti_arr = np.ascontiguousarray(pre_tokenized.token_indices, dtype=np.int16)
    off_arr = np.ascontiguousarray(pre_tokenized.offsets, dtype=np.int32)
    cdef const cnp.int16_t[:] ti_view = ti_arr
    cdef const cnp.int32_t[:] off_view = off_arr

    # Build phoneme universe
    cdef set all_phonemes_set = set(source_tokens)
    cdef Py_ssize_t pi
    for pi in range(ptd_inv_size):
        all_phonemes_set.add(ptd_inventory[pi])
    for ph in merged_feats:
        all_phonemes_set.add(ph)

    cdef list all_phonemes = sorted(all_phonemes_set)
    cdef dict ph_to_idx = {}
    cdef Py_ssize_t num_ph = len(all_phonemes)
    for pi in range(num_ph):
        ph_to_idx[all_phonemes[pi]] = pi

    cdef Py_ssize_t UNK_IDX = num_ph
    cdef Py_ssize_t matrix_dim = num_ph + 1

    # Build flat C arrays for the parallel section
    cdef double* c_dist_matrix = <double*>malloc(matrix_dim * matrix_dim * sizeof(double))
    if c_dist_matrix == NULL:
        raise MemoryError("Could not allocate distance matrix")

    cdef Py_ssize_t* src_idx_arr = <Py_ssize_t*>malloc(source_len * sizeof(Py_ssize_t))
    if src_idx_arr == NULL:
        free(c_dist_matrix)
        raise MemoryError("Could not allocate source index array")

    cdef Py_ssize_t* inv_to_mat = <Py_ssize_t*>malloc(ptd_inv_size * sizeof(Py_ssize_t))
    if inv_to_mat == NULL:
        free(src_idx_arr)
        free(c_dist_matrix)
        raise MemoryError("Could not allocate inventory translation array")

    # Pre-allocate flat target index buffer (all target tokens concatenated)
    cdef Py_ssize_t total_tokens = off_view[n_entries]
    cdef Py_ssize_t* all_tgt_idx = <Py_ssize_t*>malloc(total_tokens * sizeof(Py_ssize_t))
    if all_tgt_idx == NULL:
        free(inv_to_mat)
        free(src_idx_arr)
        free(c_dist_matrix)
        raise MemoryError("Could not allocate target index array")

    # Result array: one double per entry (-1.0 = skipped/filtered)
    cdef double* result_distances = <double*>malloc(n_entries * sizeof(double))
    if result_distances == NULL:
        free(all_tgt_idx)
        free(inv_to_mat)
        free(src_idx_arr)
        free(c_dist_matrix)
        raise MemoryError("Could not allocate result distances array")

    # Source bitmask for overlap filtering
    cdef unsigned char* src_mask = <unsigned char*>malloc(matrix_dim * sizeof(unsigned char))
    if src_mask == NULL:
        free(result_distances)
        free(all_tgt_idx)
        free(inv_to_mat)
        free(src_idx_arr)
        free(c_dist_matrix)
        raise MemoryError("Could not allocate source mask")

    cdef double d

    # Variables used inside the try block (must be declared before try)
    cdef Py_ssize_t src_unique_count = 0
    cdef Py_ssize_t ti
    cdef unsigned char* tgt_mask = NULL
    cdef Py_ssize_t* pass_entries = NULL
    cdef Py_ssize_t n_pass = 0
    cdef Py_ssize_t tgt_unique_count, overlap_count, min_overlap
    cdef Py_ssize_t j_idx
    cdef Py_ssize_t entry_idx, tgt_start, tgt_end, target_len, max_len
    cdef double ratio, raw
    cdef Py_ssize_t pass_idx
    cdef Py_ssize_t eidx
    cdef Py_ssize_t chunk_sz
    # Resolve thread count: 0 means "use all available" via OpenMP runtime
    cdef int n_threads_c
    if num_threads > 0:
        n_threads_c = num_threads
    else:
        n_threads_c = _get_max_threads()
        if n_threads_c < 1:
            n_threads_c = 1
    cdef list candidates

    # Offset array as a C pointer for nogil access
    cdef Py_ssize_t* c_offsets = <Py_ssize_t*>malloc((n_entries + 1) * sizeof(Py_ssize_t))
    if c_offsets == NULL:
        free(src_mask)
        free(result_distances)
        free(all_tgt_idx)
        free(inv_to_mat)
        free(src_idx_arr)
        free(c_dist_matrix)
        raise MemoryError("Could not allocate offsets C array")

    try:
        # Fill distance matrix (shared helper)
        _fill_dist_matrix(c_dist_matrix, all_phonemes, num_ph, matrix_dim, UNK_IDX, merged_feats)

        # Build inventory translation table
        for pi in range(ptd_inv_size):
            inv_to_mat[pi] = ph_to_idx.get(ptd_inventory[pi], UNK_IDX)

        # Convert source tokens to matrix indices
        src_unique_count = 0
        memset(src_mask, 0, matrix_dim * sizeof(unsigned char))
        for pi in range(source_len):
            src_idx_arr[pi] = ph_to_idx.get(source_tokens[pi], UNK_IDX)
            if src_mask[src_idx_arr[pi]] == 0:
                src_mask[src_idx_arr[pi]] = 1
                src_unique_count += 1

        # Pre-translate ALL target tokens to matrix indices
        for ti in range(total_tokens):
            all_tgt_idx[ti] = inv_to_mat[ti_view[ti]]

        # Copy offsets to C array
        for pi in range(n_entries + 1):
            c_offsets[pi] = off_view[pi]

        # Sequential pre-filter: length ratio + overlap
        tgt_mask = <unsigned char*>malloc(matrix_dim * sizeof(unsigned char))
        if tgt_mask == NULL:
            raise MemoryError("Could not allocate target mask")

        pass_entries = <Py_ssize_t*>malloc(n_entries * sizeof(Py_ssize_t))
        if pass_entries == NULL:
            free(tgt_mask)
            raise MemoryError("Could not allocate pass_entries array")

        n_pass = 0

        # Initialise all distances to -1 (skipped)
        for entry_idx in range(n_entries):
            result_distances[entry_idx] = -1.0

        for entry_idx in range(n_entries):
            tgt_start = c_offsets[entry_idx]
            tgt_end = c_offsets[entry_idx + 1]
            target_len = tgt_end - tgt_start

            if target_len == 0:
                continue

            # Length-ratio pre-filter
            if source_len >= target_len:
                ratio = (<double>source_len) / (<double>target_len)
            else:
                ratio = (<double>target_len) / (<double>source_len)
            if ratio > 2.0:
                continue

            # Phoneme-set overlap pre-filter
            memset(tgt_mask, 0, matrix_dim * sizeof(unsigned char))
            tgt_unique_count = 0
            for pi in range(tgt_start, tgt_end):
                j_idx = all_tgt_idx[pi]
                if tgt_mask[j_idx] == 0:
                    tgt_mask[j_idx] = 1
                    tgt_unique_count += 1

            overlap_count = 0
            min_overlap = src_unique_count if src_unique_count < tgt_unique_count else tgt_unique_count
            if min_overlap > 0:
                for pi in range(matrix_dim):
                    if src_mask[pi] != 0 and tgt_mask[pi] != 0:
                        overlap_count += 1
                if (<double>overlap_count) / (<double>min_overlap) < 0.20:
                    continue

            pass_entries[n_pass] = entry_idx
            n_pass += 1

        free(tgt_mask)
        tgt_mask = NULL

        # Parallel DP phase (nogil)
        if n_pass > 0:
            # Clamp chunksize: at least 1, at most ceil(n_pass / n_threads_c)
            chunk_sz = (n_pass + n_threads_c - 1) // n_threads_c
            if chunk_sz < 1:
                chunk_sz = 1
            if chunk_sz > 64:
                chunk_sz = 64
            with nogil:
                for pass_idx in prange(n_pass, num_threads=n_threads_c, schedule='dynamic', chunksize=chunk_sz):
                    eidx = pass_entries[pass_idx]
                    tgt_start = c_offsets[eidx]
                    tgt_end = c_offsets[eidx + 1]
                    target_len = tgt_end - tgt_start

                    raw = _dp_edit_distance_nogil(
                        src_idx_arr, source_len,
                        &all_tgt_idx[tgt_start], target_len,
                        c_dist_matrix, matrix_dim,
                    )

                    if raw < 0.0:
                        result_distances[eidx] = -1.0
                    else:
                        max_len = source_len if source_len >= target_len else target_len
                        result_distances[eidx] = raw / (<double>max_len)

        free(pass_entries)
        pass_entries = NULL

        # Collect results
        candidates = []
        for entry_idx in range(n_entries):
            d = result_distances[entry_idx]
            if d >= 0.0 and d <= max_distance:
                candidates.append((ptd_words[entry_idx], ptd_ipas[entry_idx], d))

    finally:
        if tgt_mask != NULL:
            free(tgt_mask)
        if pass_entries != NULL:
            free(pass_entries)
        free(c_offsets)
        free(src_mask)
        free(result_distances)
        free(all_tgt_idx)
        free(inv_to_mat)
        free(src_idx_arr)
        free(c_dist_matrix)

    candidates.sort(key=lambda t: t[2])
    return candidates[:top_n]



cdef Py_ssize_t _split_cluster_nogil(
    const int* son,
    Py_ssize_t k,
    bint sibilant_appendix,
) noexcept nogil:
    """Find onset start index in a consonant cluster.

    Returns the index where the onset begins (everything before it is
    coda).  Pure C — no GIL required.

    The split maximises the onset subject to SSP (strictly rising
    sonority towards the nucleus).  When *sibilant_appendix* is true,
    a fricative (rank 2) may attach to the left of a stop (rank 1).
    """
    if k <= 1:
        return 0  # single consonant → all onset

    cdef Py_ssize_t onset_start = k - 1
    while onset_start > 0:
        if son[onset_start - 1] < son[onset_start]:
            onset_start -= 1
        else:
            break

    # Sibilant appendix (/s/ exception)
    if (sibilant_appendix
            and onset_start > 0
            and son[onset_start - 1] == 2    # RANK_FRICATIVE
            and son[onset_start] == 1):      # RANK_STOP
        onset_start -= 1

    return onset_start


def cython_syllabify(
    list tokens,
    frozenset vowels,
    dict sonority_map,
    bint sibilant_appendix = True,
):
    """Syllabify *tokens* using MOP + SSP in Cython.

    Parameters
    ----------
    tokens : list of str
        IPA phoneme tokens.
    vowels : frozenset of str
        Vowel inventory.
    sonority_map : dict
        ``{phoneme: int}`` sonority ranks.
    sibilant_appendix : bool
        Allow /s/-exception.

    Returns
    -------
    list of (onset_list, nucleus_list, coda_list) tuples.
    """
    cdef Py_ssize_t n = len(tokens)
    if n == 0:
        return []

    # All cdef declarations up front (Cython requirement)
    cdef int* son = <int*>malloc(n * sizeof(int))
    if son == NULL:
        raise MemoryError("Could not allocate sonority array")

    cdef bint* is_vowel = <bint*>malloc(n * sizeof(bint))
    if is_vowel == NULL:
        free(son)
        raise MemoryError("Could not allocate vowel mask")

    cdef Py_ssize_t i
    cdef str ph
    cdef Py_ssize_t max_spans = n
    cdef Py_ssize_t* span_starts = NULL
    cdef Py_ssize_t* span_ends = NULL
    cdef Py_ssize_t n_spans = 0
    cdef list results = []
    cdef Py_ssize_t v_start, v_end, prev_end
    cdef Py_ssize_t cluster_len, split
    cdef Py_ssize_t idx
    cdef list syllables = []
    cdef list onset, nucleus, coda, old_syl

    try:
        for i in range(n):
            ph = tokens[i]
            son[i] = sonority_map.get(ph, 0)
            is_vowel[i] = (ph in vowels)

        # Phase 1: find vowel spans
        span_starts = <Py_ssize_t*>malloc(max_spans * sizeof(Py_ssize_t))
        span_ends = <Py_ssize_t*>malloc(max_spans * sizeof(Py_ssize_t))
        if span_starts == NULL or span_ends == NULL:
            raise MemoryError("Could not allocate span arrays")

        i = 0
        while i < n:
            if is_vowel[i]:
                span_starts[n_spans] = i
                while i < n and is_vowel[i]:
                    i += 1
                span_ends[n_spans] = i
                n_spans += 1
            else:
                i += 1

        if n_spans == 0:
            # No vowels — degenerate syllable (onset only)
            results.append((list(tokens), [], []))
            free(span_ends)
            free(span_starts)
            free(is_vowel)
            free(son)
            return results

        # Phase 2: split inter-vocalic clusters
        for idx in range(n_spans):
            v_start = span_starts[idx]
            v_end = span_ends[idx]

            if idx == 0:
                onset = list(tokens[:v_start])
            else:
                prev_end = span_ends[idx - 1]
                cluster_len = v_start - prev_end
                if cluster_len == 0:
                    onset = []
                else:
                    split = _split_cluster_nogil(
                        &son[prev_end], cluster_len, sibilant_appendix,
                    )
                    if split > 0:
                        old_syl = syllables[len(syllables) - 1]
                        for i in range(split):
                            old_syl[2].append(tokens[prev_end + i])
                    onset = []
                    for i in range(split, cluster_len):
                        onset.append(tokens[prev_end + i])

            nucleus = list(tokens[v_start:v_end])

            if idx == n_spans - 1:
                coda = list(tokens[v_end:])
            else:
                coda = []

            syllables.append([onset, nucleus, coda])

        if span_ends != NULL:
            free(span_ends)
            span_ends = NULL
        if span_starts != NULL:
            free(span_starts)
            span_starts = NULL

        results = [(s[0], s[1], s[2]) for s in syllables]

    finally:
        if span_ends != NULL:
            free(span_ends)
        if span_starts != NULL:
            free(span_starts)
        free(is_vowel)
        free(son)

    return results


def batch_cython_syllabify(
    list token_lists,
    frozenset vowels,
    dict sonority_map,
    bint sibilant_appendix = True,
):
    """Syllabify a batch of token lists in one Cython call.

    Parameters
    ----------
    token_lists : list of list of str
    vowels : frozenset of str
    sonority_map : dict
    sibilant_appendix : bool

    Returns
    -------
    list of list of (onset_list, nucleus_list, coda_list)
    """
    cdef Py_ssize_t n_words = len(token_lists)
    cdef list results = []
    cdef Py_ssize_t i

    for i in range(n_words):
        results.append(
            cython_syllabify(token_lists[i], vowels, sonority_map, sibilant_appendix)
        )

    return results



DEF _NUM_COART_FEATURES = 24


cdef double _coarticulated_phoneme_distance_c(
    const double* va,
    const double* vb,
    Py_ssize_t n_feat,
    bint either_fric,
    bint either_sib,
    double fric_w,
    double sib_w,
    Py_ssize_t cont_idx,
    Py_ssize_t strid_idx,
) noexcept nogil:
    """Phoneme distance between two co-articulated float vectors.

    Mirrors ``coarticulated_phoneme_distance`` in ``coarticulation.py``
    but operates entirely on C arrays without the GIL.

    Features where both values are near zero (|v| < 0.01) are skipped.
    Differences on ``cont`` and ``strid`` are weighted by *fric_w* and
    *sib_w* respectively when the corresponding fricative/sibilant flag
    is set.

    Returns normalised distance in [0, 1].
    """
    cdef Py_ssize_t idx
    cdef int comparable = 0
    cdef double total_diff = 0.0
    cdef double a_val, b_val, diff

    for idx in range(n_feat):
        a_val = va[idx]
        b_val = vb[idx]
        # Skip features where both are near-zero (unspecified)
        if -0.01 < a_val < 0.01 and -0.01 < b_val < 0.01:
            continue
        comparable += 1
        diff = a_val - b_val
        if diff < 0.0:
            diff = -diff
        diff *= 0.5

        if either_fric and idx == cont_idx:
            diff *= fric_w
        elif either_sib and idx == strid_idx:
            diff *= sib_w

        total_diff += diff

    if comparable == 0:
        return 0.0
    return total_diff / <double>comparable


def coarticulated_feature_edit_distance_c(
    list vecs_a,
    list vecs_b,
    double fric_w,
    double sib_w,
    Py_ssize_t son_idx,
    Py_ssize_t cont_idx,
    Py_ssize_t strid_idx,
    double insert_cost = 1.0,
    double delete_cost = 1.0,
) -> float:
    """DP edit distance on pre-computed co-articulated float vectors.

    This is the Cython-accelerated inner loop for
    ``coarticulated_feature_edit_distance``.  The caller (Python) is
    responsible for running ``perturb_sequence`` and passing the
    resulting list-of-24-float-tuples here.

    Parameters
    ----------
    vecs_a, vecs_b : list of tuple[float, ...]
        Co-articulated 24-float feature vectors.
    fric_w, sib_w : float
        Weights from ``FricativeConfig.fricative_weight`` / ``sibilant_weight``.
    son_idx, cont_idx, strid_idx : int
        Indices of the ``son``, ``cont``, and ``strid`` features in the
        24-feature vector.
    insert_cost, delete_cost : float
        Indel costs (default 1.0).

    Returns
    -------
    float
        Raw (unnormalised) edit distance.
    """
    cdef Py_ssize_t len_a = len(vecs_a)
    cdef Py_ssize_t len_b = len(vecs_b)
    cdef Py_ssize_t n_feat = _NUM_COART_FEATURES

    # Allocate flat C arrays for vectors
    cdef double* flat_a = NULL
    cdef double* flat_b = NULL
    # Fricative/sibilant classification flags per token
    cdef bint* fric_a = NULL
    cdef bint* fric_b = NULL
    cdef bint* sib_a = NULL
    cdef bint* sib_b = NULL
    # DP table
    cdef double* prev_row = NULL
    cdef double* curr_row = NULL

    cdef Py_ssize_t i, j, k
    cdef double a_val, b_val
    cdef bint is_fric, is_sib
    cdef bint ef, es, ef_pair, es_pair
    cdef double sub_cost, d_val, ins_val, sub_val
    cdef double result
    cdef tuple vec

    flat_a = <double*>malloc(len_a * n_feat * sizeof(double))
    flat_b = <double*>malloc(len_b * n_feat * sizeof(double))
    fric_a = <bint*>malloc(len_a * sizeof(bint))
    fric_b = <bint*>malloc(len_b * sizeof(bint))
    sib_a = <bint*>malloc(len_a * sizeof(bint))
    sib_b = <bint*>malloc(len_b * sizeof(bint))
    prev_row = <double*>malloc((len_b + 1) * sizeof(double))
    curr_row = <double*>malloc((len_b + 1) * sizeof(double))

    if (flat_a == NULL or flat_b == NULL or
            fric_a == NULL or fric_b == NULL or
            sib_a == NULL or sib_b == NULL or
            prev_row == NULL or curr_row == NULL):
        free(flat_a); free(flat_b)
        free(fric_a); free(fric_b)
        free(sib_a); free(sib_b)
        free(prev_row); free(curr_row)
        raise MemoryError("coarticulated_feature_edit_distance_c: malloc failed")

    try:
        # Unpack Python tuples into flat C arrays
        for i in range(len_a):
            vec = vecs_a[i]
            for k in range(n_feat):
                flat_a[i * n_feat + k] = <double>vec[k]
            # Classify: fricative = [son < -0.5 AND cont > 0.5]
            a_val = flat_a[i * n_feat + son_idx]
            b_val = flat_a[i * n_feat + cont_idx]
            is_fric = (a_val < -0.5 and b_val > 0.5)
            fric_a[i] = is_fric
            # Sibilant = fricative AND strid > 0.5
            sib_a[i] = is_fric and (flat_a[i * n_feat + strid_idx] > 0.5)

        for j in range(len_b):
            vec = vecs_b[j]
            for k in range(n_feat):
                flat_b[j * n_feat + k] = <double>vec[k]
            a_val = flat_b[j * n_feat + son_idx]
            b_val = flat_b[j * n_feat + cont_idx]
            is_fric = (a_val < -0.5 and b_val > 0.5)
            fric_b[j] = is_fric
            sib_b[j] = is_fric and (flat_b[j * n_feat + strid_idx] > 0.5)

        # DP edit distance (two-row)
        for j in range(len_b + 1):
            prev_row[j] = insert_cost * <double>j

        for i in range(1, len_a + 1):
            curr_row[0] = prev_row[0] + delete_cost
            ef = fric_a[i - 1]
            es = sib_a[i - 1]

            for j in range(1, len_b + 1):
                # Combined fric/sib flags for this pair
                ef_pair = ef or fric_b[j - 1]
                es_pair = es or sib_b[j - 1]

                sub_cost = _coarticulated_phoneme_distance_c(
                    &flat_a[(i - 1) * n_feat],
                    &flat_b[(j - 1) * n_feat],
                    n_feat,
                    ef_pair,
                    es_pair,
                    fric_w,
                    sib_w,
                    cont_idx,
                    strid_idx,
                )

                d_val = prev_row[j] + delete_cost
                ins_val = curr_row[j - 1] + insert_cost
                sub_val = prev_row[j - 1] + sub_cost

                if ins_val < d_val:
                    d_val = ins_val
                if sub_val < d_val:
                    d_val = sub_val
                curr_row[j] = d_val

            # Swap rows
            prev_row, curr_row = curr_row, prev_row

        result = prev_row[len_b]

    finally:
        free(flat_a)
        free(flat_b)
        free(fric_a)
        free(fric_b)
        free(sib_a)
        free(sib_b)
        free(prev_row)
        free(curr_row)

    return result
