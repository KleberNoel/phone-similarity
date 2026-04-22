"""Type stubs for the Cython-accelerated ``_core`` extension module.

These stubs allow mypy / pyright to check callers without needing the
compiled ``.so`` available at type-checking time.
"""

from collections.abc import Sequence
from typing import Union

from bitarray import bitarray

def hamming_distance(a: bitarray, b: bitarray) -> int:
    """Hamming distance (number of differing bits) between two bitarrays.

    Raises ``ValueError`` if lengths differ.
    """
    ...

def hamming_similarity(a: bitarray, b: bitarray) -> float:
    """Normalised bit-match ratio.  1.0 = identical, 0.0 = max different."""
    ...

def feature_edit_distance(
    seq_a: Sequence[str],
    seq_b: Sequence[str],
    phoneme_features: dict[str, dict[str, Union[bool, str]]],
    insert_cost: float = 1.0,
    delete_cost: float = 1.0,
) -> float:
    """Feature-weighted edit distance between two phoneme sequences.

    Substitution cost between phonemes is the normalised feature-mismatch
    ratio.
    """
    ...

def batch_pairwise_hamming(arrays: list[bitarray]) -> list[list[float]]:
    """Pairwise Hamming similarity matrix (N x N) for a list of bitarrays."""
    ...

def phoneme_feature_distance(
    fa: dict[str, Union[bool, str]],
    fb: dict[str, Union[bool, str]],
) -> float:
    """Normalised feature-mismatch ratio between two phoneme feature dicts."""
    ...

def invert_features(
    feature_vector: dict[str, Union[bool, str]],
    target_phoneme_features: dict[str, dict[str, Union[bool, str]]],
    top_n: int = 0,
    max_distance: float = 1.0,
) -> list[tuple[str, float]]:
    """Find phonemes in a target inventory closest to *feature_vector*.

    Returns list of ``(phoneme, distance)`` tuples sorted ascending.
    """
    ...

def cython_ipa_tokenizer(
    ipa_str: str,
    phone_set: frozenset[str],
    max_phoneme_size: int,
) -> list[str]:
    """Tokenize an IPA string using greedy longest-match against a phoneme set."""
    ...

def batch_ipa_tokenize(
    ipa_strings: list[str],
    phone_set: frozenset[str],
    max_phoneme_size: int,
    min_tokens: int = 2,
) -> list[list[str]]:
    """Tokenize a batch of IPA strings in one Cython call.

    Entries below *min_tokens* are returned as empty lists.
    """
    ...

def batch_dictionary_scan(
    source_tokens: list[str],
    source_len: int,
    pre_tokenized: object,
    merged_feats: dict[str, dict[str, Union[bool, str]]],
    top_n: int = 10,
    max_distance: float = 0.50,
) -> list[tuple[str, str, float]]:
    """Scan a pre-tokenized dictionary for entries closest to *source_tokens*.

    Returns list of ``(word, ipa, distance)`` sorted by ascending distance.
    """
    ...

def prange_batch_dictionary_scan(
    source_tokens: list[str],
    source_len: int,
    pre_tokenized: object,
    merged_feats: dict[str, dict[str, Union[bool, str]]],
    top_n: int = 10,
    max_distance: float = 0.50,
    num_threads: int = 0,
) -> list[tuple[str, str, float]]:
    """Parallel dictionary scan using OpenMP prange.

    Same semantics as ``batch_dictionary_scan`` but parallelises the
    per-entry DP computation across multiple OS threads.

    Parameters
    ----------
    num_threads : int
        Number of OpenMP threads.  0 means use OMP_NUM_THREADS or all cores.

    Returns list of ``(word, ipa, distance)`` sorted by ascending distance.
    """
    ...

def beam_expand_candidates(
    source_idx_arr: object,
    source_len: int,
    consumed: int,
    pre_tokenized: object,
    all_tgt_idx_arr: object,
    dist_flat_arr: object,
    matrix_dim: int,
    max_seg_distance: float = 0.50,
    max_len_ratio: float = 3.0,
    min_target_tokens: int = 1,
) -> list[tuple[str, str, int, float]]:
    """Cython beam-expansion kernel over pre-tokenized dictionaries.

    Returns candidate expansions as ``(word, ipa, n_tok, seg_cost)``.
    """
    ...

def beam_expand_candidates_ids(
    source_idx_arr: object,
    source_len: int,
    consumed: int,
    pre_tokenized: object,
    all_tgt_idx_arr: object,
    dist_flat_arr: object,
    matrix_dim: int,
    max_seg_distance: float = 0.50,
    max_len_ratio: float = 3.0,
    min_target_tokens: int = 1,
) -> list[tuple[int, float, int]]:
    """Cython beam-expansion kernel returning compact entry ids.

    Returns candidate expansions as ``(n_tok, seg_cost, entry_id)``.
    """
    ...

def beam_state_search(
    candidates_by_consumed: list[list[tuple[int, float, int]]],
    source_len: int,
    beam_width: int = 10,
    max_words: int = 4,
    max_distance: float = 0.50,
    prune_ratio: float = 2.0,
) -> tuple[
    list[int],
    list[int],
    list[float],
    list[float],
    list[int],
]:
    """Run beam search with struct-of-arrays node storage.

    Returns ``(node_parent, node_entry, node_raw, node_score, complete_nodes)``.
    """
    ...

def beam_collect_complete_paths(
    node_parent: list[int],
    node_entry: list[int],
    node_raw: list[float],
    node_score: list[float],
    complete_nodes: list[int],
    max_candidates: int = 0,
) -> list[tuple[float, tuple[int, ...], float]]:
    """Assemble and deduplicate complete beam paths from node arrays."""
    ...

def beam_rescore_paths(
    source_idx_arr: object,
    source_len: int,
    packed_paths: list[tuple[float, tuple[int, ...], float]],
    pre_tokenized: object,
    all_tgt_idx_arr: object,
    dist_flat_arr: object,
    matrix_dim: int,
    max_distance: float = 0.50,
) -> list[tuple[float, tuple[int, ...], float]]:
    """Rescore complete paths against source indices in Cython."""
    ...
