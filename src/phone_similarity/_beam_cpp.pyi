def beam_state_search_cpp(
    *,
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
    """C++ beam-state expansion kernel.

    Returns ``(node_parent, node_entry, node_raw, node_score, complete_nodes)``.
    """
    ...

def beam_rescore_paths_cpp(
    *,
    source_idx_arr: object,
    source_len: int,
    packed_paths: list[tuple[float, tuple[int, ...], float]],
    offsets_arr: object,
    all_tgt_idx_arr: object,
    dist_flat_arr: object,
    matrix_dim: int,
    max_distance: float = 0.50,
) -> list[tuple[float, tuple[int, ...], float]]:
    """C++ beam rescoring kernel."""
    ...
