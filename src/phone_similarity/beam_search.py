"""
Beam search for multi-word phonological segmentation.

Maintains a beam of partial hypotheses, expanding each by trying dictionary entries
whose token length is compatible with the remaining source sub-sequence.  Complete
hypotheses are length-normalised and re-scored end-to-end.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from phone_similarity._dispatch import (
    HAS_CPP_BEAM_STATE,
    HAS_CYTHON_BEAM_EXPAND,
    HAS_CYTHON_BEAM_STATE,
    HAS_CYTHON_DIST_MATRIX,
    cpp_beam_state_search,
    cy_beam_collect_complete_paths,
    cy_beam_expand_candidates,
    cy_beam_expand_candidates_ids,
    cy_beam_state_search,
    cy_build_phoneme_dist_matrix,
)
from phone_similarity.base_bit_array_specification import BaseBitArraySpecification
from phone_similarity.gpu_rescore import gpu_rescore_paths
from phone_similarity.pretokenize import PreTokenizedDictionary
from phone_similarity.primitives import normalised_feature_edit_distance, phoneme_feature_distance


def _build_dist_matrix(
    merged: dict[str, dict],
) -> tuple[dict[str, int], list[float], int]:
    """Build a flat phoneme distance matrix for all phonemes in *merged*."""
    if HAS_CYTHON_DIST_MATRIX:
        return cy_build_phoneme_dist_matrix(merged)

    phonemes = sorted(merged)
    n = len(phonemes)
    dim = n + 1  # extra row/col for unknown phonemes
    ph_to_idx = {p: i for i, p in enumerate(phonemes)}

    dist_flat = [0.0] * (dim * dim)

    for i in range(n):
        fi = merged[phonemes[i]]
        for j in range(i, n):
            fj = merged[phonemes[j]]
            d = phoneme_feature_distance(fi, fj)
            dist_flat[i * dim + j] = d
            dist_flat[j * dim + i] = d

    # UNK row/col: distance 1.0 to everything, 0.0 to itself
    unk = n  # index of UNK sentinel
    for i in range(dim):
        dist_flat[i * dim + unk] = 1.0
        dist_flat[unk * dim + i] = 1.0
    dist_flat[unk * dim + unk] = 0.0

    return ph_to_idx, dist_flat, dim


# Phoneme trie for fast approximate dictionary matching


class _TrieNode:
    """Node in a phoneme trie.  Leaf nodes store dictionary entries."""

    __slots__ = ("children", "entries")

    def __init__(self):
        self.children: dict[str, _TrieNode] = {}
        self.entries: list[tuple[str, str]] = []  # (word, ipa)


def _build_trie(ptd: PreTokenizedDictionary, min_tokens: int) -> _TrieNode:
    """Build a phoneme trie from a pre-tokenized dictionary.

    Each root-to-leaf path corresponds to an entry's phoneme token
    sequence.  Shared prefixes are stored once.
    """
    root = _TrieNode()
    for i in range(len(ptd)):
        word, ipa, tokens = ptd[i]
        if len(tokens) < min_tokens:
            continue
        node = root
        for tok in tokens:
            child = node.children.get(tok)
            if child is None:
                child = _TrieNode()
                node.children[tok] = child
            node = child
        node.entries.append((word, ipa))
    return root


def _trie_expand(
    root: _TrieNode,
    source_tokens: Sequence[str],
    consumed: int,
    max_seg_distance: float,
    max_len_ratio: float,
    min_target_tokens: int,
    ph_to_idx: dict[str, int],
    dist_flat: list[float],
    dim: int,
) -> list[tuple[str, str, int, float]]:
    """Walk trie depth-first returning entries within *max_seg_distance* of source_tokens[consumed:]."""
    source_len = len(source_tokens)
    remaining = source_len - consumed
    if remaining <= 0:
        return []

    max_depth = int(remaining * max_len_ratio) + 1
    # Conservative cost ceiling: any valid result satisfies
    # seg_cost / max(chunk_len, entry_len) <= max_seg_distance.
    # The largest possible denominator is max(remaining, max_depth).
    cost_ceil = max_seg_distance * max(remaining, max_depth)

    unk_idx = dim - 1  # UNK sentinel is the last row/col

    # Pre-resolve source token indices for the remaining segment
    src_indices = [ph_to_idx.get(source_tokens[consumed + i], unk_idx) for i in range(remaining)]

    # Initial DP column (depth 0): source prefix vs empty target = deletions
    init_col = [float(i) for i in range(remaining + 1)]

    results: list[tuple[str, str, int, float]] = []

    # DFS stack: (node, depth, dp_column)
    stack: list[tuple[_TrieNode, int, list[float]]] = [(root, 0, init_col)]

    while stack:
        node, depth, col = stack.pop()

        if node.entries and depth >= min_target_tokens:
            chunk_len = min(depth, remaining)
            seg_cost = col[chunk_len]
            denom = max(chunk_len, depth)
            if denom > 0 and seg_cost / denom <= max_seg_distance:
                for word, ipa in node.entries:
                    results.append((word, ipa, depth, seg_cost))

        if depth >= max_depth:
            continue

        for phoneme, child in node.children.items():
            tgt_idx = ph_to_idx.get(phoneme, unk_idx)

            new_col = [0.0] * (remaining + 1)
            new_col[0] = col[0] + 1.0  # insert target phoneme

            min_val = new_col[0]

            for j in range(1, remaining + 1):
                sub = dist_flat[src_indices[j - 1] * dim + tgt_idx]
                val = min(
                    new_col[j - 1] + 1.0,  # delete source phoneme
                    col[j] + 1.0,  # insert target phoneme
                    col[j - 1] + sub,  # substitute
                )
                new_col[j] = val
                if val < min_val:
                    min_val = val

            # Prune: if no alignment in this subtree can be good enough
            if min_val > cost_ceil:
                continue

            stack.append((child, depth + 1, new_col))

    return results


def _build_expand_cache(
    source_len: int,
    source_idx_arr: np.ndarray,
    source_tokens: Sequence[str],
    target_ptd: PreTokenizedDictionary,
    trie_root: _TrieNode,
    ph_to_idx: dict[str, int],
    dist_flat: list[float],
    resources: "BeamSearchResources",
    dim: int,
    max_distance: float,
    max_len_ratio: float,
    min_target_tokens: int,
    consumed_positions: Sequence[int],
) -> dict[int, list[tuple[str, str, int, float]]]:
    cache: dict[int, list[tuple[str, str, int, float]]] = {}
    for consumed_pos in consumed_positions:
        if HAS_CYTHON_BEAM_EXPAND:
            candidates = cy_beam_expand_candidates(
                source_idx_arr,
                source_len,
                consumed_pos,
                target_ptd,
                resources.all_tgt_idx_arr,
                resources.dist_flat_arr,
                dim,
                max_distance,
                max_len_ratio,
                min_target_tokens,
            )
        else:
            candidates = _trie_expand(
                trie_root,
                source_tokens,
                consumed_pos,
                max_distance,
                max_len_ratio,
                min_target_tokens,
                ph_to_idx,
                dist_flat,
                dim,
            )
        cache[consumed_pos] = candidates
    return cache


@dataclass(order=True)
class _Hypothesis:
    """A partial segmentation hypothesis on the beam.

    Ordered by ``score`` (normalised cost) so that ``heapq`` operations
    give us the lowest-cost hypotheses first.
    """

    score: float  # normalised cost: raw_cost / max(consumed, 1)
    consumed: int = field(compare=False)
    words: tuple[str, ...] = field(compare=False)
    ipas: tuple[str, ...] = field(compare=False)
    raw_cost: float = field(compare=False)


@dataclass()
class BeamSearchResources:
    """Precomputed resources reused across beam-search calls.

    Building the phoneme-distance matrix and trie dominates startup cost for
    large dictionaries. Reusing these resources across many source phrases
    avoids rebuilding the same structures repeatedly.
    """

    merged_features: dict[str, dict[str, Union[bool, str]]]
    trie_root: _TrieNode
    ph_to_idx: dict[str, int]
    dist_flat: list[float]
    dist_flat_arr: np.ndarray
    all_tgt_idx_arr: np.ndarray
    words: list[str]
    ipas: list[str]
    word_to_id: dict[str, int]
    dim: int
    min_target_tokens: int
    target_ptd: PreTokenizedDictionary


def build_beam_search_resources(
    source_features: dict[str, dict[str, Union[bool, str]]],
    target_ptd: PreTokenizedDictionary,
    target_features: dict[str, dict[str, Union[bool, str]]],
    *,
    min_target_tokens: int = 1,
) -> BeamSearchResources:
    """Precompute trie + distance matrix for repeated segmentations.

    This helper is useful when many source phrases are searched against the
    same target dictionary/language pair.
    """
    merged = {**target_features, **source_features}
    ph_to_idx, dist_flat, dim = _build_dist_matrix(merged)
    trie_root = _build_trie(target_ptd, min_target_tokens)

    unk_idx = dim - 1
    inv_to_mat = [ph_to_idx.get(tok, unk_idx) for tok in target_ptd.inventory]

    inv_to_mat_arr = np.ascontiguousarray(np.asarray(inv_to_mat, dtype=np.intp))
    token_indices = np.ascontiguousarray(np.asarray(target_ptd.token_indices, dtype=np.intp))
    all_tgt_idx_arr = np.ascontiguousarray(inv_to_mat_arr[token_indices], dtype=np.intp)
    dist_flat_arr = np.ascontiguousarray(np.asarray(dist_flat, dtype=np.float64))
    words = list(target_ptd.words)
    ipas = list(target_ptd.ipas)
    word_to_id = {w: i for i, w in enumerate(words)}

    return BeamSearchResources(
        merged_features=merged,
        trie_root=trie_root,
        ph_to_idx=ph_to_idx,
        dist_flat=dist_flat,
        dist_flat_arr=dist_flat_arr,
        all_tgt_idx_arr=all_tgt_idx_arr,
        words=words,
        ipas=ipas,
        word_to_id=word_to_id,
        dim=dim,
        min_target_tokens=min_target_tokens,
        target_ptd=target_ptd,
    )


def _build_source_idx(
    source_tokens: Sequence[str], ph_to_idx: dict[str, int], unk_idx: int
) -> list[int]:
    return [ph_to_idx.get(tok, unk_idx) for tok in source_tokens]


@dataclass()
class BeamResult:
    """A single complete segmentation from beam search."""

    words: tuple[str, ...]
    ipas: tuple[str, ...]
    glued_ipa: str
    distance: float
    segment_cost: float


def beam_search_segmentation(
    source_tokens: Sequence[str],
    source_features: dict[str, dict[str, Union[bool, str]]],
    target_ptd: PreTokenizedDictionary,
    target_spec: BaseBitArraySpecification,
    target_features: dict[str, dict[str, Union[bool, str]]],
    *,
    beam_width: int = 10,
    top_k: int = 5,
    max_words: int = 4,
    max_distance: float = 0.50,
    prune_ratio: float = 2.0,
    max_len_ratio: float = 3.0,
    min_target_tokens: int = 1,
    resources: BeamSearchResources | None = None,
) -> list[BeamResult]:
    """Find the best multi-word foreign segmentations for a source phoneme sequence."""
    source_len = len(source_tokens)
    if source_len == 0:
        return []

    if resources is None:
        resources = build_beam_search_resources(
            source_features,
            target_ptd,
            target_features,
            min_target_tokens=min_target_tokens,
        )
    elif resources.min_target_tokens != min_target_tokens:
        raise ValueError(
            "min_target_tokens does not match precomputed resources "
            f"({min_target_tokens} != {resources.min_target_tokens})"
        )

    merged = resources.merged_features
    ph_to_idx = resources.ph_to_idx
    dim = resources.dim
    trie_root = resources.trie_root
    if resources.target_ptd is not target_ptd:
        raise ValueError("resources were built for a different target_ptd object")

    if not trie_root.children:
        return []

    unk_idx = dim - 1
    source_idx = _build_source_idx(source_tokens, ph_to_idx, unk_idx)
    source_idx_arr = np.ascontiguousarray(np.asarray(source_idx, dtype=np.intp))

    if HAS_CYTHON_BEAM_STATE:
        candidates_by_consumed_ids: list[list[tuple[int, float, int]]] = [
            [] for _ in range(source_len + 1)
        ]
        for consumed_pos in range(source_len):
            if HAS_CYTHON_BEAM_EXPAND:
                candidates = cy_beam_expand_candidates_ids(
                    source_idx_arr,
                    source_len,
                    consumed_pos,
                    target_ptd,
                    resources.all_tgt_idx_arr,
                    resources.dist_flat_arr,
                    dim,
                    max_distance,
                    max_len_ratio,
                    min_target_tokens,
                )
                if candidates:
                    candidates_by_consumed_ids[consumed_pos] = candidates
            else:
                candidates = _trie_expand(
                    trie_root,
                    source_tokens,
                    consumed_pos,
                    max_distance,
                    max_len_ratio,
                    min_target_tokens,
                    ph_to_idx,
                    resources.dist_flat,
                    dim,
                )
                if candidates:
                    candidates_by_consumed_ids[consumed_pos] = [
                        (n_tok, seg_cost, resources.word_to_id[word])
                        for word, _ipa, n_tok, seg_cost in candidates
                    ]

        if not any(candidates_by_consumed_ids):
            return []

        if HAS_CPP_BEAM_STATE:
            (
                node_parent,
                node_entry,
                node_raw,
                node_score,
                complete_nodes,
            ) = cpp_beam_state_search(
                candidates_by_consumed=candidates_by_consumed_ids,
                source_len=source_len,
                beam_width=beam_width,
                max_words=max_words,
                max_distance=max_distance,
                prune_ratio=prune_ratio,
            )
        else:
            (
                node_parent,
                node_entry,
                node_raw,
                node_score,
                complete_nodes,
            ) = cy_beam_state_search(
                candidates_by_consumed_ids,
                source_len,
                beam_width=beam_width,
                max_words=max_words,
                max_distance=max_distance,
                prune_ratio=prune_ratio,
            )

        packed = cy_beam_collect_complete_paths(
            node_parent,
            node_entry,
            node_raw,
            node_score,
            complete_nodes,
            max_candidates=top_k * 3,
        )

        if not packed:
            return []

        rescored = gpu_rescore_paths(
            source_idx_arr=source_idx_arr,
            source_len=source_len,
            packed_paths=packed,
            pre_tokenized=target_ptd,
            all_tgt_idx_arr=resources.all_tgt_idx_arr,
            dist_flat_arr=resources.dist_flat_arr,
            matrix_dim=dim,
            max_distance=max_distance,
            backend="auto",
        )

        if not rescored:
            return []

        seen_words: set[tuple[str, ...]] = set()
        results: list[BeamResult] = []
        for e2e_dist, entry_ids, raw_cost in rescored:
            words = tuple(resources.words[eid] for eid in entry_ids)
            if words in seen_words:
                continue
            seen_words.add(words)

            ipas = tuple(resources.ipas[eid] for eid in entry_ids)
            glued_ipa = "".join(ipas)
            results.append(
                BeamResult(
                    words=words,
                    ipas=ipas,
                    glued_ipa=glued_ipa,
                    distance=e2e_dist,
                    segment_cost=raw_cost,
                )
            )
            if len(results) >= top_k:
                break

        results.sort(key=lambda r: r.distance)
        return results[:top_k]

    beam: list[_Hypothesis] = [_Hypothesis(score=0.0, consumed=0, words=(), ipas=(), raw_cost=0.0)]

    complete: list[_Hypothesis] = []
    best_complete_score = float("inf")

    score_ceil = max_distance * prune_ratio

    for _ in range(max_words):
        if not beam:
            break

        next_beam: list[_Hypothesis] = []

        active_consumed = sorted({h.consumed for h in beam if h.consumed < source_len})
        expand_cache = _build_expand_cache(
            source_len,
            source_idx_arr,
            source_tokens,
            target_ptd,
            trie_root,
            ph_to_idx,
            resources.dist_flat,
            resources,
            dim,
            max_distance,
            max_len_ratio,
            min_target_tokens,
            active_consumed,
        )

        for hyp in beam:
            remaining = source_len - hyp.consumed
            if remaining <= 0:
                continue

            candidates = expand_cache.get(hyp.consumed, [])

            for word, ipa, n_tok, seg_cost in candidates:
                new_consumed = min(hyp.consumed + n_tok, source_len)
                new_raw_cost = hyp.raw_cost + seg_cost
                new_score = new_raw_cost / max(new_consumed, 1)

                if new_score > score_ceil:
                    continue

                new_hyp = _Hypothesis(
                    score=new_score,
                    consumed=new_consumed,
                    words=(*hyp.words, word),
                    ipas=(*hyp.ipas, ipa),
                    raw_cost=new_raw_cost,
                )

                if new_consumed >= source_len:
                    complete.append(new_hyp)
                    if new_score < best_complete_score:
                        best_complete_score = new_score
                        score_ceil = best_complete_score * prune_ratio
                elif len(new_hyp.words) < max_words:
                    next_beam.append(new_hyp)

        if len(next_beam) > beam_width:
            next_beam.sort()
            next_beam = next_beam[:beam_width]

        beam = next_beam

    if not complete:
        return []

    seen: dict[tuple[str, ...], _Hypothesis] = {}
    for hyp in complete:
        existing = seen.get(hyp.words)
        if existing is None or hyp.score < existing.score:
            seen[hyp.words] = hyp
    unique_complete = sorted(seen.values())

    results: list[BeamResult] = []

    for hyp in unique_complete[: top_k * 3]:  # evaluate extra for filtering
        glued_ipa = "".join(hyp.ipas)
        glued_tokens = target_spec.ipa_tokenizer(glued_ipa)

        if not glued_tokens:
            continue

        e2e_dist = normalised_feature_edit_distance(list(source_tokens), glued_tokens, merged)

        if e2e_dist <= max_distance:
            results.append(
                BeamResult(
                    words=hyp.words,
                    ipas=hyp.ipas,
                    glued_ipa=glued_ipa,
                    distance=e2e_dist,
                    segment_cost=hyp.raw_cost,
                )
            )

    results.sort(key=lambda r: r.distance)
    return results[:top_k]


def beam_search_phrases(
    phrases: list[tuple[str, str]],
    source_spec: BaseBitArraySpecification,
    source_features: dict[str, dict[str, Union[bool, str]]],
    targets: dict[
        str,
        tuple[
            BaseBitArraySpecification,
            dict[str, dict[str, Union[bool, str]]],
            PreTokenizedDictionary,
        ],
    ],
    *,
    beam_width: int = 10,
    top_k: int = 1,
    max_words: int = 4,
    max_distance: float = 0.50,
    prune_ratio: float = 2.0,
    max_len_ratio: float = 3.0,
    min_target_tokens: int = 1,
) -> list[tuple[str, str, BeamResult]]:
    """Run beam search segmentation for multiple phrases across multiple languages."""
    all_results: list[tuple[str, str, BeamResult]] = []

    target_resources: dict[str, BeamSearchResources] = {}
    for lang_code, (_t_spec, t_features, t_ptd) in targets.items():
        target_resources[lang_code] = build_beam_search_resources(
            source_features,
            t_ptd,
            t_features,
            min_target_tokens=min_target_tokens,
        )

    for phrase_key, ipa_str in phrases:
        source_tokens = source_spec.ipa_tokenizer(ipa_str)
        if not source_tokens:
            continue

        for lang_code, (t_spec, t_features, t_ptd) in targets.items():
            segmentations = beam_search_segmentation(
                source_tokens,
                source_features,
                t_ptd,
                t_spec,
                t_features,
                beam_width=beam_width,
                top_k=top_k,
                max_words=max_words,
                max_distance=max_distance,
                prune_ratio=prune_ratio,
                max_len_ratio=max_len_ratio,
                min_target_tokens=min_target_tokens,
                resources=target_resources[lang_code],
            )
            for result in segmentations:
                all_results.append((phrase_key, lang_code, result))

    all_results.sort(key=lambda t: t[2].distance)
    return all_results
