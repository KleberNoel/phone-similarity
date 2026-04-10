"""
Beam search for multi-word phonological segmentation.

Given a target phoneme sequence (e.g. from an English phrase) and a foreign
language's pre-tokenized dictionary, find the top-K segmentations into
foreign words that minimise total phonological distance.

This replaces the greedy per-word "glue" approach in the example script with
a principled search over the combinatorial space of segmentations.

Algorithm
---------
Maintain a beam of width *B* partial hypotheses, each represented as::

    (consumed, words, ipas, raw_cost, n_words)

where *consumed* is the number of source phonemes explained so far.  At each
step, expand every hypothesis by trying all dictionary entries whose token
length is compatible with the remaining unconsumed segment, compute the
feature edit distance for the sub-sequence alignment, and keep the top-B
expansions.  Complete hypotheses (consumed == len(source)) are scored by
``raw_cost / consumed`` (length-normalised) and collected separately.

Pruning
-------
* **Admissible bound** -- discard partial hypotheses whose normalised cost
  already exceeds ``prune_ratio × best_complete_score`` (default 2.0).
* **Max words** -- discard hypotheses that exceed *max_words* (default 4).
* **Length ratio** -- skip dictionary entries whose token count differs from
  the remaining source sub-sequence by more than ``max_len_ratio`` (default 3.0).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Union

from phone_similarity.base_bit_array_specification import BaseBitArraySpecification
from phone_similarity.pretokenize import PreTokenizedDictionary
from phone_similarity.primitives import (
    normalised_feature_edit_distance,
    phoneme_feature_distance,
)

# ===================================================================
# Phoneme trie for fast approximate dictionary matching
# ===================================================================


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
    merged: dict[str, dict],
    max_seg_distance: float,
    max_len_ratio: float,
    min_target_tokens: int,
) -> list[tuple[str, str, int, float]]:
    """Search the trie for entries matching source_tokens[consumed:].

    Walks the trie depth-first, maintaining one column of the edit-distance
    DP matrix at each level.  When the minimum value in a column exceeds
    the cost ceiling, the entire subtree is pruned.

    Returns
    -------
    list of (word, ipa, n_tok, seg_cost)
        Entries whose per-segment normalised feature edit distance is
        within *max_seg_distance*.
    """
    source_len = len(source_tokens)
    remaining = source_len - consumed
    if remaining <= 0:
        return []

    max_depth = int(remaining * max_len_ratio) + 1
    # Conservative cost ceiling: any valid result satisfies
    # seg_cost / max(chunk_len, entry_len) <= max_seg_distance.
    # The largest possible denominator is max(remaining, max_depth).
    cost_ceil = max_seg_distance * max(remaining, max_depth)

    # Pre-fetch source feature dicts for the remaining tokens
    src_feats = [merged.get(source_tokens[consumed + i], {}) for i in range(remaining)]

    # Initial DP column (depth 0): source prefix vs empty target = deletions
    init_col = [float(i) for i in range(remaining + 1)]

    results: list[tuple[str, str, int, float]] = []

    # DFS stack: (node, depth, dp_column)
    stack: list[tuple[_TrieNode, int, list[float]]] = [(root, 0, init_col)]

    while stack:
        node, depth, col = stack.pop()

        # Collect complete entries at this node
        if node.entries and depth >= min_target_tokens:
            chunk_len = min(depth, remaining)
            seg_cost = col[chunk_len]
            denom = max(chunk_len, depth)
            if denom > 0 and seg_cost / denom <= max_seg_distance:
                for word, ipa in node.entries:
                    results.append((word, ipa, depth, seg_cost))

        if depth >= max_depth:
            continue

        # Expand children: compute next DP column for each phoneme edge
        for phoneme, child in node.children.items():
            tgt_feat = merged.get(phoneme, {})

            new_col = [0.0] * (remaining + 1)
            new_col[0] = col[0] + 1.0  # insert target phoneme

            min_val = new_col[0]

            for j in range(1, remaining + 1):
                sub = phoneme_feature_distance(src_feats[j - 1], tgt_feat)
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
class BeamResult:
    """A single complete segmentation from beam search.

    Attributes
    ----------
    words : tuple of str
        Foreign words comprising the segmentation.
    ipas : tuple of str
        IPA transcriptions of each foreign word.
    glued_ipa : str
        Concatenated IPA of all foreign words.
    distance : float
        Length-normalised feature edit distance of the *entire* glued
        foreign IPA against the *entire* source IPA.  This is the "true"
        end-to-end score, recomputed after beam search finishes (not
        the sum of per-segment costs).
    segment_cost : float
        Sum of per-segment feature edit distances (the cost that the beam
        search optimised).  Useful for diagnostics but ``distance`` is the
        canonical quality metric.
    """

    words: tuple[str, ...]
    ipas: tuple[str, ...]
    glued_ipa: str
    distance: float
    segment_cost: float


def beam_search_segmentation(
    source_tokens: Sequence[str],
    source_spec: BaseBitArraySpecification,
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
) -> list[BeamResult]:
    """Find the best multi-word foreign segmentations for a source phoneme sequence.

    Parameters
    ----------
    source_tokens : sequence of str
        Tokenised IPA of the source phrase (e.g. English).
    source_spec : BaseBitArraySpecification
        Specification for the source language (used only for end-to-end
        re-scoring via ``ipa_tokenizer``).
    source_features : dict
        ``PHONEME_FEATURES`` of the source language.
    target_ptd : PreTokenizedDictionary
        Pre-tokenized foreign dictionary to segment into.
    target_spec : BaseBitArraySpecification
        Specification for the target language (used for tokenising the
        glued IPA in end-to-end re-scoring).
    target_features : dict
        ``PHONEME_FEATURES`` of the target language.
    beam_width : int
        Number of partial hypotheses to keep at each expansion step.
    top_k : int
        Return at most this many complete segmentations.
    max_words : int
        Maximum number of foreign words per segmentation.
    max_distance : float
        Discard final segmentations whose end-to-end normalised distance
        exceeds this threshold.
    prune_ratio : float
        Prune partial hypotheses whose normalised cost exceeds
        ``prune_ratio * best_complete_normalised_cost``.
    max_len_ratio : float
        Skip dictionary entries whose token count differs from the
        remaining source sub-sequence by more than this ratio.
    min_target_tokens : int
        Minimum token count for a dictionary entry to be considered
        (default 1; set to 2 to avoid single-phoneme words).

    Returns
    -------
    list of BeamResult
        Up to *top_k* complete segmentations sorted by ascending
        ``distance`` (end-to-end normalised feature edit distance).
    """
    source_len = len(source_tokens)
    if source_len == 0:
        return []

    merged = {**target_features, **source_features}

    # ── Build phoneme trie for fast approximate matching ──
    trie_root = _build_trie(target_ptd, min_target_tokens)
    if not trie_root.children:
        return []

    # Seed the beam with the initial empty hypothesis
    beam: list[_Hypothesis] = [_Hypothesis(score=0.0, consumed=0, words=(), ipas=(), raw_cost=0.0)]

    # Collect complete hypotheses (consumed == source_len)
    complete: list[_Hypothesis] = []
    best_complete_score = float("inf")

    # Pre-compute the score ceiling: if we already have a good complete
    # hypothesis, we can prune aggressively.
    score_ceil = max_distance * prune_ratio  # initial ceiling

    # Iterative expansion: up to max_words rounds
    for _round in range(max_words):
        if not beam:
            break

        next_beam: list[_Hypothesis] = []

        # Cache trie expansion results by consumed position within a round.
        # Multiple hypotheses at the same consumed position will produce the
        # same expansion candidates (different only in accumulated cost).
        expand_cache: dict[int, list[tuple[str, str, int, float]]] = {}

        for hyp in beam:
            remaining = source_len - hyp.consumed
            if remaining <= 0:
                continue

            # Lookup or compute expansions for this consumed position
            candidates = expand_cache.get(hyp.consumed)
            if candidates is None:
                candidates = _trie_expand(
                    trie_root,
                    source_tokens,
                    hyp.consumed,
                    merged,
                    max_distance,
                    max_len_ratio,
                    min_target_tokens,
                )
                expand_cache[hyp.consumed] = candidates

            for word, ipa, n_tok, seg_cost in candidates:
                new_consumed = min(hyp.consumed + n_tok, source_len)
                new_raw_cost = hyp.raw_cost + seg_cost
                new_score = new_raw_cost / max(new_consumed, 1)

                # Pruning: skip if already worse than best complete × ratio
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

        # Keep only the top beam_width hypotheses
        if len(next_beam) > beam_width:
            next_beam.sort()
            next_beam = next_beam[:beam_width]

        beam = next_beam

    if not complete:
        return []

    # De-duplicate: keep the best score per unique word-tuple
    seen: dict[tuple[str, ...], _Hypothesis] = {}
    for hyp in complete:
        existing = seen.get(hyp.words)
        if existing is None or hyp.score < existing.score:
            seen[hyp.words] = hyp
    unique_complete = sorted(seen.values())

    # Re-score the top candidates end-to-end: concatenate foreign IPAs
    # and compute normalised feature edit distance against the full source.
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
    """Run beam search segmentation for multiple phrases across multiple languages.

    This is the batch entry point, analogous to
    :func:`~phone_similarity.dictionary_scan.parallel_dictionary_scan` but
    for multi-word segmentation.

    Parameters
    ----------
    phrases : list of (phrase_key, ipa_str)
        Source phrases.  *phrase_key* is an arbitrary identifier returned
        in the results; *ipa_str* is the compressed IPA of the phrase.
    source_spec : BaseBitArraySpecification
        Specification for the source language.
    source_features : dict
        ``PHONEME_FEATURES`` of the source language.
    targets : dict
        ``{lang_code: (spec, features, pre_tokenized_dict)}``.
    beam_width, top_k, max_words, max_distance, prune_ratio,
    max_len_ratio, min_target_tokens :
        Forwarded to :func:`beam_search_segmentation`.

    Returns
    -------
    list of (phrase_key, lang_code, BeamResult)
        All results across all phrases and languages, sorted by ascending
        ``BeamResult.distance``.
    """
    all_results: list[tuple[str, str, BeamResult]] = []

    for phrase_key, ipa_str in phrases:
        source_tokens = source_spec.ipa_tokenizer(ipa_str)
        if not source_tokens:
            continue

        for lang_code, (t_spec, t_features, t_ptd) in targets.items():
            segmentations = beam_search_segmentation(
                source_tokens,
                source_spec,
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
            )
            for result in segmentations:
                all_results.append((phrase_key, lang_code, result))

    all_results.sort(key=lambda t: t[2].distance)
    return all_results
