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

import heapq
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

from phone_similarity.base_bit_array_specification import BaseBitArraySpecification
from phone_similarity.pretokenize import PreTokenizedDictionary
from phone_similarity.primitives import (
    feature_edit_distance,
    normalised_feature_edit_distance,
)


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

    # Pre-filter dictionary: build a list of (word, ipa, tokens, n_tok)
    # keeping only entries with enough tokens.
    dict_entries: list[tuple[str, str, list[str], int]] = []
    for i in range(len(target_ptd)):
        word, ipa, tokens = target_ptd[i]
        n_tok = len(tokens)
        if n_tok >= min_target_tokens:
            dict_entries.append((word, ipa, tokens, n_tok))

    if not dict_entries:
        return []

    # Seed the beam with the initial empty hypothesis
    beam: list[_Hypothesis] = [_Hypothesis(score=0.0, consumed=0, words=(), ipas=(), raw_cost=0.0)]

    # Collect complete hypotheses (consumed == source_len)
    complete: list[_Hypothesis] = []
    best_complete_score = float("inf")

    # Iterative expansion: up to max_words rounds
    for _round in range(max_words):
        if not beam:
            break

        next_beam: list[_Hypothesis] = []

        for hyp in beam:
            remaining = source_len - hyp.consumed
            if remaining <= 0:
                # Already complete — should have been collected
                continue

            # Try every dictionary entry
            for word, ipa, tokens, n_tok in dict_entries:
                # Length ratio filter: the entry shouldn't be wildly
                # longer or shorter than the remaining source segment
                if n_tok > remaining * max_len_ratio:
                    continue
                if remaining > n_tok * max_len_ratio:
                    # Entry is too short relative to what's left,
                    # but only skip if we'd still have room for more
                    # words and this would leave too much unconsumed
                    pass  # allow — might be part of a multi-word chain

                # Sub-sequence: align this entry against the next chunk
                # of source phonemes.  We use the full feature edit distance
                # against the sub-sequence [consumed : consumed + n_tok]
                # (or remaining, whichever is appropriate).
                # The "chunk" is the next min(n_tok, remaining) source tokens
                # but we compare against the entry's full tokens.
                chunk_end = min(hyp.consumed + n_tok, source_len)
                chunk = source_tokens[hyp.consumed : chunk_end]
                chunk_len = len(chunk)

                # Quick length ratio check on the actual alignment
                if chunk_len > 0 and n_tok > 0:
                    ratio = max(chunk_len, n_tok) / min(chunk_len, n_tok)
                    if ratio > max_len_ratio:
                        continue

                # Compute feature edit distance for this segment
                seg_cost = feature_edit_distance(list(chunk), tokens, merged)

                # Normalise by the chunk/entry max length for pruning
                seg_norm = seg_cost / max(chunk_len, n_tok) if max(chunk_len, n_tok) > 0 else 0.0

                new_consumed = chunk_end
                new_raw_cost = hyp.raw_cost + seg_cost
                new_score = new_raw_cost / max(new_consumed, 1)

                # Pruning: skip if already worse than best complete × ratio
                if new_score > best_complete_score * prune_ratio:
                    continue

                new_words = hyp.words + (word,)
                new_ipas = hyp.ipas + (ipa,)

                new_hyp = _Hypothesis(
                    score=new_score,
                    consumed=new_consumed,
                    words=new_words,
                    ipas=new_ipas,
                    raw_cost=new_raw_cost,
                )

                if new_consumed >= source_len:
                    # Complete hypothesis
                    complete.append(new_hyp)
                    if new_score < best_complete_score:
                        best_complete_score = new_score
                elif len(new_words) < max_words:
                    # Partial — can still expand
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
    source_ipa_joined = "".join(source_tokens)
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
