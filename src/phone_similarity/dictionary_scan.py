"""
Single- and multi-language dictionary scanning.

Provides :func:`reverse_dictionary_lookup` for scanning a single
pre-tokenized dictionary, :func:`_scan_one_language` as the worker for
parallel scans, and :func:`parallel_dictionary_scan` for fanning out
scans across multiple target languages.
"""

import logging
from typing import Union

from phone_similarity._dispatch import HAS_CYTHON_EXT, HAS_PRANGE
from phone_similarity._dispatch import (
    cy_batch_dictionary_scan as _c_batch_dictionary_scan,
)
from phone_similarity._dispatch import (
    cy_prange_batch_dictionary_scan as _c_prange_batch_dictionary_scan,
)
from phone_similarity.base_bit_array_specification import BaseBitArraySpecification
from phone_similarity.pretokenize import PreTokenizedDictionary
from phone_similarity.primitives import normalised_feature_edit_distance

# Shared pre-filter constants — must match Cython _core.pyx (ratio > 2.0,
# overlap < 0.20) to guarantee identical results across Python/Cython paths.
MAX_LENGTH_RATIO: float = 2.0
MIN_OVERLAP_RATIO: float = 0.20

logger = logging.getLogger(__name__)


def _scan_one_language(args):
    """Worker function for parallel_dictionary_scan.

    Runs in a child process.  Accepts a single tuple to be compatible
    with ``Pool.map``.
    """
    (
        phrases,  # List[(phrase_key, ipa_str)]
        source_spec,
        source_features,
        target_lang,
        _target_spec,
        target_features,
        pre_tokenized,
        top_n,
        max_distance,
    ) = args

    # Re-check Cython availability in the worker process via _dispatch
    # (import once rather than duplicating try/except blocks)
    from phone_similarity._dispatch import HAS_CYTHON_EXT as _worker_has_cython
    from phone_similarity._dispatch import HAS_PRANGE as _worker_has_prange
    from phone_similarity._dispatch import (
        cy_batch_dictionary_scan as _worker_batch_scan,
    )
    from phone_similarity._dispatch import (
        cy_prange_batch_dictionary_scan as _worker_prange_scan,
    )

    results = []
    for phrase_key, source_ipa in phrases:
        merged = {**target_features, **source_features}
        source_tokens = source_spec.ipa_tokenizer(source_ipa)
        source_len = len(source_tokens)
        if source_len == 0:
            continue

        if _worker_has_prange and pre_tokenized is not None:
            matches = _worker_prange_scan(
                source_tokens,
                source_len,
                pre_tokenized,
                merged,
                top_n,
                max_distance,
                0,  # num_threads: let OpenMP decide
            )
        elif _worker_has_cython and pre_tokenized is not None:
            matches = _worker_batch_scan(
                source_tokens,
                source_len,
                pre_tokenized,
                merged,
                top_n,
                max_distance,
            )
        else:
            matches = []
            source_set = set(source_tokens)
            for word, ipa, target_tokens in pre_tokenized:
                target_len = len(target_tokens)
                if target_len == 0:
                    continue
                ratio = max(source_len, target_len) / min(source_len, target_len)
                if ratio > MAX_LENGTH_RATIO:  # FIXME: why are we doing this?
                    continue
                # Phoneme-set overlap pre-filter (matches Cython path)
                target_set = set(target_tokens)
                overlap = len(source_set & target_set)
                min_uniq = min(len(source_set), len(target_set))
                if min_uniq > 0 and overlap / min_uniq < MIN_OVERLAP_RATIO:
                    continue
                d = normalised_feature_edit_distance(
                    source_tokens,
                    target_tokens,
                    merged,
                )
                if d <= max_distance:
                    matches.append((word, ipa, d))
            matches.sort(key=lambda t: t[2])
            matches = matches[:top_n]

        for word, f_ipa, d in matches:
            results.append((phrase_key, target_lang, word, f_ipa, d))

    return results


def parallel_dictionary_scan(
    phrases: list[tuple[str, str]],
    source_spec: BaseBitArraySpecification,
    source_features: dict[str, dict],
    targets: dict[
        str,
        tuple[
            BaseBitArraySpecification,
            dict[str, dict],
            PreTokenizedDictionary,
        ],
    ],
    *,
    top_n: int = 1,
    max_distance: float = 0.50,
    max_workers: int | None = None,
) -> list[tuple[str, str, str, str, float]]:
    """Scan multiple phrases against multiple languages in parallel.

    Fans out one worker per target language using
    ``concurrent.futures.ProcessPoolExecutor``.  Each worker scans *all*
    phrases against its assigned language's pre-tokenized dictionary.

    Parameters
    ----------
    phrases : list of (phrase_key, ipa_str)
        English phrases to search for. ``phrase_key`` is an arbitrary
        string identifier returned in the results.
    source_spec : BaseBitArraySpecification
        Specification for the source (English) phoneme inventory.
    source_features : dict
        ``PHONEME_FEATURES`` of the source language.
    targets : dict
        ``{lang_code: (spec, features, pre_tokenized_dict)}`` for each
        target language.
    top_n : int
        Return at most this many matches per phrase per language.
    max_distance : float
        Distance threshold.
    max_workers : int or None
        Maximum parallel workers (default: number of target languages).

    Returns
    -------
    list of (phrase_key, lang, word, ipa, distance)
        All matches across all languages, unsorted.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if max_workers is None:
        max_workers = len(targets)

    work_items = []
    for lang_code, (t_spec, t_feats, t_ptd) in targets.items():
        work_items.append(
            (
                phrases,
                source_spec,
                source_features,
                lang_code,
                t_spec,
                t_feats,
                t_ptd,
                top_n,
                max_distance,
            )
        )

    all_results: list[tuple[str, str, str, str, float]] = []

    if max_workers <= 1 or len(targets) <= 1:
        # Sequential fallback
        for item in work_items:
            all_results.extend(_scan_one_language(item))
    else:
        # Use 'spawn' context to avoid fork-safety issues with Cython
        # extensions, numpy, and ONNX runtime internal locks.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
            futures = [pool.submit(_scan_one_language, item) for item in work_items]
            for future in as_completed(futures):
                all_results.extend(future.result())

    return all_results


def reverse_dictionary_lookup(
    source_ipa: str,
    source_lang_code: str,
    source_spec: BaseBitArraySpecification,
    source_phoneme_features: dict[str, dict[str, Union[bool, str]]],
    target_lang_code: str,
    target_spec: BaseBitArraySpecification,
    target_phoneme_features: dict[str, dict[str, Union[bool, str]]],
    target_dictionary: dict[str, str],
    *,
    top_n: int = 10,
    max_distance: float = 0.50,
    pre_tokenized: list[tuple[str, str, list[str]]] | None = None,
    num_threads: int = 0,
) -> list[tuple[str, str, float]]:
    """Find dictionary words in a target language closest to a source IPA string.

    For each entry in *target_dictionary*, tokenises its IPA and computes
    the normalised feature edit distance against *source_ipa*.  Returns
    the *top_n* closest words.

    Parameters
    ----------
    source_ipa : str
        Compressed IPA of the source phrase.
    source_lang_code : str
        Language code of the source (used only for logging).
    source_spec : BaseBitArraySpecification
        Specification for tokenising *source_ipa*.
    source_phoneme_features : dict
        ``PHONEME_FEATURES`` of the source language.
    target_lang_code : str
        Language code of the target (used only for logging).
    target_spec : BaseBitArraySpecification
        Specification for tokenising target dictionary IPA.
    target_phoneme_features : dict
        ``PHONEME_FEATURES`` of the target language.
    target_dictionary : dict
        ``{word: ipa_transcription}`` from the target language's G2P
        dictionary.  Values may contain commas (multiple pronunciations);
        only the first pronunciation is used.  Ignored when
        *pre_tokenized* is provided.
    top_n : int
        Return at most this many words (default 10).
    max_distance : float
        Ignore words with distance above this threshold (default 0.50).
    pre_tokenized : list of (word, ipa, tokens), optional
        Pre-tokenized dictionary entries.  When provided the function
        skips tokenization entirely, giving a large speed-up on repeated
        calls against the same dictionary.
    num_threads : int
        Number of OpenMP threads for parallel DP computation.
        0 (default) lets OpenMP choose.  1 forces sequential execution.
        Only effective when the Cython prange extension is available.

    Returns
    -------
    list of (word, ipa, distance)
        Target-language words sorted by ascending distance.
    """
    merged_feats = {**target_phoneme_features, **source_phoneme_features}
    source_tokens = source_spec.ipa_tokenizer(source_ipa)
    source_len = len(source_tokens)
    if source_len == 0:
        return []

    if HAS_PRANGE and pre_tokenized is not None:
        return _c_prange_batch_dictionary_scan(
            source_tokens,
            source_len,
            pre_tokenized,
            merged_feats,
            top_n,
            max_distance,
            num_threads,
        )

    if HAS_CYTHON_EXT and pre_tokenized is not None:
        return _c_batch_dictionary_scan(
            source_tokens,
            source_len,
            pre_tokenized,
            merged_feats,
            top_n,
            max_distance,
        )

    candidates: list[tuple[str, str, float]] = []

    entries = pre_tokenized if pre_tokenized is not None else None

    if entries is None:
        # Tokenize on the fly (slow path)
        entries_iter = []
        for word, raw_ipa in target_dictionary.items():
            ipa = raw_ipa.split(",")[0].strip()
            if not ipa:
                continue
            target_tokens = target_spec.ipa_tokenizer(ipa)
            if not target_tokens:
                continue
            entries_iter.append((word, ipa, target_tokens))
    else:
        entries_iter = entries

    source_set = set(source_tokens)

    for word, ipa, target_tokens in entries_iter:
        target_len = len(target_tokens)
        if target_len == 0:
            continue

        # Quick length-ratio filter
        ratio = max(source_len, target_len) / min(source_len, target_len)
        if ratio > MAX_LENGTH_RATIO:  # TODO FIXME: Why are we doing this??
            continue

        # Phoneme-set overlap pre-filter (matches Cython path)
        target_set = set(target_tokens)
        overlap = len(source_set & target_set)
        min_uniq = min(len(source_set), len(target_set))
        if min_uniq > 0 and overlap / min_uniq < MIN_OVERLAP_RATIO:
            continue

        d = normalised_feature_edit_distance(source_tokens, target_tokens, merged_feats)
        if d <= max_distance:
            candidates.append((word, ipa, d))

    candidates.sort(key=lambda t: t[2])
    return candidates[:top_n]
