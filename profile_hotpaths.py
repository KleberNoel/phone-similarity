#!/usr/bin/env python3
"""Profiling script for phone-similarity hot paths.

Exercises the realistic workload without network or NLP dependencies:
  1. Language resource loading (specs, features, G2P dicts)
  2. Pre-tokenized dictionary cache loading
  3. IPA tokenization (Cython vs Python)
  4. Feature edit distance computation
  5. Dictionary scanning (batch_dictionary_scan via Cython)
  6. Feature inversion (invert_ipa)
  7. Parallel dictionary scan (multi-language)

Run with:
    scalene --json --outfile profile_results.json profile_hotpaths.py
    scalene --html --outfile profile_results.html profile_hotpaths.py
    # or plain:
    python profile_hotpaths.py
"""

from __future__ import annotations

import gc
import importlib
import logging
import sys
import time
import tracemalloc

from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.clean_phones import clean_phones
from phone_similarity.dictionary_scan import reverse_dictionary_lookup
from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator
from phone_similarity.inversion import invert_ipa
from phone_similarity.pretokenize import (
    PreTokenizedDictionary,
    cached_pretokenize_dictionary,
)
from phone_similarity.primitives import (
    _HAS_CYTHON,
    _HAS_CYTHON_EXT,
    normalised_feature_edit_distance,
)

logging.disable(logging.WARNING)

# ── Configuration ─────────────────────────────────────────────────────
TARGET_LANGUAGES = ["fra", "ger", "spa", "ita", "dut"]
SOURCE_LANG = "eng_us"
# English test phrases (word, IPA) — covers short and long inputs
TEST_PHRASES: list[tuple[str, str]] = [
    ("cat", "kæt"),
    ("hello", "hɛloʊ"),
    ("beautiful", "bjuːtɪfəl"),
    ("international", "ɪntɚnæʃənəl"),
    ("chocolate", "tʃɑːklət"),
]

N_SCAN_ITERATIONS = 1  # repeat scans to amplify signal


def _timer(label: str):
    """Context manager to time a section."""

    class Timer:
        def __init__(self):
            self.elapsed = 0.0

        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *_):
            self.elapsed = time.perf_counter() - self._t0
            print(f"  [{label}] {self.elapsed:.3f}s")

    return Timer()


def load_language_resources(lang_code: str):
    """Load spec, features, G2P for one language."""
    mod_name = lang_code.replace("-", "_")
    lang_module = importlib.import_module(f"phone_similarity.language.{mod_name}")
    phoneme_features = lang_module.PHONEME_FEATURES
    vowels_set = lang_module.VOWELS_SET
    consonants_set = set(p for p in phoneme_features if p not in vowels_set)
    spec = BitArraySpecification(
        vowels=vowels_set,
        consonants=consonants_set,
        features_per_phoneme=phoneme_features,
        features=lang_module.FEATURES,
    )
    return spec, phoneme_features


def main():
    tracemalloc.start()
    print(f"Cython core: {_HAS_CYTHON}, Cython ext: {_HAS_CYTHON_EXT}")
    print(f"Python {sys.version}")
    print()

    # ── Step 1: Load source language ──────────────────────────────────
    print("=== Step 1: Load language resources ===")
    with _timer("source (eng_us)"):
        source_spec, source_features = load_language_resources(SOURCE_LANG)

    target_specs: dict[str, BitArraySpecification] = {}
    target_features: dict[str, dict] = {}
    for lang in TARGET_LANGUAGES:
        with _timer(f"target ({lang})"):
            spec, feats = load_language_resources(lang)
            target_specs[lang] = spec
            target_features[lang] = feats

    tracemalloc.take_snapshot()
    mem1 = tracemalloc.get_traced_memory()
    print(
        f"  Memory after specs: current={mem1[0] / 1024 / 1024:.1f}MB, peak={mem1[1] / 1024 / 1024:.1f}MB"
    )
    print()

    # ── Step 2: Load pre-tokenized dictionaries ───────────────────────
    print("=== Step 2: Load pre-tokenized dictionaries ===")
    target_ptds: dict[str, PreTokenizedDictionary] = {}
    g2ps: dict[str, CharsiuGraphemeToPhonemeGenerator] = {}
    for lang in TARGET_LANGUAGES:
        g2p_code = lang.replace("_", "-")
        g2p = CharsiuGraphemeToPhonemeGenerator(g2p_code, use_cache=True)
        g2ps[lang] = g2p
        with _timer(f"cache load ({lang})"):
            ptd = cached_pretokenize_dictionary(
                lambda g=g2p: g.pdict,
                target_specs[lang],
                lang=lang,
            )
            target_ptds[lang] = ptd
        print(f"    {lang}: {len(ptd)} entries")

    tracemalloc.take_snapshot()
    mem2 = tracemalloc.get_traced_memory()
    print(
        f"  Memory after PTDs: current={mem2[0] / 1024 / 1024:.1f}MB, peak={mem2[1] / 1024 / 1024:.1f}MB"
    )
    print()

    # ── Step 3: IPA tokenization benchmark ────────────────────────────
    print("=== Step 3: IPA tokenization (1000 iterations per phrase) ===")
    all_tokens = []
    with _timer("tokenize all phrases x1000"):
        for _ in range(1000):
            for _name, ipa in TEST_PHRASES:
                tokens = source_spec.ipa_tokenizer(clean_phones(ipa))
                all_tokens.append(tokens)
    del all_tokens
    gc.collect()
    print()

    # ── Step 4: Feature edit distance benchmark ───────────────────────
    print("=== Step 4: Feature edit distance (pairwise, 200 iterations) ===")
    # Prepare token sequences
    token_seqs = []
    for _name, ipa in TEST_PHRASES:
        tokens = source_spec.ipa_tokenizer(clean_phones(ipa))
        if tokens:
            token_seqs.append(tokens)

    total_comparisons = 0
    with _timer("pairwise edit distance"):
        for _ in range(200):
            for i in range(len(token_seqs)):
                for j in range(i + 1, len(token_seqs)):
                    normalised_feature_edit_distance(token_seqs[i], token_seqs[j], source_features)
                    total_comparisons += 1
    print(f"    {total_comparisons} comparisons")
    print()

    # ── Step 5: Single-language dictionary scan ───────────────────────
    print("=== Step 5: Dictionary scan (sequential, per-language) ===")
    phrases_for_scan = [(name, clean_phones(ipa)) for name, ipa in TEST_PHRASES[:3]]

    # Profile only dut (smallest) with 1 phrase to keep Scalene overhead manageable
    scan_languages = ["dut"]
    for lang in scan_languages:
        merged = {**target_features[lang], **source_features}
        ptd = target_ptds[lang]
        with _timer(f"scan {lang} ({len(ptd)} entries, {len(phrases_for_scan)} phrases)"):
            for _ in range(N_SCAN_ITERATIONS):
                for _phrase_key, source_ipa in phrases_for_scan[:1]:
                    source_tokens = source_spec.ipa_tokenizer(source_ipa)
                    if _HAS_CYTHON_EXT:
                        from phone_similarity._core import (
                            batch_dictionary_scan as _c_batch_dictionary_scan,
                        )

                        _c_batch_dictionary_scan(
                            source_tokens,
                            len(source_tokens),
                            ptd,
                            merged,
                            1,
                            0.50,
                        )
                    else:
                        reverse_dictionary_lookup(
                            source_ipa,
                            SOURCE_LANG,
                            source_spec,
                            source_features,
                            lang,
                            target_specs[lang],
                            target_features[lang],
                            {},
                            pre_tokenized=ptd,
                            top_n=1,
                            max_distance=0.50,
                        )

    mem3 = tracemalloc.get_traced_memory()
    print(
        f"  Memory after scans: current={mem3[0] / 1024 / 1024:.1f}MB, peak={mem3[1] / 1024 / 1024:.1f}MB"
    )
    print()

    # ── Step 6: Feature inversion ─────────────────────────────────────
    print("=== Step 6: Feature inversion (invert_ipa) ===")
    for lang in TARGET_LANGUAGES[:2]:
        with _timer(f"invert_ipa x{len(TEST_PHRASES)} phrases -> {lang}"):
            for _name, ipa in TEST_PHRASES:
                invert_ipa(
                    clean_phones(ipa),
                    source_spec,
                    source_features,
                    target_features[lang],
                    top_n=3,
                    max_distance=0.6,
                )
    print()

    # ── Step 7: Memory leak check ─────────────────────────────────────
    print("=== Step 7: Memory leak check (repeat scan cycle) ===")
    gc.collect()
    mem_before = tracemalloc.get_traced_memory()
    print(f"  Before repeat: current={mem_before[0] / 1024 / 1024:.1f}MB")

    for cycle in range(3):
        for lang in scan_languages:
            merged = {**target_features[lang], **source_features}
            ptd = target_ptds[lang]
            for _phrase_key, source_ipa in phrases_for_scan[:1]:
                source_tokens = source_spec.ipa_tokenizer(source_ipa)
                if _HAS_CYTHON_EXT:
                    from phone_similarity._core import (
                        batch_dictionary_scan as _c_batch_dictionary_scan,
                    )

                    _c_batch_dictionary_scan(
                        source_tokens,
                        len(source_tokens),
                        ptd,
                        merged,
                        1,
                        0.50,
                    )
        gc.collect()
        mem_now = tracemalloc.get_traced_memory()
        print(
            f"  After cycle {cycle + 1}: current={mem_now[0] / 1024 / 1024:.1f}MB, peak={mem_now[1] / 1024 / 1024:.1f}MB"
        )

    mem_after = tracemalloc.get_traced_memory()
    delta_mb = (mem_after[0] - mem_before[0]) / 1024 / 1024
    print(f"  Memory delta over 3 cycles: {delta_mb:+.3f}MB")
    if abs(delta_mb) > 5.0:
        print("  WARNING: Possible memory leak detected!")
    else:
        print("  OK: No significant memory growth")
    print()

    # ── Summary ───────────────────────────────────────────────────────
    snap_final = tracemalloc.take_snapshot()
    mem_final = tracemalloc.get_traced_memory()
    print("=== Final Memory Summary ===")
    print(f"  Current: {mem_final[0] / 1024 / 1024:.1f}MB")
    print(f"  Peak:    {mem_final[1] / 1024 / 1024:.1f}MB")

    # Top memory consumers
    print("\n=== Top 15 Memory Allocations (by size) ===")
    top_stats = snap_final.statistics("lineno")
    for stat in top_stats[:15]:
        print(f"  {stat}")

    tracemalloc.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
