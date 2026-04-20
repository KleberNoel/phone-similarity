import os
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator


def test_caching_performance():
    """Verify that loading a dictionary from pickle cache is faster than from TSV.

    The generator now loads lazily — the dictionary is only fetched on
    first ``.pdict`` access.  This test forces two full load cycles
    (TSV parse then pickle) and asserts the pickle path is quicker.

    We only clean up the pickle file we create, **not** the entire
    ``~/.cache/phono-sim/`` directory (other cached TSVs live there).
    """
    language = "fra"
    py_version = f"py{sys.version_info.major}.{sys.version_info.minor}"
    cache_dir = Path(os.path.expanduser("~/.cache/phono-sim"))
    cache_file = cache_dir / f"{language}_{py_version}.pkl"

    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Remove pickle cache so the first load parses from TSV
    if cache_file.exists():
        cache_file.unlink()

    try:
        # First load — parses TSV (or downloads then parses)
        gen1 = CharsiuGraphemeToPhonemeGenerator(language, use_cache=True)
        start = time.perf_counter()
        _ = gen1.pdict  # triggers _ensure_dict_loaded → TSV parse + pickle write
        first_run = time.perf_counter() - start
        assert cache_file.exists(), "Pickle cache should have been created"

        # Second load — reads from pickle cache
        gen2 = CharsiuGraphemeToPhonemeGenerator(language, use_cache=True)
        start = time.perf_counter()
        _ = gen2.pdict  # triggers _ensure_dict_loaded → pickle read
        second_run = time.perf_counter() - start

        # Pickle read should be at least somewhat faster than TSV parse
        assert second_run < first_run, (
            f"Cached load ({second_run:.4f}s) should be faster than fresh parse ({first_run:.4f}s)"
        )

    finally:
        # Only clean up the pickle file we created — leave TSVs intact
        if cache_file.exists():
            cache_file.unlink()


def test_constructor_is_lazy():
    """Constructor should be effectively instant (no I/O)."""
    start = time.perf_counter()
    gen = CharsiuGraphemeToPhonemeGenerator("eng-us", use_cache=False)
    elapsed = time.perf_counter() - start

    # Constructor should take <50ms — it does no I/O or model loading
    assert elapsed < 0.05, f"Constructor took {elapsed:.4f}s, expected <50ms"
    # Internal dict should still be None (not yet loaded)
    assert gen._pdict is None
