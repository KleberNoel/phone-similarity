# phonetic bitarray

Aim: Multilingual representation of phonemes / phones in language using lightweight, fast, bitarrays.

## Commands

```bash
pip install -e ".[dev,g2p]"          # dev install (includes Cython, pytest, ruff, G2P)
python setup.py build_ext --inplace  # build Cython extension (required before tests)
ruff check                           # lint
ruff format --check                  # format check
pytest tests/ -x -q                  # run tests (build Cython first!)
pytest tests/ -m "not slow"          # skip G2P/heavy tests
```

CI order: lint → test (with Cython built) → build wheels.

## Architecture

- **Source layout:** `src/phone_similarity/`
- **Cython core:** `_core.pyx` provides 7 capability levels (OpenMP, prange, tokenizer, etc.). Pure-Python fallbacks exist for all paths. `_dispatch.py` detects available levels at import time.
- **Lazy imports:** `__init__.py` uses PEP 562 `__getattr__`; heavy deps (panphon, numpy, transformers) load only on access. Language data loaded lazily from `language/_data.json`.
- **G2P is optional:** CharsiuG2P (ONNX ByT5-tiny) only needed for grapheme-to-phoneme; dictionary-only usage has no ML deps.

## Conventions

- IPA Unicode characters are used extensively in source. Ruff rules RUF001/RUF002/RUF003 are suppressed per-file — do not "fix" these.
- Language codes: Charsiu uses hyphens (`eng-us`), internal API uses underscores (`eng_us`). Convert with `replace("-", "_")`.
- Disk cache lives at `~/.cache/phone_similarity/pretok_{lang}_{fingerprint}.v2.pkl`.
- Test marker: `@pytest.mark.slow` for tests loading G2P dictionaries or processing thousands of words.

## Build quirks

- macOS OpenMP: requires `brew install libomp`; setup.py auto-detects via `CFLAGS`/`LDFLAGS` or homebrew paths.
- Verify Cython built correctly: `python -c "from phone_similarity._dispatch import HAS_CYTHON; assert HAS_CYTHON"`
- CI tests assert `HAS_CYTHON` — never skip the extension build before running tests.
