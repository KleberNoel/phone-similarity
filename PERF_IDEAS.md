# Performance Optimisation Ideas

Quick wins already shipped (this branch):

| # | Change | File | Expected gain |
|---|--------|------|---------------|
| A | `-O3 -march=native` for `_core.pyx` and `_beam_cpp.cpp` | `setup.py` | 5–20% on DP/beam kernels via auto-vectorisation and loop unrolling |
| B | `forkserver` instead of `spawn` on Linux/macOS | `dictionary_scan.py` | Eliminates repeated Python + extension re-import per worker (~0.5–1 s per language) |
| C | Module-level dispatch import in `pretokenize_dictionary` | `pretokenize.py` | Eliminates one dict lookup per call; negligible but correct |

---

## 4. Rust + PyO3 for the DP inner loop

### What & where

`feature_edit_distance` (`primitives.py` → `_core.pyx:_cdef_feature_edit_distance`) is the innermost hot path: an O(m·n) Levenshtein DP where every cell costs a `phoneme_feature_distance` dict lookup.  The Cython version already eliminates Python call overhead, but the substitution-cost lookup still involves a Python `dict` dereference per cell.

### Why Rust could be faster

The feature distance is currently stored as a Python dict of dicts (`{phoneme: {feat_name: value}}`).  In Rust we can pre-encode each phoneme as a `u32` bitmask (24 features → 24 bits, fits in a register) and replace the dict lookup with a **XOR + popcount** (2 instructions) per cell.

```rust
// PyO3 entry point (sketch)
#[pyfunction]
fn feature_edit_distance_packed(
    src: &[u32],   // pre-encoded source phoneme bitmasks
    tgt: &[u32],   // pre-encoded target phoneme bitmasks
    n_feats: u32,
) -> f64 {
    let m = src.len();
    let n = tgt.len();
    let mut dp = vec![0.0f64; (m + 1) * (n + 1)];
    // ... standard DP with sub_cost = (src[i] ^ tgt[j]).count_ones() / n_feats
}
```

The bitmask encoding is computed **once** at `build_beam_search_resources` time and cached in `BeamSearchResources`.  No Python objects touch the hot loop.

### Effort / risk

- ~200 lines of Rust in a `pyo3` crate under `src/phone_similarity_rs/`.
- `maturin develop` for dev; `maturin build --release` for wheels.
- `_dispatch.py` gets a new level 0.5: `try: from phone_similarity._rs import ...`.
- Risk: packaging; Rust toolchain not always present in CI — make it optional like the C++ extension.

### Expected gain

Cython currently runs `feature_edit_distance` in ~2 µs for a 10×10 pair.  Rust + bitmask should reach ~0.3–0.5 µs (4–6×) for the same pair.  Beam-search throughput (which calls this millions of times) should improve proportionally.

---

## 5. SIMD phoneme distance via 24-bit feature packing

### What & where

`phoneme_feature_distance` (`primitives.py:110`, `_core.pyx:phoneme_feature_distance`) computes the fraction of feature dimensions where two phonemes differ.  The 24-feature panphon vector is currently stored as a Python dict.  The Cython version walks both dicts with C loops, but still incurs Python object allocation per call.

### The SIMD approach

Pack each phoneme's 24 features into **one `uint32`** (or two `uint16`) at encode time:

```c
// Feature encoding: each feature has 3 states {-1, 0, +1}
// Use 2 bits per feature → 48 bits total → fits in uint64
// Or: treat absent as 0 and use 1 bit per feature → 24 bits → uint32

uint32_t encode_phoneme(const char *feat_vals, int n_feats) {
    uint32_t mask = 0;
    for (int i = 0; i < n_feats; i++)
        if (feat_vals[i]) mask |= (1u << i);
    return mask;
}

float phoneme_dist(uint32_t a, uint32_t b) {
    return __builtin_popcount(a ^ b) / 24.0f;
}
```

With AVX2, 8 pairs can be computed simultaneously in one `vpxor` + `vpopcntd` (or emulated popcount).  The `_build_dist_matrix` pre-computation already stores all pair distances in a flat array — building that array with SIMD would be ~8× faster, and it's called once per `build_beam_search_resources`.

### Where to put it

- Add `encode_phonemes_packed()` in `_core.pyx` or a new `_simd.c` compiled with `extra_compile_args=["-O3", "-mavx2"]`.
- Store encoded bitmasks in `BeamSearchResources.src_packed` / `tgt_packed` (numpy `uint32` array).
- Replace `_build_dist_matrix`'s inner loop with the popcount path.

### Expected gain

`_build_dist_matrix` for a merged vocabulary of ~300 phonemes: currently ~9 ms (Python), ~1 ms (Cython).  With SIMD bitmask popcount: ~0.05 ms.  The matrix is cached so this only matters at startup — but for Wikipedia-scale scans with many language pairs it adds up.

More impactfully: if the DP cell cost is a bitmask XOR+popcount (no array lookup), the beam-expand kernel could inline the cost instead of reading from the pre-computed `dist_flat_arr`, improving cache locality.

---

## 6. Pre-forked worker pool with shared memory for `parallel_dictionary_scan`

### Current bottleneck

`parallel_dictionary_scan` uses `ProcessPoolExecutor(mp_context="forkserver")` (after QW-B).  Each `pool.submit(_scan_one_language, item)` serialises the **entire `PreTokenizedDictionary`** (token_indices: int16 array, offsets: int32 array, word/ipa lists) into a pickle message sent over a socket.  For French (278 K entries, ~3 MB arrays), this is ~30 ms of IPC per language per scan call.

### Shared memory approach (`multiprocessing.shared_memory`)

```python
# At startup, once per language:
from multiprocessing.shared_memory import SharedMemory

shm_ti = SharedMemory(create=True, size=ptd.token_indices.nbytes)
shm_to = SharedMemory(create=True, size=ptd.offsets.nbytes)
np.ndarray(ptd.token_indices.shape, dtype=np.int16, buffer=shm_ti.buf)[:] = ptd.token_indices
np.ndarray(ptd.offsets.shape,       dtype=np.int32, buffer=shm_to.buf)[:] = ptd.offsets

# Pass only the SHM names + metadata to workers (a few bytes):
worker_args = (shm_ti.name, shm_to.name, ptd.words, ptd.ipas, ptd.inventory, ...)
```

Workers reconstruct the `PreTokenizedDictionary` from the shared memory block — zero-copy, no pickle.  The numpy arrays live in shared RAM; all worker processes read them without copying.

### Pre-forked persistent pool

An even larger win: use a **pool initializer** to load all dictionaries once per worker process, then keep the pool alive across multiple `parallel_dictionary_scan` calls:

```python
_POOL: ProcessPoolExecutor | None = None
_POOL_RESOURCES: dict = {}  # lang -> PTD loaded in worker

def _worker_init(shm_specs):
    global _POOL_RESOURCES
    # reconstruct PTDs from shared memory once per worker lifetime
    _POOL_RESOURCES = {lang: _load_ptd_from_shm(spec) for lang, spec in shm_specs.items()}

def get_persistent_pool(targets, max_workers):
    global _POOL
    if _POOL is None:
        shm_specs = _register_shm(targets)
        _POOL = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("forkserver"),
            initializer=_worker_init,
            initargs=(shm_specs,),
        )
    return _POOL
```

### Expected gain

- Eliminates ~30 ms × n_languages of IPC per scan call.
- For Wikipedia mining (10 K phrases × 5 languages): saves ~150 s of IPC overhead.
- First call pays the shared-memory setup cost (~100 ms); all subsequent calls pay near zero.

### Effort / risk

- Requires careful shared-memory lifecycle management (must call `shm.unlink()` on shutdown).
- `words` and `ipas` (Python string lists) still need to be pickled — consider encoding them as a single bytes blob separated by null bytes.
- Medium effort (~150 lines).

---

## 7. numpy structured arrays for beam search paths (Python fallback)

### Current situation

The Python-fallback beam search (`beam_search.py:601–701`) stores each hypothesis as a `_Hypothesis` dataclass with tuple fields:

```python
new_hyp = _Hypothesis(
    score=new_score,
    consumed=new_consumed,
    words=(*hyp.words, word),   # O(k) tuple copy per expansion
    ipas=(*hyp.ipas, ipa),      # O(k) tuple copy per expansion
    raw_cost=new_raw_cost,
)
```

For a beam of width 10 × 4 words × N candidates per position, this creates O(10 × 4 × N) tuple objects per search, each requiring a GC-managed allocation.

### numpy structured array beam

Replace the `_Hypothesis` list with a fixed-size structured array:

```python
dtype = np.dtype([
    ("score",    np.float32),
    ("raw_cost", np.float32),
    ("consumed", np.int16),
    ("n_words",  np.int8),
    ("word_ids", np.int32, (MAX_WORDS,)),  # word IDs instead of strings
])
beam = np.empty(beam_width, dtype=dtype)
```

- No per-hypothesis Python object allocation.
- `beam.sort(order="score")` uses numpy's C-level sort (Timsort on structured array).
- Words are stored as integer IDs into `resources.words`; string lookup happens only at result extraction.

### But: this path is rarely executed

In practice the C++/Cython beam state search (`HAS_CPP_BEAM_STATE` / `HAS_CYTHON_BEAM_STATE`) handles all real workloads.  The Python fallback only runs when the extensions are absent (e.g. pure-Python installs).

### Recommendation

This is low-priority unless the pure-Python wheel needs to be fast.  Instead, focus effort on items 4–6 which improve the **compiled** path.

---

## Implementation Priority

| Priority | Item | Effort | Expected gain |
|----------|------|--------|---------------|
| 1 | **Bitmask SIMD encoding** (#5) in `_core.pyx` | S (50 lines Cython) | 4–8× `_build_dist_matrix`; enables inline DP costs |
| 2 | **Rust DP kernel** (#4) via PyO3 | M (200 lines Rust + CI) | 4–6× `feature_edit_distance`; biggest beam-search win |
| 3 | **Shared-memory PTD** (#6) | M (150 lines Python) | Eliminates IPC for multi-language scans |
| 4 | **numpy beam** (#7) | S (100 lines Python) | Low: only Python-fallback path |
