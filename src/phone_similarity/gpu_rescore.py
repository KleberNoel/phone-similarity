"""Optional GPU/accelerated rescoring backends for beam search."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from phone_similarity._dispatch import (
    HAS_CPP_BEAM_RESCORE,
    cpp_beam_rescore_paths,
    cy_beam_rescore_paths,
)

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    njit = None
    HAS_NUMBA = False


RescoreBackend = Literal["auto", "cpp", "cython", "cupy", "numba"]

# ---------------------------------------------------------------------------
# Strategy registry (OCP): add new backends by inserting into this dict only.
# ---------------------------------------------------------------------------


def _make_exact(exact_backend: Literal["cpp", "cython"]):
    """Return a rescore strategy that delegates directly to an exact kernel."""

    def _strategy(
        *,
        source_idx_arr,
        source_len,
        packed_paths,
        pre_tokenized,
        all_tgt_idx_arr,
        dist_flat_arr,
        matrix_dim,
        max_distance,
    ):
        return _rescore_exact(
            exact_backend,
            source_idx_arr=source_idx_arr,
            source_len=source_len,
            packed_paths=packed_paths,
            pre_tokenized=pre_tokenized,
            all_tgt_idx_arr=all_tgt_idx_arr,
            dist_flat_arr=dist_flat_arr,
            matrix_dim=matrix_dim,
            max_distance=max_distance,
        )

    return _strategy


def _make_prefilter(prefilter_fn):
    """Return a rescore strategy that prefilters then delegates to best exact kernel."""

    def _strategy(
        *,
        source_idx_arr,
        source_len,
        packed_paths,
        pre_tokenized,
        all_tgt_idx_arr,
        dist_flat_arr,
        matrix_dim,
        max_distance,
    ):
        filtered = prefilter_fn(
            source_len, packed_paths, np.asarray(pre_tokenized.offsets), max_distance
        )
        exact = "cpp" if HAS_CPP_BEAM_RESCORE else "cython"
        return _rescore_exact(
            exact,
            source_idx_arr=source_idx_arr,
            source_len=source_len,
            packed_paths=filtered,
            pre_tokenized=pre_tokenized,
            all_tgt_idx_arr=all_tgt_idx_arr,
            dist_flat_arr=dist_flat_arr,
            matrix_dim=matrix_dim,
            max_distance=max_distance,
        )

    return _strategy


@dataclass(frozen=True)
class RescoreBenchmarkResult:
    backend: str
    runs: int
    mean_sec: float
    stdev_sec: float
    min_sec: float
    max_sec: float
    output_rows: int


def available_gpu_rescore_backends() -> dict[str, bool]:
    """Return backend availability flags."""
    return {
        "cpp": HAS_CPP_BEAM_RESCORE,
        "cython": True,
        "cupy": HAS_CUPY,
        "numba": HAS_NUMBA,
    }


if HAS_NUMBA:

    @njit(cache=True)
    def _numba_totals(
        path_ptr: np.ndarray,
        entry_ids_flat: np.ndarray,
        entry_lengths: np.ndarray,
    ) -> np.ndarray:
        n = path_ptr.shape[0] - 1
        out = np.zeros(n, dtype=np.int64)
        for i in range(n):
            s = 0
            start = path_ptr[i]
            end = path_ptr[i + 1]
            for j in range(start, end):
                eid = entry_ids_flat[j]
                if 0 <= eid < entry_lengths.shape[0]:
                    s += entry_lengths[eid]
            out[i] = s
        return out


def _pack_path_entries(
    packed_paths: list[tuple[float, tuple[int, ...], float]],
) -> tuple[np.ndarray, np.ndarray]:
    path_ptr = np.zeros(len(packed_paths) + 1, dtype=np.int64)
    flat: list[int] = []
    cursor = 0
    for i, (_score, entry_ids, _raw) in enumerate(packed_paths):
        path_ptr[i] = cursor
        flat.extend(int(eid) for eid in entry_ids)
        cursor += len(entry_ids)
    path_ptr[-1] = cursor
    flat_arr = np.asarray(flat, dtype=np.int64)
    return path_ptr, flat_arr


def _prefilter_paths_numpy(
    source_len: int,
    packed_paths: list[tuple[float, tuple[int, ...], float]],
    offsets_arr: np.ndarray,
    max_distance: float,
) -> list[tuple[float, tuple[int, ...], float]]:
    if not packed_paths:
        return []

    entry_lengths = np.diff(offsets_arr.astype(np.int64, copy=False))
    totals = np.empty(len(packed_paths), dtype=np.int64)
    for i, (_score, entry_ids, _raw) in enumerate(packed_paths):
        s = 0
        for eid in entry_ids:
            if 0 <= eid < len(entry_lengths):
                s += int(entry_lengths[eid])
        totals[i] = s

    denom = np.maximum(totals, source_len)
    valid = totals > 0
    ratio = np.zeros_like(denom, dtype=np.float64)
    ratio[valid] = np.abs(source_len - totals[valid]) / denom[valid]
    mask = valid & (ratio <= max_distance)
    return [packed_paths[i] for i, ok in enumerate(mask.tolist()) if ok]


def _prefilter_paths_numba(
    source_len: int,
    packed_paths: list[tuple[float, tuple[int, ...], float]],
    offsets_arr: np.ndarray,
    max_distance: float,
) -> list[tuple[float, tuple[int, ...], float]]:
    if not HAS_NUMBA:
        raise RuntimeError("Numba backend requested but numba is not installed")
    if not packed_paths:
        return []

    path_ptr, entry_ids_flat = _pack_path_entries(packed_paths)
    entry_lengths = np.diff(offsets_arr.astype(np.int64, copy=False))
    totals = _numba_totals(path_ptr, entry_ids_flat, entry_lengths)

    denom = np.maximum(totals, source_len)
    valid = totals > 0
    ratio = np.zeros_like(denom, dtype=np.float64)
    ratio[valid] = np.abs(source_len - totals[valid]) / denom[valid]
    mask = valid & (ratio <= max_distance)
    return [packed_paths[i] for i, ok in enumerate(mask.tolist()) if ok]


def _prefilter_paths_cupy(
    source_len: int,
    packed_paths: list[tuple[float, tuple[int, ...], float]],
    offsets_arr: np.ndarray,
    max_distance: float,
) -> list[tuple[float, tuple[int, ...], float]]:
    if not HAS_CUPY:
        raise RuntimeError("CuPy backend requested but cupy is not installed")
    if not packed_paths:
        return []

    path_ptr, entry_ids_flat = _pack_path_entries(packed_paths)
    entry_lengths = np.diff(offsets_arr.astype(np.int64, copy=False))

    if entry_ids_flat.size == 0:
        return []

    entry_lengths_cp = cp.asarray(entry_lengths)
    entry_ids_cp = cp.asarray(entry_ids_flat)
    path_ptr_cp = cp.asarray(path_ptr)

    seg_lengths = entry_lengths_cp[entry_ids_cp]
    prefix = cp.concatenate((cp.asarray([0], dtype=seg_lengths.dtype), cp.cumsum(seg_lengths)))
    totals = prefix[path_ptr_cp[1:]] - prefix[path_ptr_cp[:-1]]

    denom = cp.maximum(totals, source_len)
    valid = totals > 0
    ratio = cp.zeros_like(denom, dtype=cp.float64)
    ratio[valid] = cp.abs(source_len - totals[valid]) / denom[valid]
    mask = valid & (ratio <= max_distance)

    keep = cp.asnumpy(mask).tolist()
    return [packed_paths[i] for i, ok in enumerate(keep) if ok]


def _rescore_exact(
    backend: Literal["cpp", "cython"],
    *,
    source_idx_arr: np.ndarray,
    source_len: int,
    packed_paths: list[tuple[float, tuple[int, ...], float]],
    pre_tokenized: object,
    all_tgt_idx_arr: np.ndarray,
    dist_flat_arr: np.ndarray,
    matrix_dim: int,
    max_distance: float,
) -> list[tuple[float, tuple[int, ...], float]]:
    if backend == "cpp":
        if not HAS_CPP_BEAM_RESCORE:
            raise RuntimeError("C++ rescoring backend is not available")
        return cpp_beam_rescore_paths(
            source_idx_arr=source_idx_arr,
            source_len=source_len,
            packed_paths=packed_paths,
            offsets_arr=pre_tokenized.offsets,
            all_tgt_idx_arr=all_tgt_idx_arr,
            dist_flat_arr=dist_flat_arr,
            matrix_dim=matrix_dim,
            max_distance=max_distance,
        )

    return cy_beam_rescore_paths(
        source_idx_arr,
        source_len,
        packed_paths,
        pre_tokenized,
        all_tgt_idx_arr,
        dist_flat_arr,
        matrix_dim,
        max_distance=max_distance,
    )


# Module-level registry — add new backends here without touching gpu_rescore_paths.
_BACKEND_REGISTRY: dict[str, object] = {
    "cpp": _make_exact("cpp"),
    "cython": _make_exact("cython"),
    "numba": _make_prefilter(_prefilter_paths_numba),
    "cupy": _make_prefilter(_prefilter_paths_cupy),
}


def gpu_rescore_paths(
    *,
    source_idx_arr: np.ndarray,
    source_len: int,
    packed_paths: list[tuple[float, tuple[int, ...], float]],
    pre_tokenized: object,
    all_tgt_idx_arr: np.ndarray,
    dist_flat_arr: np.ndarray,
    matrix_dim: int,
    max_distance: float = 0.50,
    backend: RescoreBackend = "auto",
) -> list[tuple[float, tuple[int, ...], float]]:
    """Rescore packed beam paths with optional GPU-prefilter backends."""
    if not packed_paths:
        return []

    if backend == "auto":
        backend = "cpp" if HAS_CPP_BEAM_RESCORE else "cython"

    strategy = _BACKEND_REGISTRY.get(backend)
    if strategy is None:
        raise ValueError(f"Unknown backend: {backend}")

    return strategy(
        source_idx_arr=source_idx_arr,
        source_len=source_len,
        packed_paths=packed_paths,
        pre_tokenized=pre_tokenized,
        all_tgt_idx_arr=all_tgt_idx_arr,
        dist_flat_arr=dist_flat_arr,
        matrix_dim=matrix_dim,
        max_distance=max_distance,
    )


def benchmark_gpu_rescore_backends(
    *,
    source_idx_arr: np.ndarray,
    source_len: int,
    packed_paths: list[tuple[float, tuple[int, ...], float]],
    pre_tokenized: object,
    all_tgt_idx_arr: np.ndarray,
    dist_flat_arr: np.ndarray,
    matrix_dim: int,
    max_distance: float = 0.50,
    runs: int = 6,
    warmup: int = 1,
    backends: list[RescoreBackend] | None = None,
) -> list[RescoreBenchmarkResult]:
    """Benchmark selected rescoring backends on the same workload."""
    if backends is None:
        backends = ["cpp", "cython", "numba", "cupy"]

    out: list[RescoreBenchmarkResult] = []
    availability = available_gpu_rescore_backends()

    for backend in backends:
        if backend in {"numba", "cupy", "cpp"} and not availability.get(backend, False):
            continue
        if backend == "auto":
            continue

        times: list[float] = []
        final_rows = 0
        for i in range(runs + warmup):
            t0 = time.perf_counter()
            rows = gpu_rescore_paths(
                source_idx_arr=source_idx_arr,
                source_len=source_len,
                packed_paths=packed_paths,
                pre_tokenized=pre_tokenized,
                all_tgt_idx_arr=all_tgt_idx_arr,
                dist_flat_arr=dist_flat_arr,
                matrix_dim=matrix_dim,
                max_distance=max_distance,
                backend=backend,
            )
            dt = time.perf_counter() - t0
            if i >= warmup:
                times.append(dt)
                final_rows = len(rows)

        if times:
            out.append(
                RescoreBenchmarkResult(
                    backend=backend,
                    runs=runs,
                    mean_sec=statistics.mean(times),
                    stdev_sec=statistics.pstdev(times),
                    min_sec=min(times),
                    max_sec=max(times),
                    output_rows=final_rows,
                )
            )

    return out
