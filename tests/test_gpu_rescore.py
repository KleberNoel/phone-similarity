import numpy as np

from phone_similarity.gpu_rescore import (
    available_gpu_rescore_backends,
    benchmark_gpu_rescore_backends,
    gpu_rescore_paths,
)


class _DummyPTD:
    def __init__(self):
        self.offsets = np.array([0, 3, 6], dtype=np.int32)
        self.words = ["foo", "bar"]


def _dummy_inputs():
    source_idx_arr = np.array([1, 2, 3], dtype=np.intp)
    packed_paths = [
        (0.0, (0,), 0.0),
        (0.0, (1,), 0.0),
    ]
    all_tgt_idx_arr = np.array([1, 2, 3, 3, 2, 1], dtype=np.intp)
    dim = 8
    dist_flat_arr = np.zeros((dim * dim,), dtype=np.float64)
    return source_idx_arr, packed_paths, all_tgt_idx_arr, dist_flat_arr, dim


def test_available_gpu_rescore_backends_shape():
    avail = available_gpu_rescore_backends()
    assert set(avail) == {"cpp", "cython", "cupy", "numba"}
    assert isinstance(avail["cython"], bool)


def test_gpu_rescore_auto_matches_cython_for_toy_input():
    source_idx_arr, packed_paths, all_tgt_idx_arr, dist_flat_arr, dim = _dummy_inputs()
    ptd = _DummyPTD()

    auto_rows = gpu_rescore_paths(
        source_idx_arr=source_idx_arr,
        source_len=3,
        packed_paths=packed_paths,
        pre_tokenized=ptd,
        all_tgt_idx_arr=all_tgt_idx_arr,
        dist_flat_arr=dist_flat_arr,
        matrix_dim=dim,
        max_distance=1.0,
        backend="auto",
    )
    cy_rows = gpu_rescore_paths(
        source_idx_arr=source_idx_arr,
        source_len=3,
        packed_paths=packed_paths,
        pre_tokenized=ptd,
        all_tgt_idx_arr=all_tgt_idx_arr,
        dist_flat_arr=dist_flat_arr,
        matrix_dim=dim,
        max_distance=1.0,
        backend="cython",
    )

    assert auto_rows == cy_rows


def test_benchmark_gpu_rescore_backends_runs_cython():
    source_idx_arr, packed_paths, all_tgt_idx_arr, dist_flat_arr, dim = _dummy_inputs()
    ptd = _DummyPTD()

    rows = benchmark_gpu_rescore_backends(
        source_idx_arr=source_idx_arr,
        source_len=3,
        packed_paths=packed_paths,
        pre_tokenized=ptd,
        all_tgt_idx_arr=all_tgt_idx_arr,
        dist_flat_arr=dist_flat_arr,
        matrix_dim=dim,
        max_distance=1.0,
        runs=2,
        warmup=0,
        backends=["cython"],
    )

    assert len(rows) == 1
    assert rows[0].backend == "cython"
    assert rows[0].runs == 2
    assert rows[0].output_rows >= 1
