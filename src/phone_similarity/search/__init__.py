"""Dictionary search and beam-search segmentation."""

from phone_similarity.beam_search import (
    BeamResult,
    BeamSearchResources,
    beam_search_phrases,
    beam_search_segmentation,
    build_beam_search_resources,
)
from phone_similarity.dictionary_scan import parallel_dictionary_scan, reverse_dictionary_lookup
from phone_similarity.gpu_rescore import (
    RescoreBenchmarkResult,
    RescoreBackend,
    available_gpu_rescore_backends,
    benchmark_gpu_rescore_backends,
    gpu_rescore_paths,
)
from phone_similarity.pretokenize import (
    PreTokenizedDictionary,
    cached_pretokenize_dictionary,
    pretokenize_dictionary,
)

__all__ = [
    "BeamResult",
    "BeamSearchResources",
    "PreTokenizedDictionary",
    "RescoreBackend",
    "RescoreBenchmarkResult",
    "available_gpu_rescore_backends",
    "beam_search_phrases",
    "beam_search_segmentation",
    "benchmark_gpu_rescore_backends",
    "build_beam_search_resources",
    "cached_pretokenize_dictionary",
    "gpu_rescore_paths",
    "parallel_dictionary_scan",
    "pretokenize_dictionary",
    "reverse_dictionary_lookup",
]
