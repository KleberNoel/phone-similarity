"""
phone_similarity -- Phonological distance and similarity metrics.

Compute feature-weighted edit distances between IPA transcriptions, scan
foreign-language dictionaries for phonological near-matches, and discover
multi-word interlingual puns via beam search segmentation.  Hot paths are
Cython-accelerated when the ``_core`` extension is compiled.

Quick start::

    from phone_similarity.language import LANGUAGES

    # Builder pattern — one call instead of 5 lines:
    spec = LANGUAGES.build_spec("eng_us")
    dist = LANGUAGES.build_distance("eng_us")
    dist.normalised_edit_distance("kæt", "kæb")  # ~0.04

Public API
----------
All public symbols are importable directly from this package::

    from phone_similarity import Distance, beam_search_segmentation

For finer-grained imports, use the sub-modules:

* :mod:`phone_similarity.clean_phones` -- ``clean_phones``,
  ``CleanConfig``, ``extract_stress_marks``, pre-built configs
  (``STRIP_ALL``, ``PRESERVE_STRESS``, ``PRESERVE_LENGTH``,
  ``PRESERVE_ALL``)
* :mod:`phone_similarity.primitives` -- Hamming distance/similarity,
  phoneme feature distance, feature edit distance, batch pairwise Hamming
* :mod:`phone_similarity.distance_class` -- ``Distance`` high-level class
* :mod:`phone_similarity.pretokenize` -- ``PreTokenizedDictionary`` and
  disk caching via ``cached_pretokenize_dictionary``
* :mod:`phone_similarity.dictionary_scan` -- ``reverse_dictionary_lookup``
  and ``parallel_dictionary_scan``
* :mod:`phone_similarity.inversion` -- ``invert_features``, ``invert_ipa``
* :mod:`phone_similarity.cross_language` -- ``compare_cross_language``
* :mod:`phone_similarity.beam_search` -- ``beam_search_segmentation``,
  ``beam_search_phrases``, ``BeamResult``
* :mod:`phone_similarity.universal_features` -- ``UniversalFeatureEncoder``,
  ``encode_phoneme``, ``universal_phoneme_distance``, ``merge_inventories``
* :mod:`phone_similarity.coarticulation` -- ``DefaultCoarticulationModel``,
  ``CoarticulationRule``, ``FricativeConfig``,
  ``coarticulated_feature_edit_distance``,
  ``normalised_coarticulated_feature_edit_distance``,
  ``coarticulated_phoneme_distance``
* :mod:`phone_similarity.syllable` -- ``syllabify``, ``Syllable``,
  ``SonorityScale``, ``MaxOnsetSegmenter``, ``batch_syllabify``,
  ``stressed_syllable``, ``stress_pattern``, ``syllable_count``

"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__: str = _pkg_version("phone-similarity")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

# Lazy public API (PEP 562) — submodules are imported on first attribute
# access so that ``import phone_similarity`` is cheap and does not pull in
# heavy dependencies (panphon, pandas, …) until they are actually needed.

# Map public name → (module, attribute)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BeamResult": ("phone_similarity.beam_search", "BeamResult"),
    "beam_search_phrases": ("phone_similarity.beam_search", "beam_search_phrases"),
    "beam_search_segmentation": ("phone_similarity.beam_search", "beam_search_segmentation"),
    "PRESERVE_ALL": ("phone_similarity.clean_phones", "PRESERVE_ALL"),
    "PRESERVE_LENGTH": ("phone_similarity.clean_phones", "PRESERVE_LENGTH"),
    "PRESERVE_STRESS": ("phone_similarity.clean_phones", "PRESERVE_STRESS"),
    "STRIP_ALL": ("phone_similarity.clean_phones", "STRIP_ALL"),
    "CleanConfig": ("phone_similarity.clean_phones", "CleanConfig"),
    "clean_phones": ("phone_similarity.clean_phones", "clean_phones"),
    "extract_stress_marks": ("phone_similarity.clean_phones", "extract_stress_marks"),
    "CoarticulationRule": ("phone_similarity.coarticulation", "CoarticulationRule"),
    "DefaultCoarticulationModel": (
        "phone_similarity.coarticulation",
        "DefaultCoarticulationModel",
    ),
    "FricativeConfig": ("phone_similarity.coarticulation", "FricativeConfig"),
    "coarticulated_feature_edit_distance": (
        "phone_similarity.coarticulation",
        "coarticulated_feature_edit_distance",
    ),
    "coarticulated_phoneme_distance": (
        "phone_similarity.coarticulation",
        "coarticulated_phoneme_distance",
    ),
    "normalised_coarticulated_feature_edit_distance": (
        "phone_similarity.coarticulation",
        "normalised_coarticulated_feature_edit_distance",
    ),
    "compare_cross_language": ("phone_similarity.cross_language", "compare_cross_language"),
    "parallel_dictionary_scan": ("phone_similarity.dictionary_scan", "parallel_dictionary_scan"),
    "reverse_dictionary_lookup": (
        "phone_similarity.dictionary_scan",
        "reverse_dictionary_lookup",
    ),
    "Distance": ("phone_similarity.distance_class", "Distance"),
    "invert_features": ("phone_similarity.inversion", "invert_features"),
    "invert_ipa": ("phone_similarity.inversion", "invert_ipa"),
    "PreTokenizedDictionary": ("phone_similarity.pretokenize", "PreTokenizedDictionary"),
    "cached_pretokenize_dictionary": (
        "phone_similarity.pretokenize",
        "cached_pretokenize_dictionary",
    ),
    "pretokenize_dictionary": ("phone_similarity.pretokenize", "pretokenize_dictionary"),
    "batch_pairwise_hamming": ("phone_similarity.primitives", "batch_pairwise_hamming"),
    "feature_edit_distance": ("phone_similarity.primitives", "feature_edit_distance"),
    "hamming_distance": ("phone_similarity.primitives", "hamming_distance"),
    "hamming_similarity": ("phone_similarity.primitives", "hamming_similarity"),
    "normalised_feature_edit_distance": (
        "phone_similarity.primitives",
        "normalised_feature_edit_distance",
    ),
    "phoneme_feature_distance": ("phone_similarity.primitives", "phoneme_feature_distance"),
    "MaxOnsetSegmenter": ("phone_similarity.syllable", "MaxOnsetSegmenter"),
    "SonorityScale": ("phone_similarity.syllable", "SonorityScale"),
    "Syllable": ("phone_similarity.syllable", "Syllable"),
    "batch_syllabify": ("phone_similarity.syllable", "batch_syllabify"),
    "stress_pattern": ("phone_similarity.syllable", "stress_pattern"),
    "stressed_syllable": ("phone_similarity.syllable", "stressed_syllable"),
    "syllabify": ("phone_similarity.syllable", "syllabify"),
    "syllable_count": ("phone_similarity.syllable", "syllable_count"),
    "UniversalFeatureEncoder": (
        "phone_similarity.universal_features",
        "UniversalFeatureEncoder",
    ),
    "encode_phoneme": ("phone_similarity.universal_features", "encode_phoneme"),
    "merge_inventories": ("phone_similarity.universal_features", "merge_inventories"),
    "universal_phoneme_distance": (
        "phone_similarity.universal_features",
        "universal_phoneme_distance",
    ),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'phone_similarity' has no attribute {name!r}")


__all__ = [
    "PRESERVE_ALL",
    "PRESERVE_LENGTH",
    "PRESERVE_STRESS",
    "STRIP_ALL",
    "BeamResult",
    "CleanConfig",
    "CoarticulationRule",
    "DefaultCoarticulationModel",
    "Distance",
    "FricativeConfig",
    "MaxOnsetSegmenter",
    "PreTokenizedDictionary",
    "SonorityScale",
    "Syllable",
    "UniversalFeatureEncoder",
    "__version__",
    "batch_pairwise_hamming",
    "batch_syllabify",
    "beam_search_phrases",
    "beam_search_segmentation",
    "cached_pretokenize_dictionary",
    "clean_phones",
    "coarticulated_feature_edit_distance",
    "coarticulated_phoneme_distance",
    "compare_cross_language",
    "encode_phoneme",
    "extract_stress_marks",
    "feature_edit_distance",
    "hamming_distance",
    "hamming_similarity",
    "invert_features",
    "invert_ipa",
    "merge_inventories",
    "normalised_coarticulated_feature_edit_distance",
    "normalised_feature_edit_distance",
    "parallel_dictionary_scan",
    "phoneme_feature_distance",
    "pretokenize_dictionary",
    "reverse_dictionary_lookup",
    "stress_pattern",
    "stressed_syllable",
    "syllabify",
    "syllable_count",
    "universal_phoneme_distance",
]
