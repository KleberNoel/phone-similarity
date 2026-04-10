"""
phone_similarity -- Phonological distance and similarity metrics.

Compute feature-weighted edit distances between IPA transcriptions, scan
foreign-language dictionaries for phonological near-matches, and discover
multi-word interlingual puns via beam search segmentation.  Hot paths are
Cython-accelerated when the ``_core`` extension is compiled.

Quick start::

    from phone_similarity import Distance
    from phone_similarity.language import LANGUAGES
    from phone_similarity.bit_array_specification import BitArraySpecification

    lang = LANGUAGES["eng_us"]
    spec = BitArraySpecification(
        vowels=lang.VOWELS_SET,
        consonants=set(lang.PHONEME_FEATURES) - lang.VOWELS_SET,
        features=lang.FEATURES,
        features_per_phoneme=lang.PHONEME_FEATURES,
    )
    dist = Distance(spec)
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
* :mod:`phone_similarity.embedding` -- ``PhoneticEmbedder``,
  ``BruteForceIndex``, ``KDTreeIndex``, ``ann_dictionary_scan``
* :mod:`phone_similarity.universal_features` -- ``UniversalFeatureEncoder``,
  ``encode_phoneme``, ``universal_phoneme_distance``, ``merge_inventories``
* :mod:`phone_similarity.coarticulation` -- ``DefaultCoarticulationModel``,
  ``CoarticulationRule``, ``FricativeConfig``,
  ``coarticulated_feature_edit_distance``,
  ``normalised_coarticulated_feature_edit_distance``

Legacy imports from :mod:`phone_similarity.distance` continue to work.
"""

from phone_similarity.beam_search import (
    BeamResult,
    beam_search_phrases,
    beam_search_segmentation,
)
from phone_similarity.coarticulation import (
    CoarticulationRule,
    DefaultCoarticulationModel,
    FricativeConfig,
    coarticulated_feature_edit_distance,
    coarticulated_phoneme_distance,
    normalised_coarticulated_feature_edit_distance,
)
from phone_similarity.clean_phones import (
    CleanConfig,
    PRESERVE_ALL,
    PRESERVE_LENGTH,
    PRESERVE_STRESS,
    STRIP_ALL,
    clean_phones,
    extract_stress_marks,
)
from phone_similarity.cross_language import compare_cross_language
from phone_similarity.dictionary_scan import (
    parallel_dictionary_scan,
    reverse_dictionary_lookup,
)
from phone_similarity.distance_class import Distance
from phone_similarity.embedding import (
    BruteForceIndex,
    KDTreeIndex,
    PhoneticEmbedder,
    ann_dictionary_scan,
)
from phone_similarity.inversion import (
    invert_features,
    invert_ipa,
)
from phone_similarity.pretokenize import (
    PreTokenizedDictionary,
    cached_pretokenize_dictionary,
    pretokenize_dictionary,
)
from phone_similarity.primitives import (
    batch_pairwise_hamming,
    feature_edit_distance,
    hamming_distance,
    hamming_similarity,
    normalised_feature_edit_distance,
    phoneme_feature_distance,
)
from phone_similarity.universal_features import (
    UniversalFeatureEncoder,
    encode_phoneme,
    merge_inventories,
    universal_phoneme_distance,
)
from phone_similarity.syllable import (
    MaxOnsetSegmenter,
    Syllable,
    SonorityScale,
    batch_syllabify,
    stress_pattern,
    stressed_syllable,
    syllable_count,
    syllabify,
)

__all__ = [
    # beam search
    "BeamResult",
    # embedding / ANN
    "BruteForceIndex",
    # clean phones
    "CleanConfig",
    # co-articulation
    "CoarticulationRule",
    # Distance class
    "Distance",
    "DefaultCoarticulationModel",
    "FricativeConfig",
    "KDTreeIndex",
    # syllable
    "MaxOnsetSegmenter",
    "PRESERVE_ALL",
    "PRESERVE_LENGTH",
    "PRESERVE_STRESS",
    "PhoneticEmbedder",
    # pretokenize
    "PreTokenizedDictionary",
    "STRIP_ALL",
    "SonorityScale",
    "Syllable",
    # universal features
    "UniversalFeatureEncoder",
    "ann_dictionary_scan",
    # primitives
    "batch_pairwise_hamming",
    "batch_syllabify",
    "beam_search_phrases",
    "beam_search_segmentation",
    "cached_pretokenize_dictionary",
    "clean_phones",
    "coarticulated_feature_edit_distance",
    "coarticulated_phoneme_distance",
    # cross-language
    "compare_cross_language",
    "encode_phoneme",
    "extract_stress_marks",
    "feature_edit_distance",
    "hamming_distance",
    "hamming_similarity",
    # inversion
    "invert_features",
    "invert_ipa",
    "merge_inventories",
    "normalised_coarticulated_feature_edit_distance",
    "normalised_feature_edit_distance",
    # dictionary scan
    "parallel_dictionary_scan",
    "phoneme_feature_distance",
    "pretokenize_dictionary",
    "reverse_dictionary_lookup",
    "stress_pattern",
    "stressed_syllable",
    "syllable_count",
    "syllabify",
    "universal_phoneme_distance",
]
