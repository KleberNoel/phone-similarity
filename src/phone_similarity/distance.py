"""
Phonological distance and similarity metrics.

.. deprecated::
    This module is a **backward-compatibility shim**.  All functionality has
    been split into focused sub-modules:

    * :mod:`phone_similarity.primitives` -- low-level distance functions
    * :mod:`phone_similarity.distance_class` -- ``Distance`` high-level API
    * :mod:`phone_similarity.pretokenize` -- dictionary pre-tokenization & cache
    * :mod:`phone_similarity.dictionary_scan` -- single & parallel scanning
    * :mod:`phone_similarity.inversion` -- feature-vector inversion
    * :mod:`phone_similarity.cross_language` -- cross-language comparison

    Please update imports to use the new module paths.  All names exported
    from this module will continue to work indefinitely for backward
    compatibility.
"""

# Re-export everything so existing ``from phone_similarity.distance import X``
# continues to work.

from phone_similarity.cross_language import compare_cross_language
from phone_similarity.dictionary_scan import (
    _scan_one_language,
    parallel_dictionary_scan,
    reverse_dictionary_lookup,
)
from phone_similarity.distance_class import Distance
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
    _HAS_CYTHON,
    _HAS_CYTHON_EXT,
    batch_pairwise_hamming,
    feature_edit_distance,
    hamming_distance,
    hamming_similarity,
    normalised_feature_edit_distance,
    phoneme_feature_distance,
)

__all__ = [
    # primitives
    "_HAS_CYTHON",
    "_HAS_CYTHON_EXT",
    # Distance class
    "Distance",
    # pretokenize
    "PreTokenizedDictionary",
    # dictionary scan
    "_scan_one_language",
    "batch_pairwise_hamming",
    "cached_pretokenize_dictionary",
    # cross-language
    "compare_cross_language",
    "feature_edit_distance",
    "hamming_distance",
    "hamming_similarity",
    # inversion
    "invert_features",
    "invert_ipa",
    "normalised_feature_edit_distance",
    "parallel_dictionary_scan",
    "phoneme_feature_distance",
    "pretokenize_dictionary",
    "reverse_dictionary_lookup",
]
