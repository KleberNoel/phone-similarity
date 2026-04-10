"""
Phonological distance and similarity metrics.

.. deprecated:: 0.0.3
    This module is a **backward-compatibility shim**.  All functionality has
    been split into focused sub-modules:

    * :mod:`phone_similarity.primitives` -- low-level distance functions
    * :mod:`phone_similarity.distance_class` -- ``Distance`` high-level API
    * :mod:`phone_similarity.pretokenize` -- dictionary pre-tokenization & cache
    * :mod:`phone_similarity.dictionary_scan` -- single & parallel scanning
    * :mod:`phone_similarity.inversion` -- feature-vector inversion
    * :mod:`phone_similarity.cross_language` -- cross-language comparison

    Please update imports to use the new module paths.  This shim will be
    removed in a future release.
"""

import warnings

warnings.warn(
    "phone_similarity.distance is deprecated.  "
    "Import from phone_similarity.primitives, phone_similarity.distance_class, "
    "phone_similarity.pretokenize, phone_similarity.dictionary_scan, "
    "phone_similarity.inversion, or phone_similarity.cross_language instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything so existing ``from phone_similarity.distance import X``
# continues to work.

from phone_similarity.cross_language import compare_cross_language  # noqa: E402
from phone_similarity.dictionary_scan import (  # noqa: E402
    _scan_one_language,
    parallel_dictionary_scan,
    reverse_dictionary_lookup,
)
from phone_similarity.distance_class import Distance  # noqa: E402
from phone_similarity.inversion import (  # noqa: E402
    invert_features,
    invert_ipa,
)
from phone_similarity.pretokenize import (  # noqa: E402
    PreTokenizedDictionary,
    cached_pretokenize_dictionary,
    pretokenize_dictionary,
)
from phone_similarity.primitives import (  # noqa: E402
    _HAS_CYTHON,
    _HAS_CYTHON_EXT,
    batch_pairwise_hamming,
    feature_edit_distance,
    hamming_distance,
    hamming_similarity,
    normalised_feature_edit_distance,
    phoneme_feature_distance,
)

# Includes deprecated internal names (_HAS_CYTHON, _HAS_CYTHON_EXT,
# _scan_one_language) for backward compat only.
__all__ = [
    "_HAS_CYTHON",
    "_HAS_CYTHON_EXT",
    "Distance",
    "PreTokenizedDictionary",
    "_scan_one_language",
    "batch_pairwise_hamming",
    "cached_pretokenize_dictionary",
    "compare_cross_language",
    "feature_edit_distance",
    "hamming_distance",
    "hamming_similarity",
    "invert_features",
    "invert_ipa",
    "normalised_feature_edit_distance",
    "parallel_dictionary_scan",
    "phoneme_feature_distance",
    "pretokenize_dictionary",
    "reverse_dictionary_lookup",
]
