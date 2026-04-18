"""
Cython dispatch infrastructure.

Centralises the detection of Cython extension modules into a single
location, eliminating the duplicated ``try/except ImportError`` blocks
that were scattered across ``primitives.py``, ``dictionary_scan.py``,
and other modules.

Design Pattern
--------------
**Chain of Responsibility** — each capability level (core, extended,
prange) is probed once at import time.  Downstream modules query the
flags and function references from here.

Usage::

    from phone_similarity._dispatch import (
        HAS_CYTHON, HAS_CYTHON_EXT, HAS_PRANGE,
        cy_hamming_distance, cy_feature_edit_distance, ...
    )

    if HAS_CYTHON:
        result = cy_feature_edit_distance(...)
    else:
        result = _python_fallback(...)
"""

from __future__ import annotations

from typing import Any

# Sentinel for unavailable functions
_UNAVAILABLE: Any = None

# Level 1: Core Cython (hamming, edit distance, batch pairwise)
try:
    from phone_similarity._core import (
        batch_pairwise_hamming as cy_batch_pairwise_hamming,
    )
    from phone_similarity._core import (
        feature_edit_distance as cy_feature_edit_distance,
    )
    from phone_similarity._core import (
        hamming_distance as cy_hamming_distance,
    )
    from phone_similarity._core import (
        hamming_similarity as cy_hamming_similarity,
    )

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    cy_hamming_distance = _UNAVAILABLE
    cy_hamming_similarity = _UNAVAILABLE
    cy_feature_edit_distance = _UNAVAILABLE
    cy_batch_pairwise_hamming = _UNAVAILABLE

# Level 2: Extended Cython (dictionary scan, feature inversion, phoneme dist)
try:
    from phone_similarity._core import (
        batch_dictionary_scan as cy_batch_dictionary_scan,
    )
    from phone_similarity._core import (
        invert_features as cy_invert_features,
    )
    from phone_similarity._core import (
        phoneme_feature_distance as cy_phoneme_feature_distance,
    )

    HAS_CYTHON_EXT = True
except ImportError:
    HAS_CYTHON_EXT = False
    cy_batch_dictionary_scan = _UNAVAILABLE
    cy_invert_features = _UNAVAILABLE
    cy_phoneme_feature_distance = _UNAVAILABLE

# Level 3: OpenMP prange (parallel dictionary scan)
try:
    from phone_similarity._core import (
        prange_batch_dictionary_scan as cy_prange_batch_dictionary_scan,
    )

    HAS_PRANGE = True
except ImportError:
    HAS_PRANGE = False
    cy_prange_batch_dictionary_scan = _UNAVAILABLE

# Level 4: Cython IPA tokenizer
try:
    from phone_similarity._core import (
        batch_ipa_tokenize as cy_batch_ipa_tokenize,
    )
    from phone_similarity._core import (
        cython_ipa_tokenizer as cy_ipa_tokenizer,
    )

    HAS_CYTHON_TOKENIZER = True
except ImportError:
    HAS_CYTHON_TOKENIZER = False
    cy_ipa_tokenizer = _UNAVAILABLE
    cy_batch_ipa_tokenize = _UNAVAILABLE

# Level 5: Cython syllabifier
try:
    from phone_similarity._core import (
        batch_cython_syllabify as cy_batch_syllabify,
    )
    from phone_similarity._core import (
        cython_syllabify as cy_syllabify,
    )

    HAS_CYTHON_SYLLABIFIER = True
except ImportError:
    HAS_CYTHON_SYLLABIFIER = False
    cy_syllabify = _UNAVAILABLE
    cy_batch_syllabify = _UNAVAILABLE

# Level 6: Cython co-articulated edit distance
try:
    from phone_similarity._core import (
        coarticulated_feature_edit_distance_c as cy_coarticulated_feature_edit_distance,
    )

    HAS_CYTHON_COARTICULATION = True
except ImportError:
    HAS_CYTHON_COARTICULATION = False
    cy_coarticulated_feature_edit_distance = _UNAVAILABLE

# Level 7: Cython phoneme distance matrix builder
try:
    from phone_similarity._core import (
        build_phoneme_dist_matrix as cy_build_phoneme_dist_matrix,
    )

    HAS_CYTHON_DIST_MATRIX = True
except ImportError:
    HAS_CYTHON_DIST_MATRIX = False
    cy_build_phoneme_dist_matrix = _UNAVAILABLE
