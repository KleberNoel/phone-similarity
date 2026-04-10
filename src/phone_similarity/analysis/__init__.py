"""
Analysis tools for phonological data.

This subpackage contains research and diagnostic tools that operate on
the core ``phone_similarity`` data structures but are not part of the
main distance-computation pipeline.

* :mod:`~phone_similarity.analysis.entropy` -- ``PhonemeEntropyAnalyzer``,
  ``SyllableEncoding``, ``EntropyMetrics``
"""

from phone_similarity.analysis.entropy import (
    EntropyMetrics,
    PhonemeEntropyAnalyzer,
    SyllableEncoding,
)

__all__ = [
    "EntropyMetrics",
    "PhonemeEntropyAnalyzer",
    "SyllableEncoding",
]
