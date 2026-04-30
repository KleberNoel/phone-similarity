"""Phoneme encoding: bitarray specifications, feature tables, and phone cleaning."""

from phone_similarity.base_bit_array_specification import BaseBitArraySpecification
from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.clean_phones import (
    PRESERVE_ALL,
    PRESERVE_LENGTH,
    PRESERVE_STRESS,
    STRIP_ALL,
    CleanConfig,
    clean_phones,
    extract_stress_marks,
)
from phone_similarity.intersecting_bit_array_specification import (
    IntersectingBitArraySpecification,
)
from phone_similarity.universal_features import (
    UniversalFeatureEncoder,
    encode_phoneme,
    merge_inventories,
    universal_phoneme_distance,
)

__all__ = [
    "BaseBitArraySpecification",
    "BitArraySpecification",
    "CleanConfig",
    "IntersectingBitArraySpecification",
    "PRESERVE_ALL",
    "PRESERVE_LENGTH",
    "PRESERVE_STRESS",
    "STRIP_ALL",
    "UniversalFeatureEncoder",
    "clean_phones",
    "encode_phoneme",
    "extract_stress_marks",
    "merge_inventories",
    "universal_phoneme_distance",
]
