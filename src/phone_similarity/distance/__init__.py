"""Phonological distance metrics: primitives, edit distance, coarticulation, inversion."""

from phone_similarity.coarticulation import (
    CoarticulationRule,
    DefaultCoarticulationModel,
    FricativeConfig,
    coarticulated_feature_edit_distance,
    coarticulated_phoneme_distance,
    normalised_coarticulated_feature_edit_distance,
)
from phone_similarity.cross_language import compare_cross_language
from phone_similarity.distance_class import Distance
from phone_similarity.inversion import invert_features, invert_ipa
from phone_similarity.primitives import (
    batch_pairwise_hamming,
    feature_edit_distance,
    hamming_distance,
    hamming_similarity,
    normalised_feature_edit_distance,
    phoneme_feature_distance,
)

__all__ = [
    "CoarticulationRule",
    "DefaultCoarticulationModel",
    "Distance",
    "FricativeConfig",
    "batch_pairwise_hamming",
    "coarticulated_feature_edit_distance",
    "coarticulated_phoneme_distance",
    "compare_cross_language",
    "feature_edit_distance",
    "hamming_distance",
    "hamming_similarity",
    "invert_features",
    "invert_ipa",
    "normalised_coarticulated_feature_edit_distance",
    "normalised_feature_edit_distance",
    "phoneme_feature_distance",
]
