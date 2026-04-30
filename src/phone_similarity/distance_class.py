"""High-level Distance class for phonological comparison."""

from collections.abc import Sequence
from typing import Union

from phone_similarity.base_bit_array_specification import BaseBitArraySpecification
from phone_similarity.primitives import (
    batch_pairwise_hamming,
    feature_edit_distance,
    hamming_similarity,
    normalised_feature_edit_distance,
)


class Distance:
    """Compute phonological distances between words.

    Wraps a :class:`BaseBitArraySpecification` and phoneme feature dictionaries
    to provide Hamming similarity, feature-weighted edit distance, and pairwise matrices.
    """

    def __init__(
        self,
        spec: BaseBitArraySpecification,
        phoneme_features: dict[str, dict[str, Union[bool, str]]] | None = None,
    ):
        self._spec = spec
        self._phoneme_features: dict[str, dict[str, Union[bool, str]]] = (
            phoneme_features if phoneme_features is not None else spec._phoneme_features
        )

    def hamming(self, ipa_a: str, ipa_b: str, max_syllables: int = 6) -> float:
        """Hamming similarity between two IPA strings' bitarray encodings."""
        arr_a = self._spec.ipa_to_bitarray(ipa_a, max_syllables)
        arr_b = self._spec.ipa_to_bitarray(ipa_b, max_syllables)
        return hamming_similarity(arr_a, arr_b)

    def edit_distance(self, ipa_a: str, ipa_b: str) -> float:
        """Feature-weighted edit distance between two IPA strings."""
        tokens_a = self._spec.ipa_tokenizer(ipa_a)
        tokens_b = self._spec.ipa_tokenizer(ipa_b)
        return feature_edit_distance(tokens_a, tokens_b, self._phoneme_features)

    def normalised_edit_distance(self, ipa_a: str, ipa_b: str) -> float:
        """Normalised feature-weighted edit distance in ``[0.0, 1.0]``."""
        tokens_a = self._spec.ipa_tokenizer(ipa_a)
        tokens_b = self._spec.ipa_tokenizer(ipa_b)
        return normalised_feature_edit_distance(tokens_a, tokens_b, self._phoneme_features)

    def pairwise_hamming(
        self, ipa_strings: Sequence[str], max_syllables: int = 6
    ) -> list[list[float]]:
        """Pairwise Hamming similarity matrix for a list of IPA strings."""
        arrays = [self._spec.ipa_to_bitarray(s, max_syllables) for s in ipa_strings]
        return batch_pairwise_hamming(arrays)

    def pairwise_edit_distance(self, ipa_strings: Sequence[str]) -> list[list[float]]:
        """Pairwise normalised feature edit distance matrix."""
        token_seqs = [self._spec.ipa_tokenizer(s) for s in ipa_strings]
        n = len(token_seqs)
        result = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = normalised_feature_edit_distance(
                    token_seqs[i], token_seqs[j], self._phoneme_features
                )
                result[i][j] = d
                result[j][i] = d
        return result
