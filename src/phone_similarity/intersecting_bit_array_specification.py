"""
Merged (union) bitarray specification for cross-language comparisons.

Combines the vowel, consonant, and feature inventories of multiple
:class:`BitArraySpecification` instances so that phonemes from different
languages can be encoded into a shared bitarray space.
"""

from phone_similarity.base_bit_array_specification import BaseBitArraySpecification


class IntersectingBitArraySpecification(BaseBitArraySpecification):
    """
    A specification that combines multiple BitArraySpecifications.
    """

    def __init__(
        self, specifications: list[BaseBitArraySpecification], max_syllables_per_text: int = 6
    ):
        """Combine multiple specifications into a shared phoneme space.

        Parameters
        ----------
        specifications : list of BaseBitArraySpecification
            Specifications whose inventories will be merged (union).
        max_syllables_per_text : int
            Maximum syllables per text chunk (default 6).
        """
        vowels = set()
        consonants = set()
        features_per_phoneme = {}

        for spec in specifications:
            vowels.update(spec._vowels)
            consonants.update(spec._consonants)
            features_per_phoneme.update(spec._phoneme_features)

        super().__init__(
            vowels=vowels,
            consonants=consonants,
            features_per_phoneme=features_per_phoneme,
            max_syllables_per_text=max_syllables_per_text,
        )
