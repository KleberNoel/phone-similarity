"""Merged (union) bitarray specification for cross-language comparisons."""

from phone_similarity.base_bit_array_specification import BaseBitArraySpecification
from phone_similarity.universal_features import UniversalFeatureEncoder


class IntersectingBitArraySpecification(BaseBitArraySpecification):
    """Merges multiple BaseBitArraySpecification instances into a shared phoneme space."""

    def __init__(
        self,
        specifications: list[BaseBitArraySpecification],
        max_syllables_per_text: int = 6,
    ):
        vowels: set[str] = set()
        consonants: set[str] = set()

        # Collect raw per-language feature dicts so we can pass them
        # through the universal encoder for consistent merging.
        raw_inventories: list[dict] = []

        for spec in specifications:
            vowels.update(spec._vowels)
            consonants.update(spec._consonants)
            raw_inventories.append(spec._phoneme_features)

        # Use UniversalFeatureEncoder to produce one consistent feature
        # dict per phoneme — shared phonemes (e.g. /e/) will always get
        # the same 24-feature Panphon encoding regardless of which
        # language defined them.
        features_per_phoneme = UniversalFeatureEncoder.merge_inventories(*raw_inventories)

        super().__init__(
            vowels=vowels,
            consonants=consonants,
            features_per_phoneme=features_per_phoneme,
            max_syllables_per_text=max_syllables_per_text,
        )

    def ipa_to_bitarray(self, ipa: str, max_syllables: int):
        """Not supported; use a BitArraySpecification for encoding."""
        raise NotImplementedError("IntersectingBitArraySpecification is for tokenization only")

    def generate(self, text: str):
        """Not supported; use a BitArraySpecification for encoding."""
        raise NotImplementedError("IntersectingBitArraySpecification is for tokenization only")
