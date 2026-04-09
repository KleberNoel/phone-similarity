"""
Merged (union) bitarray specification for cross-language comparisons.

Combines the vowel, consonant, and feature inventories of multiple
:class:`BitArraySpecification` instances so that phonemes from different
languages can be encoded into a shared bitarray space.

When multiple specifications define the same phoneme with different
feature dicts, the universal Panphon 24-feature representation is used
to ensure a single, consistent encoding for each phoneme.
"""

from __future__ import annotations

from phone_similarity.base_bit_array_specification import BaseBitArraySpecification
from phone_similarity.universal_features import UniversalFeatureEncoder


class IntersectingBitArraySpecification(BaseBitArraySpecification):
    """A specification that merges multiple :class:`BaseBitArraySpecification` instances.

    Phoneme inventories are combined (union) so that IPA strings from
    any of the source languages can be tokenised and encoded.  Shared
    phonemes receive a **universal** feature dict derived from
    :class:`~phone_similarity.universal_features.UniversalFeatureEncoder`,
    guaranteeing identical bitarray encodings regardless of which
    language originally defined the phoneme.
    """

    def __init__(
        self,
        specifications: list[BaseBitArraySpecification],
        max_syllables_per_text: int = 6,
    ):
        """Combine multiple specifications into a shared phoneme space.

        Parameters
        ----------
        specifications : list of BaseBitArraySpecification
            Specifications whose inventories will be merged (union).
        max_syllables_per_text : int
            Maximum syllables per text chunk (default 6).
        """
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
