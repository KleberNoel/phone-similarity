"""
High-level Distance class for phonological comparison.

Wraps a :class:`BitArraySpecification` and phoneme feature dictionaries
to provide convenient methods for Hamming similarity, feature-weighted
edit distance, and pairwise matrices.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.primitives import (
    batch_pairwise_hamming,
    feature_edit_distance,
    hamming_similarity,
    normalised_feature_edit_distance,
)


class Distance:
    """Compute phonological distances between words across languages.

    This class wraps a ``BitArraySpecification`` (for bitarray-level
    Hamming metrics) and the language's ``PHONEME_FEATURES`` (for
    feature-weighted edit distance).

    Parameters
    ----------
    spec : BitArraySpecification
        The specification for encoding IPA into bitarrays.
    phoneme_features : dict, optional
        Mapping of phoneme -> articulatory features.  If ``None``, the
        features from ``spec._phoneme_features`` are used.

    Examples
    --------
    >>> from phone_similarity.language import LANGUAGES
    >>> lang = LANGUAGES["eng_us"]
    >>> spec = BitArraySpecification(
    ...     vowels=lang.VOWELS_SET,
    ...     consonants=set(lang.PHONEME_FEATURES) - lang.VOWELS_SET,
    ...     features=lang.FEATURES,
    ...     features_per_phoneme=lang.PHONEME_FEATURES,
    ... )
    >>> dist = Distance(spec)
    >>> dist.hamming("hɛloʊ", "hɛlp")  # doctest: +SKIP
    0.85
    """

    def __init__(
        self,
        spec: BitArraySpecification,
        phoneme_features: dict[str, dict[str, Union[bool, str]]] | None = None,
    ):
        self._spec = spec
        self._phoneme_features: dict[str, dict[str, Union[bool, str]]] = (
            phoneme_features if phoneme_features is not None else spec._phoneme_features
        )

    # ----- Bitarray-level metrics ------------------------------------------

    def hamming(self, ipa_a: str, ipa_b: str, max_syllables: int = 6) -> float:
        """Hamming similarity between two IPA strings' bitarray encodings.

        Parameters
        ----------
        ipa_a, ipa_b : str
            IPA-transcribed strings.
        max_syllables : int
            Fixed-width syllable count for bitarray encoding.

        Returns
        -------
        float
            Similarity in ``[0.0, 1.0]``.
        """
        arr_a = self._spec.ipa_to_bitarray(ipa_a, max_syllables)
        arr_b = self._spec.ipa_to_bitarray(ipa_b, max_syllables)
        return hamming_similarity(arr_a, arr_b)

    # ----- Phoneme-sequence-level metrics ----------------------------------

    def edit_distance(self, ipa_a: str, ipa_b: str) -> float:
        """Feature-weighted edit distance between two IPA strings.

        Tokenises both strings into phoneme sequences, then runs
        ``feature_edit_distance`` with gradient substitution cost.

        Parameters
        ----------
        ipa_a, ipa_b : str
            IPA-transcribed strings.

        Returns
        -------
        float
            Raw edit distance (not normalised).
        """
        tokens_a = self._spec.ipa_tokenizer(ipa_a)
        tokens_b = self._spec.ipa_tokenizer(ipa_b)
        return feature_edit_distance(tokens_a, tokens_b, self._phoneme_features)

    def normalised_edit_distance(self, ipa_a: str, ipa_b: str) -> float:
        """Normalised feature-weighted edit distance in ``[0.0, 1.0]``."""
        tokens_a = self._spec.ipa_tokenizer(ipa_a)
        tokens_b = self._spec.ipa_tokenizer(ipa_b)
        return normalised_feature_edit_distance(tokens_a, tokens_b, self._phoneme_features)

    # ----- Batch / corpus methods ------------------------------------------

    def pairwise_hamming(
        self, ipa_strings: Sequence[str], max_syllables: int = 6
    ) -> list[list[float]]:
        """Pairwise Hamming similarity matrix for a list of IPA strings.

        Parameters
        ----------
        ipa_strings : sequence of str
            IPA-transcribed strings.
        max_syllables : int
            Fixed-width syllable count for encoding.

        Returns
        -------
        list[list[float]]
            Symmetric ``N x N`` similarity matrix.
        """
        arrays = [self._spec.ipa_to_bitarray(s, max_syllables) for s in ipa_strings]
        return batch_pairwise_hamming(arrays)

    def pairwise_edit_distance(self, ipa_strings: Sequence[str]) -> list[list[float]]:
        """Pairwise normalised feature edit distance matrix.

        Parameters
        ----------
        ipa_strings : sequence of str
            IPA-transcribed strings.

        Returns
        -------
        list[list[float]]
            Symmetric ``N x N`` distance matrix (0 = identical).
        """
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
