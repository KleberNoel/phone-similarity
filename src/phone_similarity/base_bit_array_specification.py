"""
Abstract base class for phoneme-to-bitarray mapping specifications.

Defines the interface and shared logic for tokenising IPA strings and
converting phonological feature dictionaries into fixed-width bitarray
representations suitable for Hamming distance computation.
"""

import abc
import logging
from functools import lru_cache
from typing import Optional, Union

from bitarray import bitarray

from phone_similarity._dispatch import HAS_CYTHON_TOKENIZER, cy_ipa_tokenizer


class BaseBitArraySpecification(  # noqa: B024
    abc.ABC
):  # pylint: disable=too-many-instance-attributes
    """
    Base Abstract class that maps from character based phonological units to bitarray representation
    """

    def __init__(
        self,
        vowels: set[str],
        consonants: set[str],
        features_per_phoneme: dict[str, dict[str, bool]],
        max_syllables_per_text: int = 6,
    ):
        self._vowels = vowels
        self._consonants = consonants
        self._phoneme_features = features_per_phoneme

        self._consonant_features = dict(
            filter(lambda kv: kv[0] in consonants, features_per_phoneme.items())
        )
        self._vowel_features = dict(
            filter(lambda kv: kv[0] in vowels, features_per_phoneme.items())
        )
        self._max_syllables_per_text_chunk = max_syllables_per_text
        self._phones_sorted = tuple(
            sorted(
                features_per_phoneme.keys(),
                key=lambda x: len(x),  # pylint: disable=unnecessary-lambda
                reverse=True,
            )
        )
        self._phone_set = frozenset(features_per_phoneme.keys())
        self._max_phoneme_size = max(len(p) for p in self._phones_sorted)

    @staticmethod
    def sort_features(
        features: dict[str, set[str]],
    ) -> dict[str, tuple[str]]:
        """Sorts features for consistent ordering.

        Parameters
        ----------
        features : Dict[str, Set[str]]
            A dictionary where keys are feature names and values are sets of strings.

        Returns
        -------
        Dict[str, Tuple[str]]
            A new dictionary with the same keys but with values as sorted tuples.

        """
        _features = {}
        for feat, feat_set in features.items():
            _features.update({feat: tuple(sorted(feat_set))})
        return _features

    @lru_cache(maxsize=128)
    def get_phoneme_features(self, phoneme: str) -> tuple:
        """Retrieves the feature set for a given phoneme.

        Parameters
        ----------
        phoneme : str
            The phoneme for which to retrieve features.

        Returns
        -------
        Tuple
            A tuple of (feature_name, value) pairs.

        Raises
        ------
        ValueError
            If the phoneme is not found in the feature set.

        """
        if phoneme in self._phoneme_features:
            return tuple(self._phoneme_features[phoneme].items())
        raise ValueError(f"Unknown Phoneme input '{phoneme}'")

    @lru_cache(128)
    def features_to_bitarray(
        self, feature_dict: Union[tuple[tuple[str, bool]], dict[str, bool]], columns: tuple[str]
    ) -> "bitarray":
        """Converts a feature dictionary to a bitarray.

        Parameters
        ----------
        feature_dict : Union[Tuple[Tuple[str, bool]], Dict[str, bool]]
            A dictionary of feature names to their boolean values.
        columns : Tuple[str]
            An ordered tuple of feature names that determines the bitarray structure.

        Returns
        -------
        bitarray
            The resulting bitarray representation of the phoneme's features.

        """
        fd = feature_dict if isinstance(feature_dict, dict) else dict(feature_dict)
        bits: list[int] = []

        for _, col in enumerate(columns, start=0):
            # Find feature in column name ('voiced': binary, or values 'place', 'manner')
            if "=" in col:
                # e.g. "place=alveolar", "height=low"..
                attr, val = col.split("=")
                bit = bool(fd.get(attr) == val)
            else:
                val = fd.get(col, False)
                if isinstance(val, str):
                    bit = int(col in fd.values())
                else:
                    bit = int(val or col in fd.values())
            bits.append(bit)

        return bitarray(bits)

    @lru_cache(maxsize=256)
    def search_phonemes(self, ipa_str: str) -> Optional[str]:
        """Searches for the longest matching phoneme in the internal list.

        This method iterates through a pre-sorted list of phonemes (longest to shortest)
        and returns the first one that matches the input string.

        Parameters
        ----------
        ipa_str : str
            The IPA string segment to match against known phonemes.

        Returns
        -------
        Optional[str]
            The matching phoneme string, or None if no match is found.

        """
        for idx in range(len(ipa_str), 0, -1):
            for phone in self._phones_sorted:
                if ipa_str[:idx] == phone:
                    return phone
        return None

    def ipa_tokenizer(self, ipa_str: str) -> list[str]:
        """Tokenizes an IPA string into a list of recognized phonemes.

        This method uses a basic parsing strategy that prioritizes longer phoneme
        matches. It iterates through the input string and identifies the longest
        possible phoneme at each position.

        When the Cython extension is available, uses a hash-set based lookup
        (O(max_phoneme_length) per position) instead of the Python linear scan
        (O(inventory_size) per position).

        Parameters
        ----------
        ipa_str : str
            The IPA string to be tokenized, e.g., 'strɪŋz'.

        Returns
        -------
        List[str]
            A list of phoneme tokens, e.g., ['s', 't', 'r', 'ɪ', 'ŋ', 'z'].

        Raises
        ------
        ValueError
            If a segment of the IPA string cannot be matched to any known phoneme
            up to the maximum phoneme length.

        """
        if HAS_CYTHON_TOKENIZER:
            return cy_ipa_tokenizer(ipa_str, self._phone_set, self._max_phoneme_size)

        tokens = []
        start = 0
        while start < len(ipa_str):
            end_idx = min(start + self._max_phoneme_size, len(ipa_str))
            phoneme = self.search_phonemes(ipa_str[start:end_idx])
            if phoneme is None:
                logging.warning(
                    f"IPA string contains phonemes outside usual range {ipa_str} "
                    f"(max phoneme length = {self._max_phoneme_size}) "
                    f"Searched: {ipa_str[start:end_idx]}"
                )
                start += 1
            else:
                start += len(phoneme)
                tokens.append(phoneme)

        return tokens
