from __future__ import annotations

import abc
import logging
from functools import lru_cache
from typing import Union

from bitarray import bitarray

from phone_similarity._dispatch import HAS_CYTHON_TOKENIZER, cy_ipa_tokenizer


class BaseBitArraySpecification(abc.ABC):
    """Abstract base mapping phonological units to bitarray representations."""

    @abc.abstractmethod
    def ipa_to_bitarray(self, ipa: str, max_syllables: int) -> "bitarray":
        """Convert an IPA string into a padded fixed-width bitarray."""

    @abc.abstractmethod
    def generate(self, text: str) -> "bitarray":
        """Convert a text string into its bitarray representation."""

    def __init__(
        self,
        vowels: set[str],
        consonants: set[str],
        features_per_phoneme: dict[str, dict[str, bool]],
        max_syllables_per_text: int = 6,
    ) -> None:
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

    @property
    def vowels(self) -> set[str]:
        """Vowel set for this specification."""
        return self._vowels

    @property
    def consonants(self) -> set[str]:
        """Consonant set for this specification."""
        return self._consonants

    @property
    def phoneme_features(self) -> dict[str, dict[str, bool]]:
        """Full phoneme-feature mapping."""
        return self._phoneme_features

    @staticmethod
    def sort_features(
        features: dict[str, set[str]],
    ) -> dict[str, tuple[str, ...]]:
        """Return *features* with each value set replaced by a sorted tuple."""
        return {feat: tuple(sorted(feat_set)) for feat, feat_set in features.items()}

    @lru_cache(maxsize=128)
    def get_phoneme_features(self, phoneme: str) -> tuple[tuple[str, bool], ...]:
        """Return ``(feature, value)`` pairs for *phoneme*.

        Raises
        ------
        ValueError
            If *phoneme* is not in the feature set.
        """
        if phoneme in self._phoneme_features:
            return tuple(self._phoneme_features[phoneme].items())
        raise ValueError(f"Unknown Phoneme input '{phoneme}'")

    @lru_cache(128)
    def features_to_bitarray(
        self,
        feature_dict: Union[tuple[tuple[str, bool], ...], dict[str, bool]],
        columns: tuple[str, ...],
    ) -> bitarray:
        """Convert a feature dict (or tuple of pairs) to a bitarray."""
        fd = feature_dict if isinstance(feature_dict, dict) else dict(feature_dict)
        bits: list[int] = []

        for _, col in enumerate(columns, start=0):
            if "=" in col:
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
    def search_phonemes(self, ipa_str: str) -> str | None:
        """Return the longest phoneme matching the start of *ipa_str*, or ``None``."""
        for idx in range(len(ipa_str), 0, -1):
            for phone in self._phones_sorted:
                if ipa_str[:idx] == phone:
                    return phone
        return None

    def ipa_tokenizer(self, ipa_str: str) -> list[str]:
        """Tokenize *ipa_str* into recognized phoneme tokens.

        Uses the Cython hash-set tokenizer when available; falls back to a
        Python longest-match scan otherwise.
        """
        if HAS_CYTHON_TOKENIZER:
            logging.debug("Using Cython based tokenizer")
            return cy_ipa_tokenizer(ipa_str, self._phone_set, self._max_phoneme_size)

        tokens: list[str] = []
        start = 0
        while start < len(ipa_str):
            end_idx = min(start + self._max_phoneme_size, len(ipa_str))
            phoneme = self.search_phonemes(ipa_str[start:end_idx])
            if phoneme is None:
                logging.warning(
                    "IPA string contains phonemes outside usual range %s "
                    "(max phoneme length = %s) "
                    "Searched: %s",
                    ipa_str,
                    self._max_phoneme_size,
                    ipa_str[start:end_idx],
                )
                start += 1
            else:
                start += len(phoneme)
                tokens.append(phoneme)

        return tokens
