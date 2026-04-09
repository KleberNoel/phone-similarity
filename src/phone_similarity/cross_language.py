"""
Cross-language phonological comparison.

Compares the pronunciation of a word across multiple languages using
either Hamming similarity on bitarray encodings or feature-weighted
edit distance.
"""

from __future__ import annotations

from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.primitives import normalised_feature_edit_distance
from phone_similarity.universal_features import UniversalFeatureEncoder


class _DistanceLite:
    """Minimal Distance-like wrapper for cross-language Hamming comparison.

    Avoids importing the full :class:`Distance` to break circular
    dependencies (Distance lives in distance_class.py).
    """

    def __init__(self, spec: BitArraySpecification):
        from phone_similarity.primitives import _HAS_CYTHON

        self._spec = spec
        self._has_cython = _HAS_CYTHON

    def hamming(self, ipa_a: str, ipa_b: str, max_syllables: int = 6) -> float:
        arr_a = self._spec.ipa_to_bitarray(ipa_a, max_syllables)
        arr_b = self._spec.ipa_to_bitarray(ipa_b, max_syllables)
        if self._has_cython:
            from phone_similarity.primitives import _c_hamming_similarity

            return _c_hamming_similarity(arr_a, arr_b)
        from phone_similarity.primitives import hamming_similarity

        return hamming_similarity(arr_a, arr_b)


def compare_cross_language(
    word_ipa_by_lang: dict[str, str],
    specs_by_lang: dict[str, BitArraySpecification],
    features_by_lang: dict[str, dict[str, dict]],
    metric: str = "edit",
) -> dict[tuple[str, str], float]:
    """Compare a word's pronunciation across multiple languages.

    Parameters
    ----------
    word_ipa_by_lang : dict
        ``{lang_code: ipa_transcription}`` for the same word.
    specs_by_lang : dict
        ``{lang_code: BitArraySpecification}`` per language.
    features_by_lang : dict
        ``{lang_code: PHONEME_FEATURES}`` per language.
    metric : ``"edit"`` | ``"hamming"``
        Which metric to use.

    Returns
    -------
    dict
        ``{(lang_a, lang_b): distance}`` for every unordered pair.
    """
    from phone_similarity.intersecting_bit_array_specification import (
        IntersectingBitArraySpecification,
    )

    langs = sorted(word_ipa_by_lang)
    results: dict[tuple[str, str], float] = {}

    for i, la in enumerate(langs):
        for lb in langs[i + 1 :]:
            if metric == "hamming":
                merged = IntersectingBitArraySpecification([specs_by_lang[la], specs_by_lang[lb]])
                merged_bit = BitArraySpecification(
                    vowels=merged._vowels,
                    consonants=merged._consonants,
                    features_per_phoneme=merged._phoneme_features,
                    features={
                        "consonant": set(specs_by_lang[la]._features.get("consonant", ()))
                        | set(specs_by_lang[lb]._features.get("consonant", ())),
                        "vowel": set(specs_by_lang[la]._features.get("vowel", ()))
                        | set(specs_by_lang[lb]._features.get("vowel", ())),
                    },
                )
                d = _DistanceLite(merged_bit)
                results[(la, lb)] = d.hamming(word_ipa_by_lang[la], word_ipa_by_lang[lb])
            else:
                merged_feats = UniversalFeatureEncoder.merge_inventories(
                    features_by_lang[la], features_by_lang[lb]
                )
                tokens_a = specs_by_lang[la].ipa_tokenizer(word_ipa_by_lang[la])
                tokens_b = specs_by_lang[lb].ipa_tokenizer(word_ipa_by_lang[lb])
                results[(la, lb)] = normalised_feature_edit_distance(
                    tokens_a, tokens_b, merged_feats
                )

    return results
