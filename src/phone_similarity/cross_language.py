# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Literal, Union

from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.intersecting_bit_array_specification import (
    IntersectingBitArraySpecification,
)
from phone_similarity.primitives import _HAS_CYTHON, normalised_feature_edit_distance
from phone_similarity.universal_features import UniversalFeatureEncoder

if _HAS_CYTHON:
    from phone_similarity.primitives import _c_hamming_similarity as hamming_similarity
else:
    from phone_similarity.primitives import hamming_similarity


def compare_cross_language(  # pylint: disable=too-many-locals
    word_ipa_by_lang: dict[str, str],
    specs_by_lang: dict[str, BitArraySpecification],
    features_by_lang: dict[str, dict[str, dict]],
    metric: Literal["edit", "hamming"] = "edit",
) -> dict[tuple[str, str], float]:
    """Compare a word's pronunciation across multiple languages.

    Returns ``{(lang_a, lang_b): distance}`` for every unordered pair.
    """
    langs = sorted(word_ipa_by_lang)
    results: dict[tuple[str, str], float] = {}

    def bit_union(
        lang_a: str,
        lang_b: str,
        spec: dict[str, BitArraySpecification],
        features_part: Literal["consonant", "vowel"],
    ) -> set[str]:
        return set(spec[lang_a].features[features_part]) | set(
            spec[lang_b].features[features_part]
        )

    for i, lang_a in enumerate(langs):
        for lang_b in langs[i + 1 :]:
            if metric == "hamming":
                merged = IntersectingBitArraySpecification(
                    [specs_by_lang[lang_a], specs_by_lang[lang_b]]
                )
                merged_bit = BitArraySpecification(
                    vowels=merged.vowels,
                    consonants=merged.consonants,
                    features_per_phoneme=merged.phoneme_features,
                    features={
                        "consonant": bit_union(
                            lang_a, lang_b, specs_by_lang, features_part="consonant"
                        ),
                        "vowel": bit_union(lang_a, lang_b, specs_by_lang, features_part="vowel"),
                    },
                )

                arr_a = merged_bit.ipa_to_bitarray(word_ipa_by_lang[lang_a], max_syllables=6)
                arr_b = merged_bit.ipa_to_bitarray(word_ipa_by_lang[lang_b], max_syllables=6)
                results[(lang_a, lang_b)] = hamming_similarity(arr_a, arr_b)
            else:
                merged_feats = UniversalFeatureEncoder.merge_inventories(
                    features_by_lang[lang_a], features_by_lang[lang_b]
                )
                tokens_a = specs_by_lang[lang_a].ipa_tokenizer(word_ipa_by_lang[lang_a])
                tokens_b = specs_by_lang[lang_b].ipa_tokenizer(word_ipa_by_lang[lang_b])
                results[(lang_a, lang_b)] = normalised_feature_edit_distance(
                    tokens_a,
                    tokens_b,
                    merged_feats,  # type: ignore
                )

    return results
