# pylint: disable=missing-docstring
from typing import Literal

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

    langs = sorted(word_ipa_by_lang)
    results: dict[tuple[str, str], float] = {}

    def bit_union(
        lang_a,
        lang_b,
        spec: dict[str, BitArraySpecification],
        features_part: Literal["consonant", "vowel"],
    ):
        return set(spec[lang_a].features[features_part]) | set(
            spec[lang_b].features[features_part]
        )

    # TODO: improve with collections library e.g. do pairwise comparison using
    # -> product or combinations...  # pylint: disable=fixme
    for i, lang_a in enumerate(langs):
        for lang_b in langs[i + 1 :]:
            if metric == "hamming":
                merged = IntersectingBitArraySpecification(
                    [specs_by_lang[lang_a], specs_by_lang[lang_b]]
                )
                merged_bit = BitArraySpecification(  # Merging=comparison btwn bitarray lang specs
                    vowels=merged.vowels,
                    consonants=merged.consonants,
                    features_per_phoneme=merged.phoneme_features,
                    features={
                        "consonant": bit_union(
                            lang_a, lang_b, specs_by_lang, features_part="consonant"
                        ),
                        "vowel": bit_union(
                            lang_a, lang_b, specs_by_lang, features_part="vowel"
                        ),
                    },
                )

                arr_a = merged_bit.ipa_to_bitarray(
                    word_ipa_by_lang[lang_a],
                    max_syllables=6,  #  TODO: FIXME - reassess this number of syllables (is it good for all languages?) - parameterize if needed...
                )
                arr_b = merged_bit.ipa_to_bitarray(
                    word_ipa_by_lang[lang_b], max_syllables=6
                )

                results[(lang_a, lang_b)] = hamming_similarity(arr_a, arr_b)
            else:
                merged_feats = UniversalFeatureEncoder.merge_inventories(
                    features_by_lang[lang_a], features_by_lang[lang_b]
                )
                tokens_a = specs_by_lang[lang_a].ipa_tokenizer(word_ipa_by_lang[lang_a])
                tokens_b = specs_by_lang[lang_b].ipa_tokenizer(word_ipa_by_lang[lang_b])
                results[(lang_a, lang_b)] = normalised_feature_edit_distance(
                    tokens_a, tokens_b, merged_feats  # type: ignore
                )

    return results
