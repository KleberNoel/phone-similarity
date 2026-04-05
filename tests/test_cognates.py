# import logging
# from collections import defaultdict
# from itertools import combinations
# from typing import Dict, List
#
# import numpy as np
# from bitarray import bitarray
#
# from phone_similarity.bit_array_specification import BitArraySpecification
# from phone_similarity.clean_phones import clean_phones
# from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator
# from phone_similarity.language.combined import (
#    COMBINED_CONSONANTS,
#    COMBINED_FEATURES,
#    COMBINED_PHONEME_FEATURES,
#    COMBINED_VOWELS,
# )
#
#
# def test_cognates():
#    cognates = {
#        1: ("father", "père", "vater", "vader", "padre"),  # father
#        2: ("mother", "mère", "mutter", "moeder", "madre"),  # mother
#        3: ("brother", "frère", "bruder", "broer", "hermano"),  # brother
#        4: ("sister", "sœur", "schwester", "zus", "hermana"),  # sister
#        5: ("sun", "soleil", "sonne", "zon", "sol"),  # sun
#        6: ("moon", "lune", "mond", "maan", "luna"),  # moon
#        7: ("water", "eau", "wasser", "water", "agua"),  # water
#        8: ("fire", "feu", "feuer", "vuur", "fuego"),  # fire
#    }
#
#    langs = ["eng-us", "fra", "ger", "dut", "spa"]
#    g2ps = {
#        lang: CharsiuGraphemeToPhonemeGenerator(lang, use_cache=True) for lang in langs
#    }
#
#    spec = BitArraySpecification(
#        vowels=COMBINED_VOWELS,
#        consonants=COMBINED_CONSONANTS,
#        features_per_phoneme=COMBINED_PHONEME_FEATURES,
#        features=COMBINED_FEATURES,
#    )
#    langpair_to_cosine_similarity_scores = defaultdict(list)
#
#    cognate_to_bitarray = defaultdict(dict)
#    for i, cognate_set in cognates.items():
#        for l_idx, word in enumerate(cognate_set):
#            pronounciations = g2ps[langs[l_idx]].pdict.get(word)
#            if not pronounciations:
#                logging.error(
#                    f"No pronunciation found for {word} (language: {langs[l_idx]})"
#                )
#                continue
#
#            pronounciation = pronounciations.split()[0]
#            bit_array = spec.ipa_to_syllable(clean_phones(pronounciation))
#            # index i, word idx is same as lang idx in original 'langs' List
#            cognate_to_bitarray[i].update(
#                {
#                    langs[l_idx]: (
#                        word,
#                        bit_array,
#                    )
#                }
#            )
#
#    # Add comparisons here
#    # For now, just check that we can generate bitarrays
#    def unpack_bits(syllable: Dict[str, bitarray], key: str):
#        return (
#            np.frombuffer(
#                syllable.get(key, spec.empty_vector("consonant")).unpack(), dtype="bool"
#            ).astype(int)
#            if key == "consonant"
#            else np.frombuffer(
#                syllable.get(key, spec.empty_vector("vowel")).unpack(), dtype="bool"
#            ).astype(int)
#        )
#
#    def to_onc(syl: Dict[str, bitarray]) -> List[np.ndarray]:
#        return [
#            unpack_bits(syl, "onset"),
#            unpack_bits(syl, "nucleus"),
#            unpack_bits(syl, "coda"),
#        ]
#
#    for i in cognates:
#        for language_a, language_b in combinations(langs, r=2):
#            word_a: str = cognate_to_bitarray[i][language_a][0]
#            word_b: str = cognate_to_bitarray[i][language_b][0]
#
#            try:
#                word_bits_a: np.ndarray = np.vstack(
#                    [
#                        np.hstack(to_onc(syl))
#                        for syl in cognate_to_bitarray[i][language_a][1]
#                    ]
#                )
#            except Exception as e:
#                print(e)
#                continue
#            try:
#                word_bits_b: np.ndarray = np.vstack(
#                    [
#                        np.hstack(to_onc(syl))
#                        for syl in cognate_to_bitarray[i][language_b][1]
#                    ]
#                )
#            except Exception as e:
#                print(e)
#                continue
#
#            # if not (word_bits_a or word_bits_b):
#            #    continue
#
#            if word_bits_a.shape == word_bits_b.shape:
#                try:
#                    cosine_similarity = (np.dot(word_bits_a, word_bits_b.T)) / (
#                        np.linalg.norm(word_bits_a)
#                        * np.linalg.norm(word_bits_b.astype(int))
#                    )
#                    langpair_to_cosine_similarity_scores[
#                        f"{language_a}-{language_b}"
#                    ].append(cosine_similarity)
#                except Exception as e:
#                    print(str(e))
#                    continue
#
#            else:
#                logging.error(
#                    (
#                        "Word A shape is incompatible with Word B shape"
#                        f" {word_bits_a.shape[0]} ({word_a}) != {word_bits_b.shape[0]} ({word_b})"
#                    )
#                )
#                continue
#
#    mean_results = {
#        k: np.mean([np.mean(v0) for v0 in v])
#        for k, v in langpair_to_cosine_similarity_scores.items()
#    }
#    print(mean_results)
