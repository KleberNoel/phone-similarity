import logging

from bitarray import frozenbitarray
from tqdm import tqdm

from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.clean_phones import clean_phones
from phone_similarity.entropy_analyzer import PhonemeEntropyAnalyzer, SyllableEncoding
from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator
from phone_similarity.language.eng_uk import FEATURES, PHONEME_FEATURES, VOWELS_SET

ISO_689_3_LANGUAGE = "eng"
ISO_3166_1_ALPHA_2 = "us"

CONSONANTS_SET = set(filter(lambda phone: phone not in VOWELS_SET, PHONEME_FEATURES))

BITARRAY_SPEC = BitArraySpecification(
    vowels=VOWELS_SET,
    consonants=CONSONANTS_SET,
    features_per_phoneme=PHONEME_FEATURES,
    features=FEATURES,
)

MAX_SYLLABLES = 6


def test_dictionary_metrics():  # pylint: disable=missing-function-docstring
    language = f"{ISO_689_3_LANGUAGE}-{ISO_3166_1_ALPHA_2}"
    g2p = CharsiuGraphemeToPhonemeGenerator(language)

    bitarrays = {}

    # TODO: add this to function or class..
    for word, pronounciations in tqdm(g2p.pdict.items()):
        for p in pronounciations.split():
            try:
                bitarray_for_word = BITARRAY_SPEC.ipa_to_syllable(clean_phones(p))
                if word not in bitarrays:
                    bitarrays[word] = []
                bitarrays[word].append(bitarray_for_word)
            except IndexError as index_error:
                logging.error(index_error)
                continue

    analyzer = PhonemeEntropyAnalyzer(BITARRAY_SPEC)
    for _, b_array_list in bitarrays.items():
        for b_array in b_array_list:
            syllables = []
            for syllable in b_array:
                onset = frozenbitarray(
                    syllable.get("onset", BITARRAY_SPEC.empty_vector("consonant"))
                )
                nucleus = frozenbitarray(
                    syllable.get("nucleus", BITARRAY_SPEC.empty_vector("vowel"))
                )
                coda = frozenbitarray(
                    syllable.get("coda", BITARRAY_SPEC.empty_vector("consonant"))
                )
                syllables.append((onset, nucleus, coda))

            encoding = SyllableEncoding(syllables=syllables)
            analyzer.add_word(encoding)

    metrics = analyzer.get_entropy_metrics()
    assert metrics.unique_onset_patterns > 0
    assert metrics.unique_nucleus_patterns > 0
    assert metrics.unique_coda_patterns > 0
    assert metrics.total_words > 0
