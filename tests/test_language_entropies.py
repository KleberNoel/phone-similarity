import logging

import pytest
from bitarray import frozenbitarray
from tqdm import tqdm

from phone_similarity.analysis.entropy import PhonemeEntropyAnalyzer, SyllableEncoding
from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.clean_phones import clean_phones
from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator
from phone_similarity.language import LANGUAGES

# List of languages to test
# Omitting por-bz and por-po as they are dialects of Portuguese
# Omitting lat-clas as it's a classical version of Latin
LANGUAGES_TO_TEST = ["ita", "rus", "spa", "fra", "eng_us", "ger"]

# Expected entropy ranges for each language
# Calibrated based on actual metrics produced from Charsiu dictionaries
ENTROPY_RANGES = {
    "ita": {"onset": (4.0, 5.5), "nucleus": (1.5, 3.0), "coda": (0.0, 1.0)},
    "rus": {"onset": (6.0, 7.5), "nucleus": (2.0, 3.5), "coda": (1.0, 2.5)},
    "spa": {"onset": (3.5, 6.0), "nucleus": (1.0, 3.0), "coda": (0.0, 2.0)},
    "fra": {"onset": (3.5, 5.5), "nucleus": (2.0, 4.0), "coda": (0.0, 1.5)},
    "eng_us": {"onset": (4.0, 6.0), "nucleus": (2.5, 5.0), "coda": (1.0, 3.0)},
    "ger": {"onset": (4.0, 6.5), "nucleus": (2.0, 5.0), "coda": (0.5, 3.0)},
}


@pytest.mark.parametrize("language", LANGUAGES_TO_TEST)
def test_language_entropy_metrics(language):
    lang_module = LANGUAGES[language]

    phoneme_features = lang_module.PHONEME_FEATURES

    vowels_set = lang_module.VOWELS_SET
    consonants_set = set(filter(lambda p: p not in vowels_set, phoneme_features))

    bitarray_spec = BitArraySpecification(
        vowels=vowels_set,
        consonants=consonants_set,
        features_per_phoneme=phoneme_features,
        features=lang_module.FEATURES,
    )

    g2p = CharsiuGraphemeToPhonemeGenerator(language.replace("_", "-"), use_cache=True)
    bitarrays = {}

    MAX_WORDS = 10000
    count = 0
    import random

    g2p_pdict = list(g2p.pdict.items())
    random.seed(42)
    random.shuffle(g2p_pdict)
    for word, pronounciations in tqdm(g2p_pdict, desc=f"Processing {language}"):
        if count >= MAX_WORDS:
            break
        for p in pronounciations.split():
            try:
                bitarray_for_word = bitarray_spec.ipa_to_syllable(clean_phones(p))
                if word not in bitarrays:
                    bitarrays[word] = []
                bitarrays[word].append(bitarray_for_word)
            except (IndexError, KeyError) as e:
                logging.warning(f"Skipping '{p}' for word '{word}': {e}")
                continue
        count += 1

    analyzer = PhonemeEntropyAnalyzer(bitarray_spec)
    for _, b_array_list in bitarrays.items():
        for b_array in b_array_list:
            syllables = []
            for syllable in b_array:
                onset = frozenbitarray(
                    syllable.get("onset", bitarray_spec.empty_vector("consonant"))
                )
                nucleus = frozenbitarray(
                    syllable.get("nucleus", bitarray_spec.empty_vector("vowel"))
                )
                coda = frozenbitarray(
                    syllable.get("coda", bitarray_spec.empty_vector("consonant"))
                )
                syllables.append((onset, nucleus, coda))

            encoding = SyllableEncoding(syllables=syllables)
            analyzer.add_word(encoding)

    metrics = analyzer.get_entropy_metrics()

    assert metrics.total_words > 0

    # Check entropy ranges
    onset_min, onset_max = ENTROPY_RANGES[language]["onset"]
    nucleus_min, nucleus_max = ENTROPY_RANGES[language]["nucleus"]
    coda_min, coda_max = ENTROPY_RANGES[language]["coda"]

    assert onset_min <= metrics.onset_entropy <= onset_max, (
        f"Onset entropy for {language} is out of range."
    )
    assert nucleus_min <= metrics.nucleus_entropy <= nucleus_max, (
        f"Nucleus entropy for {language} is out of range."
    )
    assert coda_min <= metrics.coda_entropy <= coda_max, (
        f"Coda entropy for {language} is out of range."
    )
