import logging

import pytest
from bitarray import frozenbitarray

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from phone_similarity.analysis.entropy import PhonemeEntropyAnalyzer, SyllableEncoding
from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.clean_phones import clean_phones
from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator


def _make_spec(vowels_set, phoneme_features, features):
    consonants_set = set(filter(lambda ph: ph not in vowels_set, phoneme_features))
    return BitArraySpecification(
        vowels=vowels_set,
        consonants=consonants_set,
        features_per_phoneme=phoneme_features,
        features=features,
    )


def _run_dictionary_metrics(charsiu_code, spec):
    g2p = CharsiuGraphemeToPhonemeGenerator(charsiu_code)
    bitarrays = {}
    for word, pronunciations in tqdm(g2p.pdict.items()) if tqdm else g2p.pdict.items():
        for p in pronunciations.split():
            try:
                bitarray_for_word = spec.ipa_to_syllable(clean_phones(p))
                if word not in bitarrays:
                    bitarrays[word] = []
                bitarrays[word].append(bitarray_for_word)
            except IndexError as index_error:
                logging.error(index_error)
                continue

    analyzer = PhonemeEntropyAnalyzer(spec)
    for _, b_array_list in bitarrays.items():
        for b_array in b_array_list:
            syllables = []
            for syllable in b_array:
                onset = frozenbitarray(syllable.get("onset", spec.empty_vector("consonant")))
                nucleus = frozenbitarray(syllable.get("nucleus", spec.empty_vector("vowel")))
                coda = frozenbitarray(syllable.get("coda", spec.empty_vector("consonant")))
                syllables.append((onset, nucleus, coda))
            encoding = SyllableEncoding(syllables=syllables)
            analyzer.add_word(encoding)

    metrics = analyzer.get_entropy_metrics()
    assert metrics.unique_onset_patterns > 0
    assert metrics.unique_nucleus_patterns > 0
    assert metrics.unique_coda_patterns > 0
    assert metrics.total_words > 0


@pytest.mark.skipif(tqdm is None, reason="tqdm not installed")
@pytest.mark.parametrize(
    "charsiu_code, lang_module",
    [
        ("eng-us", "eng_uk"),
        ("fra", "fra"),
    ],
    ids=["english", "french"],
)
def test_dictionary_metrics(charsiu_code, lang_module):
    import importlib

    mod = importlib.import_module(f"phone_similarity.language.{lang_module}")
    spec = _make_spec(mod.VOWELS_SET, mod.PHONEME_FEATURES, mod.FEATURES)
    _run_dictionary_metrics(charsiu_code, spec)
