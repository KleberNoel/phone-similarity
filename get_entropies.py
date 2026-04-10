import importlib
import logging

from bitarray import frozenbitarray

from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.clean_phones import clean_phones
from phone_similarity.entropy_analyzer import PhonemeEntropyAnalyzer, SyllableEncoding
from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator

# Suppress warnings
logging.getLogger().setLevel(logging.ERROR)

LANGUAGES_TO_TEST = ["ita", "rus", "spa", "fra", "eng_us", "ger"]

for language in LANGUAGES_TO_TEST:
    try:
        module_name = f"phone_similarity.language.{language}"
        lang_module = importlib.import_module(module_name)
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
        analyzer = PhonemeEntropyAnalyzer(bitarray_spec)

        # Limit processing to first 5000 words for speed
        count = 0
        for _word, pronounciations in g2p.pdict.items():
            for p in pronounciations.split():
                try:
                    b_array = bitarray_spec.ipa_to_syllable(clean_phones(p))
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
                    analyzer.add_word(SyllableEncoding(syllables=syllables))
                except (KeyError, ValueError, IndexError) as exc:
                    logging.debug("Skipping phoneme %r for %s: %s", p, language, exc)
                    continue
            count += 1
            if count >= 5000:
                break

        metrics = analyzer.get_entropy_metrics()
        print(
            f"'{language}': {{'onset': ({round(metrics.onset_entropy - 0.5, 1)}, {round(metrics.onset_entropy + 0.5, 1)}), 'nucleus': ({round(metrics.nucleus_entropy - 0.5, 1)}, {round(metrics.nucleus_entropy + 0.5, 1)}), 'coda': ({round(metrics.coda_entropy - 0.5, 1)}, {round(metrics.coda_entropy + 0.5, 1)})}},"
        )
    except Exception as e:
        print(f"Error processing {language}: {e}")
