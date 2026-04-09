from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.language.eng_uk import FEATURES, PHONEME_FEATURES, VOWELS_SET


def test_splitting():
    consonants_set = set(filter(lambda phone: phone not in VOWELS_SET, PHONEME_FEATURES))
    bitarray_generator = BitArraySpecification(
        vowels=VOWELS_SET,
        consonants=consonants_set,
        features_per_phoneme=PHONEME_FEATURES,
        features=FEATURES,
    )
    syllables = bitarray_generator.ipa_to_syllable("nɑːsti")
    assert len(syllables) == 2
