from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.language import LANGUAGES

_eng_uk = LANGUAGES["eng_uk"]
VOWELS_SET = _eng_uk.VOWELS_SET
PHONEME_FEATURES = _eng_uk.PHONEME_FEATURES
FEATURES = _eng_uk.FEATURES


def test_make_empty_vector():  # pylint: disable=missing-function-docstring
    consonants_set = set(filter(lambda phone: phone not in VOWELS_SET, PHONEME_FEATURES))
    bitarray_specification = BitArraySpecification(
        vowels=VOWELS_SET,
        consonants=consonants_set,
        features_per_phoneme=PHONEME_FEATURES,
        features=FEATURES,
    )

    max_syllable_length = len(
        bitarray_specification.empty_vector(feature_type="consonant")
    ) * 2 + len(bitarray_specification.empty_vector(feature_type="vowel"))

    assert max_syllable_length == 38, "Max Syllable Length (in bits)"
    assert len(bitarray_specification.empty_vector(feature_type="vowel")) == 10, (
        "Max Syllable Length (in bits)"
    )
    assert max_syllable_length == 38, "Max Syllable Length (in bits)"
