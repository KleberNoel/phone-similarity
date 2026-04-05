from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.language.eng_uk import FEATURES, PHONEME_FEATURES, VOWELS_SET


def test_make_empty_vector():  # pylint: disable=missing-function-docstring
    consonants_set = set(
        filter(lambda phone: phone not in VOWELS_SET, PHONEME_FEATURES)
    )
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
    assert (
        len(bitarray_specification.empty_vector(feature_type="vowel")) == 10
    ), "Max Syllable Length (in bits)"
    assert max_syllable_length == 38, "Max Syllable Length (in bits)"
