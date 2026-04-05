from bitarray import bitarray

from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.language.eng_uk import FEATURES, PHONEME_FEATURES, VOWELS_SET

MAX_SYLLABLES = 6


def test_bits_per_word():  # pylint: disable=missing-function-docstring
    consonants_set = set(
        filter(lambda phone: phone not in VOWELS_SET, PHONEME_FEATURES)
    )
    bitarray_generator = BitArraySpecification(
        vowels=VOWELS_SET,
        consonants=consonants_set,
        features_per_phoneme=PHONEME_FEATURES,
        features=FEATURES,
        max_syllables_per_text=MAX_SYLLABLES,
    )

    examples = [
        ("cat", "kæt"),  # 1 syllable
        ("strong", "strɒŋ"),  # 1 syllable
        ("banana", "bənænə"),  # 3 vowels => 3 syllables
        ("computer", "kəmpjuːtə"),  # 3 vowels => 3 syllables
        ("strings", "strɪŋz"),  # 1 syllable
        ("insight", "ɪnsaɪt"),  # 2 vowels => 2 syllables
    ]
    assert (
        bitarray_generator._max_syllables_per_text_chunk == MAX_SYLLABLES
    ), "Maximum bits per Syllable"

    bitarrays = {}
    for w, p in examples:
        bitarrays.update({w: bitarray_generator.ipa_to_syllable(p)})

    assert bitarrays == {
        "cat": [
            {
                "onset": bitarray("00000000001010"),
                "nucleus": bitarray("0001001000"),
                "coda": bitarray("01000000001000"),
            }
        ],
        "strong": [
            {
                "onset": bitarray("01101000001101"),
                "nucleus": bitarray("1000001010"),
                "coda": bitarray("00000000100011"),
            }
        ],
        "banana": [
            {"onset": bitarray("00000010001001"), "nucleus": bitarray("0100000100")},
            {"onset": bitarray("01000000100001"), "nucleus": bitarray("0001001000")},
            {"onset": bitarray("01000000100001"), "nucleus": bitarray("0100000100")},
        ],
        "computer": [
            {"onset": bitarray("00000000001010"), "nucleus": bitarray("0100000100")},
            {"onset": bitarray("00100010111001"), "nucleus": bitarray("1000110011")},
            {"onset": bitarray("01000000001000"), "nucleus": bitarray("0100000100")},
        ],
        "strings": [
            {
                "onset": bitarray("01101000001101"),
                "nucleus": bitarray("0001100000"),
                "coda": bitarray("01001000100011"),
            }
        ],
        "insight": [
            {"nucleus": bitarray("0001100000")},
            {
                "onset": bitarray("01001000100001"),
                "nucleus": bitarray("1000111000"),
                "coda": bitarray("01000000001000"),
            },
        ],
    }
