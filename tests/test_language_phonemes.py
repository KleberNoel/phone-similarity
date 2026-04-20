import unicodedata
import unittest

import pytest

pytestmark = pytest.mark.slow

from phone_similarity.g2p.charsiu.load_dictionary import load_dictionary_tsv
from phone_similarity.language import LANGUAGES

# Symbols to ignore in dictionaries as they are typically non-phonemic markers
# or redundant diacritics for the purpose of this v0/v1 check.
IGNORE_SYMBOLS = ",.[]'ˈˌːˑ̥̩̯̆̃̍͜͡|…\u200b\u2060:ʰʷᶣˀˤ̪̠?"


# Phoneme *characters* that are valid IPA for a language but are not used in
# the CharsiuG2P dictionary transcription convention.  Keyed by Charsiu
# language code (e.g. "eng-us").  These are excluded from the "module phonemes
# must all appear in dictionary" assertion.
_SUPPLEMENTARY_PHONEMES = {
    # CharsiuG2P eng-us uses ə for both /ʌ/ and /ə/, and ɝ for both /ɝ/ and /ɚ/
    "eng-us": {"ʌ", "ɚ"},
}


def decompose_ligatures(s: str) -> str:
    """Decompose common IPA ligatures into their constituent parts."""
    replacements = {
        "ʧ": "tʃ",
        "ʤ": "dʒ",
        "ʦ": "ts",
        "ʣ": "dz",
        "ʨ": "tɕ",
        "ʥ": "dʑ",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s.replace("g", "ɡ")


class TestLanguagePhonemes(  # pylint: disable=missing-class-docstring
    unittest.TestCase
):
    def test_phonemes_in_dictionary(self):  # pylint: disable=missing-function-docstring
        for module_name in sorted(LANGUAGES.keys()):
            lang = module_name.replace("_", "-")
            with self.subTest(lang=lang):
                lang_module = LANGUAGES[module_name]
                # Split up the phonemes into constituent parts (e.g. ignore dipthongs for now)
                # Normalize to NFKD to decompose characters and handle diacritics consistently
                module_phones: set[str] = set()
                for p in lang_module.PHONEME_FEATURES:
                    p_norm = decompose_ligatures(unicodedata.normalize("NFKD", p))
                    for phone_char in p_norm:
                        if phone_char not in IGNORE_SYMBOLS:
                            module_phones.add(phone_char)

                # Ignore non-phonetic symbols
                dict_phones: set[str] = set()
                for word in load_dictionary_tsv(lang).values():
                    word_norm = decompose_ligatures(unicodedata.normalize("NFKD", word))
                    for phone_char in word_norm:
                        if phone_char not in IGNORE_SYMBOLS:
                            dict_phones.add(phone_char)

                # Check if all module phonemes are in the dictionary and vice-versa
                phones_in_module_not_dict = module_phones - dict_phones
                phones_in_dict_not_module = dict_phones - module_phones

                # Exclude known supplementary phonemes (valid IPA but not in
                # the CharsiuG2P transcription convention for this language)
                supplementary = _SUPPLEMENTARY_PHONEMES.get(lang, set())
                phones_in_module_not_dict -= supplementary

                # Minimally, phones in the module must all occur in the dict
                self.assertEqual(
                    len(phones_in_module_not_dict),
                    0,
                    (
                        f"Phonemes in {lang} not found in dictionary (but found in module)"
                        f"{phones_in_module_not_dict}"
                    ),
                )
                assert len(phones_in_dict_not_module) <= 3, (
                    "Minor set difference is allowed, major is not. "
                    f"Phonemes in {lang} not found in module (but found in dictionary) "
                    f"{phones_in_dict_not_module}"
                )
