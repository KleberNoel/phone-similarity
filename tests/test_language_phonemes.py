import importlib
import unittest
import unicodedata
from pathlib import Path
from typing import Set

import phone_similarity.language
from phone_similarity.g2p.charsiu.load_dictionary import load_dictionary_tsv

LANGUAGE_PATH = Path(phone_similarity.language.__file__).parent

# Symbols to ignore in dictionaries as they are typically non-phonemic markers
# or redundant diacritics for the purpose of this v0/v1 check.
IGNORE_SYMBOLS = ",.[]'ˈˌːˑ̥̩̯̆̃̍͜͡|…\u200b\u2060:ʰʷᶣˀˤ̪̠?"


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
    # Standardize Latin 'g' to IPA 'ɡ' (U+0261) for comparison with dictionaries
    return s.replace("g", "ɡ")


def rm_suffix(x: Path):  # pylint: disable=missing-function-docstring
    return x.with_suffix("")


def is_language_module(x: Path):  # pylint: disable=missing-function-docstring
    return not (
        str(x.name).startswith("__") or str(x.name).startswith(".") or x.is_dir()
    )


class TestLanguagePhonemes(  # pylint: disable=missing-class-docstring
    unittest.TestCase
):
    def test_phonemes_in_dictionary(self):  # pylint: disable=missing-function-docstring
        for lang_file in filter(is_language_module, LANGUAGE_PATH.iterdir()):
            module_name = str(rm_suffix(lang_file).name)
            lang = module_name.replace("_", "-")
            with self.subTest(lang=lang):
                lang_module = importlib.import_module(
                    f"phone_similarity.language.{module_name}"
                )
                # Split up the phonemes into constituent parts (e.g. ignore dipthongs for now)
                # Normalize to NFKD to decompose characters and handle diacritics consistently
                module_phones: Set[str] = set()
                for p in lang_module.PHONEME_FEATURES.keys():
                    p_norm = decompose_ligatures(unicodedata.normalize("NFKD", p))
                    for phone_char in p_norm:
                        if phone_char not in IGNORE_SYMBOLS:
                            module_phones.add(phone_char)

                # Ignore non-phonetic symbols
                dict_phones: Set[str] = set()
                for word in load_dictionary_tsv(lang).values():
                    word_norm = decompose_ligatures(unicodedata.normalize("NFKD", word))
                    for phone_char in word_norm:
                        if phone_char not in IGNORE_SYMBOLS:
                            dict_phones.add(phone_char)

                # Check if all module phonemes are in the dictionary and vice-versa
                phones_in_module_not_dict = module_phones - dict_phones
                phones_in_dict_not_module = dict_phones - module_phones

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
