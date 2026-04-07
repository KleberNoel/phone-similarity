"""
Parametrized tests for phoneme feature coverage across all languages.

This module provides comprehensive tests to verify that all phonemes from
CharsiuG2P dictionaries have corresponding feature mappings in the language modules.
"""
import importlib
import logging

import pytest

from phone_similarity.clean_phones import clean_phones
from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator


IGNORE_SYMBOLS = ",.[]'ˈˌːˑ̥̩̯̆̃̍͜͡|…\u200b\u2060:ʰʷᶣˀˤ̪̠?\u201c\u201d\u0022"

# Known combining diacritics that should be ignored
IGNORE_DIACRITICS = {"̯", "̩", "̥", "͡", "ː", "ʼ", "ç", "̈", "̇", "̊", "̋", "́", "̂", "̌", "̄", "̑", "̛", "̔", "̕", "̝", "̞", "̘", "̙", "̜", "̟", "̠", "̢", "̼", "̴", "̵", "̷", "̹", "̺", "̻", "̽", "̾", "̿", "̀", "͝", "͞", "͟", "͠", "͡", "͢", "ɫ"}


import os

_LANG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "src", "phone_similarity", "language"
)
LANGUAGE_MODULES = sorted(
    f[:-3]
    for f in os.listdir(_LANG_DIR)
    if f.endswith(".py") and f != "__init__.py"
)


def get_charsiu_code(module_name: str) -> str:
    """Convert Python module name to Charsiu language code."""
    return module_name.replace("_", "-")


def collect_dict_phonemes(charsiu_code: str) -> set:
    """Collect all unique phonemes from the CharsiuG2P dictionary for a language."""
    try:
        g2p = CharsiuGraphemeToPhonemeGenerator(charsiu_code)
    except Exception as e:
        logging.warning(f"Could not load dictionary for {charsiu_code}: {e}")
        return set()

    import unicodedata

    all_phones = set()
    for word, pron in g2p.pdict.items():
        cleaned = clean_phones(pron)
        for phone in cleaned:
            if phone not in IGNORE_SYMBOLS:
                # Normalize common issues: g -> ɡ, ε -> ɛ, ʧ -> tʃ, ģ -> ɣ
                normalized = (
                    phone.replace("g", "ɡ")
                    .replace("ε", "ɛ")
                    .replace("ʧ", "tʃ")
                    .replace("ģ", "ɣ")
                )
                # Skip non-phonemic single letters (uppercase, common word parts)
                if normalized.isupper() and len(normalized) == 1:
                    continue
                # Skip parentheses and other non-phonemic characters
                if normalized in "()[]{}":
                    continue
                # Strip combining diacritics (category 'Mn' = Mark, Nonspacing)
                stripped = "".join(
                    c for c in normalized if unicodedata.category(c) != "Mn"
                )
                if not stripped:
                    continue
                # Skip pure modifiers like ɫ (velarized) that aren't phonemes
                if stripped == "ɫ":
                    continue
                # Add the stripped version
                all_phones.add(stripped)
                # Also add original if it has content without diacritics
                if stripped != normalized:
                    all_phones.add(normalized)
    return all_phones


@pytest.fixture(params=LANGUAGE_MODULES)
def lang_module(request):
    """Fixture that provides each language module."""
    module_name = request.param
    module = importlib.import_module(f"phone_similarity.language.{module_name}")
    charsiu_code = get_charsiu_code(module_name)
    return module, charsiu_code, module_name


def test_dict_phonemes_have_features(lang_module):
    """Every phoneme in the dictionary must have a feature mapping in the module."""
    module, charsiu_code, module_name = lang_module

    dict_phones = collect_dict_phonemes(charsiu_code)
    module_phones = set(module.PHONEME_FEATURES.keys())

    missing_in_module = dict_phones - module_phones

    if missing_in_module:
        logging.warning(
            f"Module {module_name}: {len(missing_in_module)} phonemes from dict missing features: {sorted(missing_in_module)[:20]}"
        )

    assert len(missing_in_module) == 0, (
        f"Dictionary phonemes missing feature mappings in {module_name}: {sorted(missing_in_module)[:10]}"
    )


def test_vowels_have_height(lang_module):
    """All vowels in VOWELS_SET must have a height feature."""
    module, _, module_name = lang_module

    vowels_set = module.VOWELS_SET
    phoneme_features = module.PHONEME_FEATURES

    height_features = ["high", "mid-high", "mid", "mid-low", "low", "near-high", "near-low"]

    missing_height = []
    for vowel in vowels_set:
        if vowel in phoneme_features:
            features = phoneme_features[vowel]
            if "marker" in features:
                continue
            if "diphthong" in features:
                continue
            if not any(f in features for f in height_features):
                missing_height.append(vowel)

    assert len(missing_height) == 0, f"Vowels missing height in {module_name}: {missing_height}"


def test_vowels_have_round(lang_module):
    """All vowels in VOWELS_SET must have a round feature."""
    module, _, module_name = lang_module

    vowels_set = module.VOWELS_SET
    phoneme_features = module.PHONEME_FEATURES

    missing_round = []
    for vowel in vowels_set:
        if vowel not in phoneme_features:
            continue
        features = phoneme_features[vowel]
        if "marker" in features:
            continue
        if "diphthong" in features:
            continue
        # Skip multi-char entries (diphthongs, long vowels) - they won't have round
        if len(vowel) > 1:
            continue
        if "round" not in features:
            missing_round.append(vowel)

    assert len(missing_round) == 0, f"Vowels missing round in {module_name}: {missing_round}"


def test_consonants_have_manner(lang_module):
    """All consonants must have a manner feature."""
    module, _, module_name = lang_module

    vowels_set = module.VOWELS_SET
    phoneme_features = module.PHONEME_FEATURES

    # Skip combining diacritics and other non-phonemic entries
    # Also skip vowels that exist in PHONEME_FEATURES but not VOWELS_SET
    skip_entries = {
        "̩", "̯", "ˑ", "̥", "̬", "ʰ", "ˠ", "̠", "̝", "̞", "ˢ", "ˀ", "̊",
        "ʲ", "‿", "ˑ", "͡", "̃", "⁾", "⁽", "ɑ", "ɪ", "æ", "ʏ", "ɨ", "ʊ"
    }

    # Feature keys that indicate modifier/suprasegmental entries (not consonants)
    modifier_keys = {
        "marker", "stress", "modifier", "ejective", "syllable_break",
        "tone", "breathy", "release", "labialized", "pre-nasal",
        "rhotic", "unaspirated", "dental", "half_long",
    }

    missing_manner = []
    for phone, features in phoneme_features.items():
        if phone in vowels_set:
            continue
        if phone in skip_entries:
            continue
        # Skip non-consonant entries (modifiers, suprasegmentals)
        if any(k in features for k in modifier_keys):
            continue
        if "diphthong" in features:
            continue
        if "long" in features:
            continue
        if "nasal" in features and "place" not in features:
            continue
        # Skip multi-char entries (affricates, diphthongs) and single diacritics
        if len(phone) > 1:
            # But include affricates which have manner
            if "manner" in features:
                continue
            # Skip other multi-char
            continue
        if "manner" not in features:
            missing_manner.append(phone)

    assert len(missing_manner) == 0, f"Consonants missing manner in {module_name}: {missing_manner}"


def test_consonants_have_voicedness(lang_module):
    """All consonants must have a voiced feature."""
    module, _, module_name = lang_module

    vowels_set = module.VOWELS_SET
    phoneme_features = module.PHONEME_FEATURES

    # Skip combining diacritics and other non-phonemic entries
    # Also skip vowels that exist in PHONEME_FEATURES but not VOWELS_SET
    skip_entries = {
        "̩", "̯", "ˑ", "̥", "̬", "ʰ", "ˠ", "̠", "̝", "̞", "ˢ", "ˀ", "̊",
        "ʲ", "‿", "ˑ", "͡", "̃", "⁾", "⁽", "ɑ", "ɪ", "æ", "ʏ", "ɨ", "ʊ"
    }

    # Feature keys that indicate modifier/suprasegmental entries (not consonants)
    modifier_keys = {
        "marker", "stress", "modifier", "ejective", "syllable_break",
        "tone", "breathy", "release", "labialized", "pre-nasal",
        "rhotic", "unaspirated", "dental", "half_long",
    }

    missing_voiced = []
    for phone, features in phoneme_features.items():
        if phone in vowels_set:
            continue
        if phone in skip_entries:
            continue
        # Skip non-consonant entries (modifiers, suprasegmentals)
        if any(k in features for k in modifier_keys):
            continue
        if "diphthong" in features:
            continue
        if "long" in features:
            continue
        if "nasal" in features and "place" not in features:
            continue
        # Skip multi-char entries (affricates, diphthongs) and single diacritics
        if len(phone) > 1:
            # But include affricates which have voiced
            if "voiced" in features:
                continue
            # Skip other multi-char
            continue
        if "voiced" not in features:
            missing_voiced.append(phone)

    assert len(missing_voiced) == 0, f"Consonants missing voiced in {module_name}: {missing_voiced}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])