"""
Parametrized tests for phoneme feature coverage across all languages.

Each language module is tested once for completeness: dictionary phonemes
have feature mappings, vowels have height and rounding, consonants have
manner and voicedness.
"""

import logging
import unicodedata

import pytest

from phone_similarity.clean_phones import clean_phones
from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator
from phone_similarity.language import LANGUAGES

# ---------------------------------------------------------------------------

IGNORE_SYMBOLS = ",.[]'ˈˌːˑ̥̩̯̆̃̍͜͡|…\u200b\u2060:ʰʷᶣˀˤ̪̠?\u201c\u201d\u0022"

LANGUAGE_MODULES = sorted(LANGUAGES.keys())

_HEIGHT_FEATURES = {"high", "mid-high", "mid", "mid-low", "low", "near-high", "near-low"}

_SKIP_ENTRIES = {
    "̩",
    "̯",
    "ˑ",
    "̥",
    "̬",
    "ʰ",
    "ˠ",
    "̠",
    "̝",
    "̞",
    "ˢ",
    "ˀ",
    "̊",
    "ʲ",
    "‿",
    "͡",
    "̃",
    "⁾",
    "⁽",
    "ɑ",
    "ɪ",
    "æ",
    "ʏ",
    "ɨ",
    "ʊ",
}

_MODIFIER_KEYS = {
    "marker",
    "stress",
    "modifier",
    "ejective",
    "syllable_break",
    "tone",
    "breathy",
    "release",
    "labialized",
    "pre-nasal",
    "rhotic",
    "unaspirated",
    "dental",
    "half_long",
}


def _get_charsiu_code(module_name: str) -> str:
    return module_name.replace("_", "-")


def _collect_dict_phonemes(charsiu_code: str) -> set:
    try:
        g2p = CharsiuGraphemeToPhonemeGenerator(charsiu_code)
    except Exception as e:
        logging.warning(f"Could not load dictionary for {charsiu_code}: {e}")
        return set()

    all_phones: set[str] = set()
    for _word, pron in g2p.pdict.items():
        cleaned = clean_phones(pron)
        for phone in cleaned:
            if phone in IGNORE_SYMBOLS:
                continue
            normalized = (
                phone.replace("g", "ɡ").replace("ε", "ɛ").replace("ʧ", "tʃ").replace("ģ", "ɣ")
            )
            if normalized.isupper() and len(normalized) == 1:
                continue
            if normalized in "()[]{}":
                continue
            stripped = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
            if not stripped or stripped == "ɫ":
                continue
            all_phones.add(stripped)
            if stripped != normalized:
                all_phones.add(normalized)
    return all_phones


def _is_consonant_entry(phone, features, vowels_set):
    """Return True if this PHONEME_FEATURES entry is a true consonant."""
    if phone in vowels_set or phone in _SKIP_ENTRIES:
        return False
    if any(k in features for k in _MODIFIER_KEYS):
        return False
    if "diphthong" in features or "long" in features:
        return False
    if "nasal" in features and "place" not in features:
        return False
    return not (len(phone) > 1 and "manner" not in features and "voiced" not in features)


@pytest.fixture(params=LANGUAGE_MODULES)
def lang_module(request):
    module_name = request.param
    module = LANGUAGES[module_name]
    charsiu_code = _get_charsiu_code(module_name)
    return module, charsiu_code, module_name


def test_language_feature_completeness(lang_module):
    """All phoneme feature constraints for a single language.

    Checks:
    1. Every dictionary phoneme has a feature mapping.
    2. Every vowel has a height feature.
    3. Every monophthong vowel has a rounding feature.
    4. Every consonant has a manner feature.
    5. Every consonant has a voicedness feature.
    """
    module, charsiu_code, module_name = lang_module
    phoneme_features = module.PHONEME_FEATURES
    vowels_set = module.VOWELS_SET
    errors: list[str] = []

    # 1 — dictionary coverage
    dict_phones = _collect_dict_phonemes(charsiu_code)
    missing_in_module = dict_phones - set(phoneme_features.keys())
    if missing_in_module:
        errors.append(f"Dict phonemes missing features: {sorted(missing_in_module)[:10]}")

    # 2 — vowel height
    missing_height = [
        v
        for v in vowels_set
        if v in phoneme_features
        and "marker" not in phoneme_features[v]
        and "diphthong" not in phoneme_features[v]
        and not (_HEIGHT_FEATURES & set(phoneme_features[v]))
    ]
    if missing_height:
        errors.append(f"Vowels missing height: {missing_height}")

    # 3 — vowel rounding
    missing_round = [
        v
        for v in vowels_set
        if v in phoneme_features
        and "marker" not in phoneme_features[v]
        and "diphthong" not in phoneme_features[v]
        and len(v) == 1
        and "round" not in phoneme_features[v]
    ]
    if missing_round:
        errors.append(f"Vowels missing round: {missing_round}")

    # 4 — consonant manner
    missing_manner = [
        ph
        for ph, feats in phoneme_features.items()
        if _is_consonant_entry(ph, feats, vowels_set) and "manner" not in feats
    ]
    if missing_manner:
        errors.append(f"Consonants missing manner: {missing_manner}")

    # 5 — consonant voicedness
    missing_voiced = [
        ph
        for ph, feats in phoneme_features.items()
        if _is_consonant_entry(ph, feats, vowels_set) and "voiced" not in feats
    ]
    if missing_voiced:
        errors.append(f"Consonants missing voiced: {missing_voiced}")

    assert not errors, f"Feature completeness errors in {module_name}:\n" + "\n".join(errors)
