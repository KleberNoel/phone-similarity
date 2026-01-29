from phone_similarity.language import (
    da,
    de,
    en_gb,
    eng_us,
    es,
    fr,
    geo,
    ger,
    ita,
    lat_clas,
    nl,
    por_bz,
    por_po,
    rus,
    spa,
    sqi,
)


def get_combined_phoneme_features():
    all_features = {}
    # TODO: add all languages
    all_features.update(en_gb.PHONEME_FEATURES)
    all_features.update(fr.PHONEME_FEATURES)
    all_features.update(de.PHONEME_FEATURES)
    all_features.update(nl.PHONEME_FEATURES)
    all_features.update(es.PHONEME_FEATURES)
    return all_features


def get_combined_vowels():
    all_vowels = set()
    # TODO: add all languages
    all_vowels.update(en_gb.VOWELS_SET)
    all_vowels.update(fr.VOWELS_SET)
    all_vowels.update(de.VOWELS_SET)
    all_vowels.update(nl.VOWELS_SET)
    all_vowels.update(es.VOWELS_SET)
    return all_vowels


def get_combined_features():
    consonant_columns = set()
    vowel_columns = set()
    for lang in [en_gb, fr, de, nl, es]:
        consonant_columns.update(lang.FEATURES["consonant"])
        vowel_columns.update(lang.FEATURES["vowel"])
    return {"consonant": consonant_columns, "vowel": vowel_columns}


COMBINED_PHONEME_FEATURES = get_combined_phoneme_features()
COMBINED_VOWELS = get_combined_vowels()
COMBINED_FEATURES = get_combined_features()

COMBINED_CONSONANTS = set(
    filter(lambda p: p not in COMBINED_VOWELS, COMBINED_PHONEME_FEATURES)
)
