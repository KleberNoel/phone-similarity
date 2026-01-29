from phone_similarity.g2p.charsiu.load_dictionary import load_dictionary_tsv
from phone_similarity.raw_phone_features import RAW_FEATURES, VOWELS


def get_lang_data(lang_code):
    """
    Get language data for a given language code.
    This includes the vowel set, phoneme features, and features.
    """
    pdict = load_dictionary_tsv(lang_code)
    all_phonemes = set()
    for phones in pdict.values():
        all_phonemes.update(phones.split())

    lang_vowels = VOWELS.intersection(all_phonemes)
    lang_phoneme_features = {p: f for p, f in RAW_FEATURES.items() if p in all_phonemes}
    lang_features = set(
        f for features in lang_phoneme_features.values() for f in features
    )

    return lang_vowels, lang_phoneme_features, lang_features


def print_feature_vector(columns):
    """TODO FIXME"""
    print()
    new_columns = [feat for feat in columns]
    for char_idx in range(max([len(n) for n in new_columns])):
        print("               ", end="")
        for word in new_columns:
            if char_idx < len(word):
                print(word[char_idx], end="  ")
            else:
                print(" ", end="  ")
        print()
    print()


def print_syl_verbose(
    syl, cons_columns, vowel_columns, print_zeros: bool = True
) -> None:
    """TODO FIXME"""
    print(" " * 10, "=== ONSET ===")
    for i in range(len(syl["onset"])):
        if print_zeros:
            print("            ", syl["onset"][i], cons_columns[i])
        elif syl["onset"][i] != 0:
            print("            ", syl["onset"][i], cons_columns[i])

    print(" " * 10, "=== NUCLEUS ===")
    for i in range(len(syl["nucleus"])):
        if print_zeros:
            print("            ", syl["nucleus"][i], vowel_columns[i])
        elif syl["nucleus"][i] != 0:
            print("            ", syl["nucleus"][i], vowel_columns[i])

    print(" " * 10, "=== CODA ===")
    for i in range(len(syl["coda"])):
        if print_zeros:
            print("            ", syl["coda"][i], cons_columns[i])
        elif syl["coda"][i] != 0:
            print("            ", syl["coda"][i], cons_columns[i])
