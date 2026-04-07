VOWELS_SET = {
    "a",
    "e",
    "i",
    "o",
    "u",
    "y",
    "ɔ",
    "ə",
    "ɛ",
    "ʉ",
}

PHONEME_FEATURES = {
    # Vowels
    "a": {"front": True, "low": True, "round": False},
    "e": {"front": True, "mid-high": True, "round": False},
    "i": {"front": True, "high": True, "round": False},
    "o": {"back": True, "mid-high": True, "round": True},
    "u": {"back": True, "high": True, "round": True},
    "y": {"front": True, "high": True, "round": True},
    "ɔ": {"back": True, "mid-low": True, "round": True},
    "ə": {"central": True, "mid": True, "round": False},
    "ɛ": {"front": True, "mid-low": True, "round": False},
    "ʉ": {"central": True, "high": True, "round": True},
    # Consonants
    ".": {"syllable_break": True},
    ":": {"marker": True},
    "?": {"marker": True},
    "b": {"labial": True, "manner": "plosive", "voiced": True},
    "c": {"manner": "plosive", "place": "palatal", "voiced": False},
    "d": {"manner": "plosive", "place": "alveolar", "voiced": True},
    "f": {"dental": True, "labial": True, "manner": "fricative", "voiced": False},
    "ɡ": {"manner": "plosive", "place": "velar", "voiced": True},
    "h": {"manner": "fricative", "place": "glottal", "voiced": False},
    "j": {"manner": "approximant", "place": "palatal", "voiced": True},
    "k": {"manner": "plosive", "place": "velar", "voiced": False},
    "l": {"manner": "lateral_approximant", "place": "alveolar", "voiced": True},
    "m": {"labial": True, "manner": "nasal", "voiced": True},
    "n": {"manner": "nasal", "place": "alveolar", "voiced": True},
    "p": {"labial": True, "manner": "plosive", "voiced": False},
    "s": {"manner": "fricative", "place": "alveolar", "voiced": False},
    "t": {"manner": "plosive", "place": "alveolar", "voiced": False},
    "v": {"dental": True, "labial": True, "manner": "fricative", "voiced": True},
    "w": {"labial": True, "manner": "approximant", "place": "velar", "voiced": True},
    "ŋ": {"manner": "nasal", "place": "velar", "voiced": True},
    "ɲ": {"manner": "nasal", "place": "palatal", "voiced": True},
}

CONSONANT_COLUMNS = {
    "alveolar",
    "approximant",
    "consonant",
    "dental",
    "fricative",
    "glottal",
    "labial",
    "lateral_approximant",
    "nasal",
    "palatal",
    "plosive",
    "velar",
    "voiced",
}

VOWEL_COLUMNS = {
    "back",
    "central",
    "front",
    "high",
    "low",
    "mid",
    "mid-high",
    "mid-low",
    "round",
}

MODIFIER_COLUMNS = {
    "syllable_break",
}

FEATURES = {"consonant": CONSONANT_COLUMNS, "vowel": VOWEL_COLUMNS, "modifier": MODIFIER_COLUMNS}
