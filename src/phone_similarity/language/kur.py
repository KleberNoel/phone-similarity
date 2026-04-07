VOWELS_SET = {
    "a",
    "e",
    "i",
    "o",
    "u",
    "y",
    "ê",
    "î",
    "û",
}

PHONEME_FEATURES = {
    # Vowels
    "a": {"front": True, "low": True, "round": False},
    "e": {"front": True, "mid-high": True, "round": False},
    "i": {"front": True, "high": True, "round": False},
    "o": {"back": True, "mid-high": True, "round": True},
    "u": {"back": True, "high": True, "round": True},
    "y": {"front": True, "high": True, "round": True},
    # Consonants
    ".": {"syllable_break": True},
    "[": {"marker": True},
    "]": {"marker": True},
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
    "q": {"manner": "plosive", "place": "uvular", "voiced": False},
    "r": {"manner": "trill", "place": "alveolar", "voiced": True},
    "s": {"manner": "fricative", "place": "alveolar", "voiced": False},
    "t": {"manner": "plosive", "place": "alveolar", "voiced": False},
    "v": {"dental": True, "labial": True, "manner": "fricative", "voiced": True},
    "w": {"labial": True, "manner": "approximant", "place": "velar", "voiced": True},
    "x": {"manner": "fricative", "place": "velar", "voiced": False},
    "z": {"manner": "fricative", "place": "alveolar", "voiced": True},
    "ç": {"manner": "fricative", "place": "palatal", "voiced": False},
    "ê": {"front": True, "mid-high": True, "round": False},
    "î": {"front": True, "high": True, "round": False},
    "û": {"back": True, "high": True, "round": True},
    "ł": {"labial": True, "manner": "approximant", "place": "velar", "voiced": True},
    "ř": {"manner": "trill", "place": "alveolar", "voiced": True},
    "ş": {"manner": "fricative", "place": "alveolar", "voiced": False},
    "ƹ": {"manner": "fricative", "place": "pharyngeal", "voiced": True},
    "ʔ": {"manner": "plosive", "place": "glottal", "voiced": False},
    "ة": {"marker": True},
    "ḧ": {"manner": "fricative", "place": "glottal", "voiced": False},
    "ẍ": {"manner": "fricative", "place": "velar", "voiced": False},
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
    "trill",
    "uvular",
    "velar",
    "voiced",
}

VOWEL_COLUMNS = {
    "back",
    "front",
    "high",
    "low",
    "mid-high",
    "round",
}

MODIFIER_COLUMNS = {
    "syllable_break",
}

FEATURES = {"consonant": CONSONANT_COLUMNS, "vowel": VOWEL_COLUMNS, "modifier": MODIFIER_COLUMNS}
