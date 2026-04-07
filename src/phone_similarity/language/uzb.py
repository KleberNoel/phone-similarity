VOWELS_SET = {
    "a",
    "e",
    "i",
    "o",
    "oː",
    "y",
    "æ",
    "ɑ",
    "ʊ",
}

PHONEME_FEATURES = {
    # Vowels
    "a": {"front": True, "low": True, "round": False},
    "e": {"front": True, "mid-high": True, "round": False},
    "i": {"front": True, "high": True, "round": False},
    "o": {"back": True, "mid-high": True, "round": True},
    "oː": {"back": True, "long": True, "mid-high": True, "round": True},
    "y": {"front": True, "high": True, "round": True},
    "æ": {"front": True, "near-low": True, "round": False},
    "ɑ": {"back": True, "low": True, "round": False},
    "ʊ": {"back": True, "near-high": True, "round": True},
    # Consonants
    "S": {"marker": True},
    "b": {"labial": True, "manner": "plosive", "voiced": True},
    "d": {"manner": "plosive", "place": "alveolar", "voiced": True},
    "d͡": {"manner": "plosive", "place": "alveolar", "voiced": True},
    "f": {"dental": True, "labial": True, "manner": "fricative", "voiced": False},
    "h": {"manner": "fricative", "place": "glottal", "voiced": False},
    "j": {"manner": "approximant", "place": "palatal", "voiced": True},
    "k": {"manner": "plosive", "place": "velar", "voiced": False},
    "l": {"manner": "lateral_approximant", "place": "alveolar", "voiced": True},
    "m": {"labial": True, "manner": "nasal", "voiced": True},
    "n": {"manner": "nasal", "place": "alveolar", "voiced": True},
    "p": {"labial": True, "manner": "plosive", "voiced": False},
    "q": {"manner": "plosive", "place": "uvular", "voiced": False},
    "s": {"manner": "fricative", "place": "alveolar", "voiced": False},
    "t": {"manner": "plosive", "place": "alveolar", "voiced": False},
    "t͡": {"manner": "plosive", "place": "alveolar", "voiced": False},
    "z": {"manner": "fricative", "place": "alveolar", "voiced": True},
    "ŋ": {"manner": "nasal", "place": "velar", "voiced": True},
    "ɡ": {"manner": "plosive", "place": "velar", "voiced": True},
    "ɫ": {"manner": "lateral_approximant", "place": "alveolar", "velarized": True, "voiced": True},
    "ɾ": {"manner": "tap", "place": "alveolar", "voiced": True},
    "ʁ": {"manner": "fricative", "place": "uvular", "voiced": True},
    "ʃ": {"manner": "fricative", "place": "post-alveolar", "voiced": False},
    "ʋ": {"dental": True, "labial": True, "manner": "approximant", "voiced": True},
    "ʒ": {"manner": "fricative", "place": "post-alveolar", "voiced": True},
    "ʔ": {"manner": "plosive", "place": "glottal", "voiced": False},
    "χ": {"manner": "fricative", "place": "uvular", "voiced": False},
    # Modifiers
    "ˈ": {"marker": True},
    "ˌ": {"marker": True},
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
    "post-alveolar",
    "tap",
    "uvular",
    "velar",
    "velarized",
    "voiced",
}

VOWEL_COLUMNS = {
    "back",
    "front",
    "high",
    "long",
    "low",
    "mid-high",
    "near-high",
    "near-low",
    "round",
}

MODIFIER_COLUMNS = {
    "marker",
}

FEATURES = {"consonant": CONSONANT_COLUMNS, "vowel": VOWEL_COLUMNS, "modifier": MODIFIER_COLUMNS}
