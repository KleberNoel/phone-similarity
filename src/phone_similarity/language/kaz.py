VOWELS_SET = {
    "a",
    "e",
    "o",
    "u",
    "æ",
    "ɑ",
    "ə",
    "ɪ",
    "ɵ",
    "ʊ",
}

PHONEME_FEATURES = {
    # Vowels
    "a": {"front": True, "low": True, "round": False},
    "e": {"front": True, "mid-high": True, "round": False},
    "o": {"back": True, "mid-high": True, "round": True},
    "u": {"back": True, "high": True, "round": True},
    "æ": {"front": True, "near-low": True, "round": False},
    "ɑ": {"back": True, "low": True, "round": False},
    "ə": {"central": True, "mid": True, "round": False},
    "ɪ": {"front": True, "near-high": True, "round": False},
    "ɵ": {"central": True, "mid-high": True, "round": True},
    "ʊ": {"back": True, "near-high": True, "round": True},
    # Consonants
    "b": {"labial": True, "manner": "plosive", "voiced": True},
    "d": {"manner": "plosive", "place": "alveolar", "voiced": True},
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
    "v": {"dental": True, "labial": True, "manner": "fricative", "voiced": True},
    "w": {"labial": True, "manner": "approximant", "place": "velar", "voiced": True},
    "x": {"manner": "fricative", "place": "velar", "voiced": False},
    "z": {"manner": "fricative", "place": "alveolar", "voiced": True},
    "ŋ": {"manner": "nasal", "place": "velar", "voiced": True},
    "ɡ": {"manner": "plosive", "place": "velar", "voiced": True},
    "ɫ": {"manner": "lateral_approximant", "place": "alveolar", "velarized": True, "voiced": True},
    "ɾ": {"manner": "tap", "place": "alveolar", "voiced": True},
    "ʀ": {"manner": "trill", "place": "uvular", "voiced": True},
    "ʃ": {"manner": "fricative", "place": "post-alveolar", "voiced": False},
    "ʒ": {"manner": "fricative", "place": "post-alveolar", "voiced": True},
    "ʔ": {"manner": "plosive", "place": "glottal", "voiced": False},
    # Modifiers
    "ˈ": {"marker": True},
    "ˌ": {"marker": True},
}

CONSONANT_COLUMNS = {
    "alveolar",
    "approximant",
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
    "trill",
    "uvular",
    "velar",
    "velarized",
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
    "near-high",
    "near-low",
    "round",
}

MODIFIER_COLUMNS = {
    "marker",
}

FEATURES = {"consonant": CONSONANT_COLUMNS, "vowel": VOWEL_COLUMNS, "modifier": MODIFIER_COLUMNS}
