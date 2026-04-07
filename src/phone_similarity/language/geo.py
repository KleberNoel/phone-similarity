VOWELS_SET = {
    "i",
    "u",
    "ɑ",
    "ɔ",
    "ɛ",
}

PHONEME_FEATURES = {
    # Vowels
    "i": {"front": True, "high": True, "round": False},
    "u": {"back": True, "high": True, "round": True},
    "ɑ": {"back": True, "low": True, "round": False},
    "ɔ": {"back": True, "mid-low": True, "round": True},
    "ɛ": {"front": True, "mid-low": True, "round": False},
    # Consonants
    "b": {"labial": True, "manner": "plosive", "voiced": True},
    "d": {"manner": "plosive", "place": "alveolar", "voiced": True},
    "d͡": {"manner": "plosive", "place": "alveolar", "voiced": True},
    "h": {"manner": "fricative", "place": "glottal", "voiced": False},
    "k": {"manner": "plosive", "place": "velar", "voiced": False},
    "kʰ": {"aspirated": True, "manner": "plosive", "place": "velar", "voiced": False},
    "l": {"manner": "lateral_approximant", "place": "alveolar", "voiced": True},
    "m": {"labial": True, "manner": "nasal", "voiced": True},
    "n": {"manner": "nasal", "place": "alveolar", "voiced": True},
    "p": {"labial": True, "manner": "plosive", "voiced": False},
    "pʰ": {"aspirated": True, "labial": True, "manner": "plosive", "voiced": False},
    "q": {"manner": "plosive", "place": "uvular", "voiced": False},
    "r": {"manner": "trill", "place": "alveolar", "voiced": True},
    "s": {"manner": "fricative", "place": "alveolar", "voiced": False},
    "t": {"manner": "plosive", "place": "alveolar", "voiced": False},
    "tʰ": {"aspirated": True, "manner": "plosive", "place": "alveolar", "voiced": False},
    "t͡": {"manner": "plosive", "place": "alveolar", "voiced": False},
    "v": {"dental": True, "labial": True, "manner": "fricative", "voiced": True},
    "x": {"manner": "fricative", "place": "velar", "voiced": False},
    "z": {"manner": "fricative", "place": "alveolar", "voiced": True},
    "ɡ": {"manner": "plosive", "place": "velar", "voiced": True},
    "ɣ": {"manner": "fricative", "place": "velar", "voiced": True},
    "ʃ": {"manner": "fricative", "place": "post-alveolar", "voiced": False},
    "ʒ": {"manner": "fricative", "place": "post-alveolar", "voiced": True},
    # Modifiers
    "ʼ": {"ejective": True},
}

CONSONANT_COLUMNS = {
    "alveolar",
    "aspirated",
    "dental",
    "fricative",
    "glottal",
    "labial",
    "lateral_approximant",
    "nasal",
    "plosive",
    "post-alveolar",
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
    "mid-low",
    "round",
}

MODIFIER_COLUMNS = {
    "ejective",
}

FEATURES = {"consonant": CONSONANT_COLUMNS, "vowel": VOWEL_COLUMNS, "modifier": MODIFIER_COLUMNS}
