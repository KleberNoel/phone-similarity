VOWELS_SET = {
    "a",
    "e",
    "i",
    "o",
    "u",
    "ɛ",
    "ɔ",
    "ɐ",
    "ẽ",
    "ĩ",
    "õ",
    "ũ",
    "ɐ̃",
    "ɔ̃",
}

PHONEME_FEATURES = {
    # Vowels
    "i": {"high": True, "front": True, "round": False},
    "e": {"mid-high": True, "front": True, "round": False},
    "ɛ": {"mid-low": True, "front": True, "round": False},
    "a": {"low": True, "front": True, "round": False},
    "u": {"high": True, "back": True, "round": True},
    "o": {"mid-high": True, "back": True, "round": True},
    "ɔ": {"mid-low": True, "back": True, "round": True},
    "ɐ": {"low": True, "central": True, "round": False},
    # Nasal Vowels
    "ẽ": {"nasal": True, "mid-high": True, "front": True, "round": False},
    "ĩ": {"nasal": True, "high": True, "front": True, "round": False},
    "õ": {"nasal": True, "mid-high": True, "back": True, "round": True},
    "ũ": {"nasal": True, "high": True, "back": True, "round": True},
    "ɐ̃": {"nasal": True, "low": True, "central": True, "round": False},
    "ɔ̃": {"nasal": True, "mid-low": True, "back": True, "round": True},
    # Consonants
    "p": {"voiced": False, "labial": True, "manner": "plosive"},
    "b": {"voiced": True, "labial": True, "manner": "plosive"},
    "t": {"voiced": False, "place": "dental", "manner": "plosive"},
    "d": {"voiced": True, "place": "dental", "manner": "plosive"},
    "k": {"voiced": False, "place": "velar", "manner": "plosive"},
    "ɡ": {"voiced": True, "place": "velar", "manner": "plosive"},
    "f": {"voiced": False, "labial": True, "dental": True, "manner": "fricative"},
    "v": {"voiced": True, "labial": True, "dental": True, "manner": "fricative"},
    "β": {"voiced": True, "labial": True, "manner": "fricative"},
    "ð": {"voiced": True, "place": "dental", "manner": "fricative"},
    "ɣ": {"voiced": True, "place": "velar", "manner": "fricative"},
    "s": {"voiced": False, "place": "alveolar", "manner": "fricative"},
    "z": {"voiced": True, "place": "alveolar", "manner": "fricative"},
    "ʃ": {"voiced": False, "place": "post-alveolar", "manner": "fricative"},
    "ʒ": {"voiced": True, "place": "post-alveolar", "manner": "fricative"},
    "m": {"voiced": True, "labial": True, "manner": "nasal"},
    "n": {"voiced": True, "place": "alveolar", "manner": "nasal"},
    "ɲ": {"voiced": True, "place": "palatal", "manner": "nasal"},
    "l": {"voiced": True, "place": "alveolar", "manner": "lateral_approximant"},
    "ʎ": {"voiced": True, "place": "palatal", "manner": "lateral_approximant"},
    "ɫ": {"voiced": True, "place": "velar", "manner": "lateral_approximant"},
    "ɾ": {"voiced": True, "place": "alveolar", "manner": "tap"},
    "r": {"voiced": True, "place": "alveolar", "manner": "trill"},  # alveolar trill
    "ɨ": {"high": True, "central": True, "round": False},  # central high vowel
    "ʁ": {"voiced": True, "place": "uvular", "manner": "fricative"},
    "j": {"voiced": True, "place": "palatal", "manner": "approximant"},
    "w": {"voiced": True, "labial": True, "place": "velar", "manner": "approximant"},
}

CONSONANT_COLUMNS = {
    "voiced",
    "labial",
    "dental",
    "alveolar",
    "post-alveolar",
    "palatal",
    "velar",
    "uvular",
    "plosive",
    "fricative",
    "nasal",
    "lateral_approximant",
    "tap",
    "approximant",
}

VOWEL_COLUMNS = {
    "high",
    "mid-high",
    "mid",
    "mid-low",
    "low",
    "front",
    "central",
    "back",
    "round",
    "nasal",
}

FEATURES = {
    "consonant": CONSONANT_COLUMNS,
    "vowel": VOWEL_COLUMNS,
}
