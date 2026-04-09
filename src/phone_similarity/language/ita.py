VOWELS_SET = {"a", "e", "ɛ", "i", "o", "ɔ", "u"}

PHONEME_FEATURES = {
    # Vowels
    "a": {"low": True, "central": True, "round": False},
    "e": {"mid-high": True, "front": True, "round": False},
    "ɛ": {"mid-low": True, "front": True, "round": False},
    "i": {"high": True, "front": True, "round": False},
    "o": {"mid-high": True, "back": True, "round": True},
    "ɔ": {"mid-low": True, "back": True, "round": True},
    "u": {"high": True, "back": True, "round": True},
    # Consonants
    "b": {"voiced": True, "labial": True, "manner": "plosive"},
    "d": {"voiced": True, "place": "dental", "manner": "plosive"},
    "f": {"voiced": False, "labial": True, "dental": True, "manner": "fricative"},
    "ɡ": {"voiced": True, "place": "velar", "manner": "plosive"},
    "j": {"voiced": True, "place": "palatal", "manner": "approximant"},
    "k": {"voiced": False, "place": "velar", "manner": "plosive"},
    "l": {"voiced": True, "place": "alveolar", "manner": "lateral_approximant"},
    "ʎ": {"voiced": True, "place": "palatal", "manner": "lateral_approximant"},
    "m": {"voiced": True, "labial": True, "manner": "nasal"},
    "n": {"voiced": True, "place": "alveolar", "manner": "nasal"},
    "ɲ": {"voiced": True, "place": "palatal", "manner": "nasal"},
    "p": {"voiced": False, "labial": True, "manner": "plosive"},
    "r": {"voiced": True, "place": "alveolar", "manner": "trill"},
    "s": {"voiced": False, "place": "alveolar", "manner": "fricative"},
    "ʃ": {"voiced": False, "place": "post-alveolar", "manner": "fricative"},
    "t": {"voiced": False, "place": "dental", "manner": "plosive"},
    "ts": {"voiced": False, "place": "alveolar", "manner": "affricate"},
    "dz": {"voiced": True, "place": "alveolar", "manner": "affricate"},
    "tʃ": {"voiced": False, "place": "post-alveolar", "manner": "affricate"},
    "dʒ": {"voiced": True, "place": "post-alveolar", "manner": "affricate"},
    "v": {"voiced": True, "labial": True, "dental": True, "manner": "fricative"},
    "w": {"voiced": True, "labial": True, "place": "velar", "manner": "approximant"},
    "z": {"voiced": True, "place": "alveolar", "manner": "fricative"},
    "ʒ": {"voiced": True, "place": "post-alveolar", "manner": "fricative"},
    "͡": {"affricate_tie": True},
}

CONSONANT_COLUMNS = {
    "voiced",
    "labial",
    "dental",
    "alveolar",
    "post-alveolar",
    "palatal",
    "velar",
    "plosive",
    "fricative",
    "nasal",
    "lateral_approximant",
    "trill",
    "approximant",
    "affricate",
    "affricate_tie",
}

VOWEL_COLUMNS = {
    "high",
    "mid-high",
    "mid-low",
    "low",
    "front",
    "central",
    "back",
    "round",
}

FEATURES = {"consonant": CONSONANT_COLUMNS, "vowel": VOWEL_COLUMNS}
