VOWELS_SET = {"a", "e", "i", "o", "u"}

PHONEME_FEATURES = {
    # Vowels
    "a": {"low": True, "central": True, "round": False},
    "e": {"mid": True, "front": True, "round": False},
    "i": {"high": True, "front": True, "round": False},
    "o": {"mid": True, "back": True, "round": True},
    "u": {"high": True, "back": True, "round": True},
    "̯": {"marker": True},
    # Consonants
    "p": {"voiced": False, "labial": True, "manner": "plosive"},
    "b": {"voiced": True, "labial": True, "manner": "plosive"},
    "t": {"voiced": False, "place": "dental", "manner": "plosive"},
    "d": {"voiced": True, "place": "dental", "manner": "plosive"},
    "k": {"voiced": False, "place": "velar", "manner": "plosive"},
    "ɡ": {"voiced": True, "place": "velar", "manner": "plosive"},
    "c": {"voiced": False, "place": "palatal", "manner": "plosive"},
    "ɟ": {"voiced": True, "place": "palatal", "manner": "plosive"},
    "f": {"voiced": False, "labial": True, "dental": True, "manner": "fricative"},
    "v": {"voiced": True, "labial": True, "dental": True, "manner": "fricative"},
    "θ": {"voiced": False, "place": "dental", "manner": "fricative"},
    "ð": {"voiced": True, "place": "dental", "manner": "fricative"},
    "s": {"voiced": False, "place": "alveolar", "manner": "fricative"},
    "z": {"voiced": True, "place": "alveolar", "manner": "fricative"},
    "x": {"voiced": False, "place": "velar", "manner": "fricative"},
    "ɣ": {"voiced": True, "place": "velar", "manner": "fricative"},
    "ç": {"voiced": False, "place": "palatal", "manner": "fricative"},
    "ʝ": {"voiced": True, "place": "palatal", "manner": "fricative"},
    "m": {"voiced": True, "labial": True, "manner": "nasal"},
    "ɱ": {"voiced": True, "labial": True, "dental": True, "manner": "nasal"},
    "n": {"voiced": True, "place": "alveolar", "manner": "nasal"},
    "ɲ": {"voiced": True, "place": "palatal", "manner": "nasal"},
    "ŋ": {"voiced": True, "place": "velar", "manner": "nasal"},
    "l": {"voiced": True, "place": "alveolar", "manner": "lateral_approximant"},
    "ʎ": {"voiced": True, "place": "palatal", "manner": "lateral_approximant"},
    "r": {"voiced": True, "place": "alveolar", "manner": "trill"},
    "ɾ": {"voiced": True, "place": "alveolar", "manner": "tap"},
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
    "tap",
    "approximant",
    "affricate",
    "marker",
}

VOWEL_COLUMNS = {
    "high",
    "mid",
    "low",
    "front",
    "central",
    "back",
    "round",
    "diphthong",
    "start",
    "end",
    "marker",
}

FEATURES = {"consonant": CONSONANT_COLUMNS, "vowel": VOWEL_COLUMNS}
