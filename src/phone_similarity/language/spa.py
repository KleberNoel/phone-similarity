VOWELS_SET = {"a", "e", "i", "o", "u", "ai", "au", "ei", "eu", "oi", "ou"}

PHONEME_FEATURES = {
    # Consonants
    "p": {"voiced": False, "labial": True, "manner": "plosive"},
    "b": {"voiced": True, "labial": True, "manner": "plosive"},
    "t": {"voiced": False, "place": "dental", "manner": "plosive"},
    "d": {"voiced": True, "place": "dental", "manner": "plosive"},
    "k": {"voiced": False, "place": "velar", "manner": "plosive"},
    "ɡ": {"voiced": True, "place": "velar", "manner": "plosive"},
    "f": {"voiced": False, "labial": True, "dental": True, "manner": "fricative"},
    "s": {"voiced": False, "place": "alveolar", "manner": "fricative"},
    "x": {"voiced": False, "place": "velar", "manner": "fricative"},
    "ʃ": {"voiced": False, "place": "post-alveolar", "manner": "fricative"},
    "m": {"voiced": True, "labial": True, "manner": "nasal"},
    "n": {"voiced": True, "place": "alveolar", "manner": "nasal"},
    "ɲ": {"voiced": True, "place": "palatal", "manner": "nasal"},
    "l": {"voiced": True, "place": "alveolar", "manner": "lateral_approximant"},
    "ʎ": {"voiced": True, "place": "palatal", "manner": "lateral_approximant"},
    "r": {"voiced": True, "place": "alveolar", "manner": "tap"},
    "ɾ": {"voiced": True, "place": "alveolar", "manner": "trill"},
    "j": {"voiced": True, "place": "palatal", "manner": "approximant"},
    "w": {"voiced": True, "labial": True, "place": "velar", "manner": "approximant"},
    "tʃ": {"voiced": False, "place": "post-alveolar", "manner": "affricate"},
    # Vowels
    "i": {"high": True, "front": True, "round": False},
    "u": {"high": True, "back": True, "round": True},
    "e": {"mid": True, "front": True, "round": False},
    "o": {"mid": True, "back": True, "round": True},
    "a": {"low": True, "central": True, "round": False},
    # Diphthongs
    "ai": {"diphthong": True, "start": "a", "end": "i"},
    "au": {"diphthong": True, "start": "a", "end": "u"},
    "ei": {"diphthong": True, "start": "e", "end": "i"},
    "eu": {"diphthong": True, "start": "e", "end": "u"},
    "oi": {"diphthong": True, "start": "o", "end": "i"},
    "ou": {"diphthong": True, "start": "o", "end": "u"},
    "z": {"voiced": True, "place": "alveolar", "manner": "fricative"},
    "ɣ": {"voiced": True, "place": "velar", "manner": "fricative"},
    "ŋ": {"voiced": True, "place": "velar", "manner": "nasal"},
    "ð": {"voiced": True, "place": "dental", "manner": "fricative"},
    "β": {"voiced": True, "labial": True, "manner": "fricative"},
    "ʝ": {"voiced": True, "place": "palatal", "manner": "fricative"},
    "ˈ": {"marker": True},
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
    "tap",
    "trill",
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
}

FEATURES = {"consonant": CONSONANT_COLUMNS, "vowel": VOWEL_COLUMNS}
