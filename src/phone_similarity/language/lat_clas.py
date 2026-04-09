VOWELS_SET = {
    "a",
    "e",
    "i",
    "o",
    "u",
    "y",
    "ɛ",
    "ɔ",
    "aː",
    "eː",
    "iː",
    "oː",
    "uː",
    "yː",
    "ɛː",
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
    "y": {"high": True, "front": True, "round": True},
    "iː": {"high": True, "front": True, "round": False, "long": True},
    "eː": {"mid-high": True, "front": True, "round": False, "long": True},
    "ɛː": {"mid-low": True, "front": True, "round": False, "long": True},
    "aː": {"low": True, "front": True, "round": False, "long": True},
    "uː": {"high": True, "back": True, "round": True, "long": True},
    "yː": {"high": True, "front": True, "round": True, "long": True},
    "ɪ": {"high": True, "front": True, "round": False, "tense": False},
    "ʊ": {"high": True, "back": True, "round": True, "tense": False},
    "ʏ": {"high": True, "front": True, "round": True, "tense": False},
    # Consonants
    "p": {"voiced": False, "labial": True, "manner": "plosive"},
    "b": {"voiced": True, "labial": True, "manner": "plosive"},
    "t": {"voiced": False, "place": "dental", "manner": "plosive"},
    "d": {"voiced": True, "place": "dental", "manner": "plosive"},
    "k": {"voiced": False, "place": "velar", "manner": "plosive"},
    "ɡ": {"voiced": True, "place": "velar", "manner": "plosive"},
    "ʃ": {"voiced": False, "place": "post-alveolar", "manner": "fricative"},
    "f": {"voiced": False, "labial": True, "dental": True, "manner": "fricative"},
    "s": {"voiced": False, "place": "alveolar", "manner": "fricative"},
    "z": {"voiced": True, "place": "alveolar", "manner": "fricative"},
    "l": {"voiced": True, "place": "alveolar", "manner": "lateral_approximant"},
    "m": {"voiced": True, "labial": True, "manner": "nasal"},
    "n": {"voiced": True, "place": "alveolar", "manner": "nasal"},
    "ŋ": {"voiced": True, "place": "velar", "manner": "nasal"},
    "j": {"voiced": True, "place": "palatal", "manner": "approximant"},
    "w": {"voiced": True, "labial": True, "place": "velar", "manner": "approximant"},
    "ɥ": {"voiced": True, "labial": True, "place": "palatal", "manner": "approximant"},
    "r": {"voiced": True, "place": "alveolar", "manner": "trill"},
    "ɾ": {"voiced": True, "place": "alveolar", "manner": "tap"},
    "h": {"voiced": False, "place": "glottal", "manner": "plosive"},  # Can be a fricative as well
    # Diphthongs
    "ae̯": {"diphthong": True, "start": "a", "end": "e"},
    "oe̯": {"diphthong": True, "start": "o", "end": "e"},
    "au̯": {"diphthong": True, "start": "a", "end": "u"},
    "eu̯": {"diphthong": True, "start": "e", "end": "u"},
    "ei̯": {"diphthong": True, "start": "e", "end": "i"},
    "ui̯": {"diphthong": True, "start": "u", "end": "i"},
    # Modifiers
    "ː": {"long": True},
    "̯": {"diphthong": True},
}

CONSONANT_COLUMNS = {
    "voiced",
    "labial",
    "dental",
    "alveolar",
    "palatal",
    "velar",
    "glottal",
    "plosive",
    "fricative",
    "nasal",
    "lateral_approximant",
    "trill",
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
    "long",
    "diphthong",
    "start",
    "end",
}

MODIFIER_COLUMNS = {
    "marker",
    "long",
    "diphthong",
}

FEATURES = {"consonant": CONSONANT_COLUMNS, "vowel": VOWEL_COLUMNS, "modifier": MODIFIER_COLUMNS}
