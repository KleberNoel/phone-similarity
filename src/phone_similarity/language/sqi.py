VOWELS_SET = {"a", "e", "i", "u", "y", "ə", "ɛ", "ɔ", "ʌ", "ɪ", "ɑ"}

PHONEME_FEATURES = {
    # Vowels
    "a": {"low": True, "central": True, "round": False},
    "e": {"mid": True, "front": True, "round": False},
    "i": {"high": True, "front": True, "round": False},
    "u": {"high": True, "back": True, "round": True},
    "y": {"high": True, "front": True, "round": True},
    "ə": {"mid": True, "central": True, "round": False},
    "ɛ": {"mid-low": True, "front": True, "round": False},
    "ɔ": {"mid-low": True, "back": True, "round": True},
    "ʌ": {"mid-low": True, "back": True, "round": False},
    "ɪ": {"high": True, "front": True, "near-high": True, "near-front": True, "round": False},
    "ɑ": {"low": True, "back": True, "round": False},
    # Consonants
    "p": {"voiced": False, "labial": True, "manner": "plosive"},
    "b": {"voiced": True, "labial": True, "manner": "plosive"},
    "t": {"voiced": False, "place": "dental", "manner": "plosive"},
    "d": {"voiced": True, "place": "dental", "manner": "plosive"},
    "k": {"voiced": False, "place": "velar", "manner": "plosive"},
    "ɡ": {"voiced": True, "place": "velar", "manner": "plosive"},
    "f": {"voiced": False, "labial": True, "dental": True, "manner": "fricative"},
    "v": {"voiced": True, "labial": True, "dental": True, "manner": "fricative"},
    "s": {"voiced": False, "place": "alveolar", "manner": "fricative"},
    "z": {"voiced": True, "place": "alveolar", "manner": "fricative"},
    "ʃ": {"voiced": False, "place": "post-alveolar", "manner": "fricative"},
    "ʒ": {"voiced": True, "place": "post-alveolar", "manner": "fricative"},
    "h": {"voiced": False, "place": "glottal", "manner": "fricative"},
    "m": {"voiced": True, "labial": True, "manner": "nasal"},
    "n": {"voiced": True, "place": "alveolar", "manner": "nasal"},
    "ɲ": {"voiced": True, "place": "palatal", "manner": "nasal"},
    "l": {"voiced": True, "place": "alveolar", "manner": "lateral_approximant"},
    "ɫ": {"voiced": True, "place": "alveolar", "manner": "lateral_approximant", "velarized": True},
    "r": {"voiced": True, "place": "alveolar", "manner": "trill"},
    "j": {"voiced": True, "place": "palatal", "manner": "approximant"},
    "w": {"voiced": True, "labial": True, "place": "velar", "manner": "approximant"},
    "θ": {"voiced": False, "place": "dental", "manner": "fricative"},
    "ð": {"voiced": True, "place": "dental", "manner": "fricative"},
    "ɕ": {"voiced": False, "place": "alveolo-palatal", "manner": "fricative"},
    "ʑ": {"voiced": True, "place": "alveolo-palatal", "manner": "fricative"},
    # Modifiers
    "ˈ": {"marker": True},
    "ˌ": {"marker": True},
    "ː": {"long": True},
    "͡": {"affricate_tie": True},
}

CONSONANT_COLUMNS = {
    "voiced",
    "labial",
    "dental",
    "alveolar",
    "post-alveolar",
    "alveolo-palatal",
    "palatal",
    "velar",
    "glottal",
    "plosive",
    "fricative",
    "nasal",
    "lateral_approximant",
    "trill",
    "approximant",
    "velarized",
}

VOWEL_COLUMNS = {
    "high",
    "near-high",
    "mid-high",
    "mid",
    "mid-low",
    "low",
    "front",
    "near-front",
    "central",
    "near-back",
    "back",
    "round",
}

MODIFIER_COLUMNS = {
    "long",
    "affricate_tie",
    "marker",
}

FEATURES = {"consonant": CONSONANT_COLUMNS, "vowel": VOWEL_COLUMNS, "modifier": MODIFIER_COLUMNS}
