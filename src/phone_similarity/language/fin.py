VOWELS_SET = {
    "a",
    "e",
    "i",
    "o",
    "u",
    "y",
    "å",
    "æ",
    "è",
    "é",
    "ø",
    "û",
    "ɑ",
}

PHONEME_FEATURES = {
    # Vowels
    "e": {"front": True, "mid-high": True, "round": False},
    "i": {"front": True, "high": True, "round": False},
    "o": {"back": True, "mid-high": True, "round": True},
    "u": {"back": True, "high": True, "round": True},
    "y": {"front": True, "high": True, "round": True},
    "æ": {"front": True, "near-low": True, "round": False},
    "ø": {"front": True, "mid-high": True, "round": True},
    "ɑ": {"back": True, "low": True, "round": False},
    # Consonants
    "b": {"labial": True, "manner": "plosive", "voiced": True},
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
    "å": {"front": True, "low": True, "round": False},
    "è": {"front": True, "mid-high": True, "round": False},
    "é": {"front": True, "mid-high": True, "round": False},
    "û": {"back": True, "high": True, "round": True},
    "ŋ": {"manner": "nasal", "place": "velar", "voiced": True},
    "š": {"manner": "fricative", "place": "alveolar", "voiced": False},
    "ž": {"manner": "fricative", "place": "alveolar", "voiced": True},
    "ʒ": {"manner": "fricative", "place": "post-alveolar", "voiced": True},
    # Modifiers
    "ˈ": {"marker": True},
    "ˌ": {"marker": True},
    "a": {"front": True, "low": True, "round": False},
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
    "near-low",
    "round",
}

MODIFIER_COLUMNS = {
    "marker",
}

FEATURES = {"consonant": CONSONANT_COLUMNS, "vowel": VOWEL_COLUMNS, "modifier": MODIFIER_COLUMNS}
