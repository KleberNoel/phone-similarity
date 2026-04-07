VOWELS_SET = {
    "a",
    "aː",
    "e",
    "eː",
    "i",
    "iː",
    "o",
    "u",
    "ɔ",
    "ɔː",
    "ɪ",
    "ʊ",
}

PHONEME_FEATURES = {
    # Vowels
    "a": {"front": True, "low": True, "round": False},
    "aː": {"front": True, "long": True, "low": True, "round": False},
    "e": {"front": True, "mid-high": True, "round": False},
    "eː": {"front": True, "long": True, "mid-high": True, "round": False},
    "i": {"front": True, "high": True, "round": False},
    "iː": {"front": True, "high": True, "long": True, "round": False},
    "o": {"back": True, "mid-high": True, "round": True},
    "u": {"back": True, "high": True, "round": True},
    "ɔ": {"back": True, "mid-low": True, "round": True},
    "ɔː": {"back": True, "long": True, "mid-low": True, "round": True},
    "ɪ": {"front": True, "near-high": True, "round": False},
    "ʊ": {"back": True, "near-high": True, "round": True},
    # Consonants
    "f": {"dental": True, "labial": True, "manner": "fricative", "voiced": False},
    "h": {"manner": "fricative", "place": "glottal", "voiced": False},
    "k": {"manner": "plosive", "place": "velar", "voiced": False},
    "m": {"labial": True, "manner": "nasal", "voiced": True},
    "n": {"manner": "nasal", "place": "alveolar", "voiced": True},
    "p": {"labial": True, "manner": "plosive", "voiced": False},
    "r": {"manner": "trill", "place": "alveolar", "voiced": True},
    "t": {"manner": "plosive", "place": "alveolar", "voiced": False},
    "w": {"labial": True, "manner": "approximant", "place": "velar", "voiced": True},
    "ŋ": {"manner": "nasal", "place": "velar", "voiced": True},
    "ɹ": {"manner": "approximant", "place": "alveolar", "voiced": True},
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
    "nasal",
    "plosive",
    "trill",
    "velar",
    "voiced",
}

VOWEL_COLUMNS = {
    "back",
    "front",
    "high",
    "long",
    "low",
    "mid-high",
    "mid-low",
    "near-high",
    "round",
}

MODIFIER_COLUMNS = {
    "marker",
}

FEATURES = {"consonant": CONSONANT_COLUMNS, "vowel": VOWEL_COLUMNS, "modifier": MODIFIER_COLUMNS}
