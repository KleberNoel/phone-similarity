VOWELS_SET = {
    "a",
    "a͡",
    "e",
    "i",
    "o",
    "u",
    "ɔ",
    "ə",
    "ɛ",
    "ɪ",
    "ʊ",
}

PHONEME_FEATURES = {
    # Vowels
    "a": {"front": True, "low": True, "round": False},
    "a͡": {"front": True, "low": True, "round": False},
    "e": {"front": True, "mid-high": True, "round": False},
    "i": {"front": True, "high": True, "round": False},
    "o": {"back": True, "mid-high": True, "round": True},
    "u": {"back": True, "high": True, "round": True},
    "ɔ": {"back": True, "mid-low": True, "round": True},
    "ə": {"central": True, "mid": True, "round": False},
    "ɛ": {"front": True, "mid-low": True, "round": False},
    "ɪ": {"front": True, "near-high": True, "round": False},
    "ʊ": {"back": True, "near-high": True, "round": True},
    # Consonants
    "b": {"labial": True, "manner": "plosive", "voiced": True},
    "d": {"manner": "plosive", "place": "alveolar", "voiced": True},
    "d͡": {"manner": "plosive", "place": "alveolar", "voiced": True},
    "f": {"dental": True, "labial": True, "manner": "fricative", "voiced": False},
    "h": {"manner": "fricative", "place": "glottal", "voiced": False},
    "j": {"manner": "approximant", "place": "palatal", "voiced": True},
    "k": {"manner": "plosive", "place": "velar", "voiced": False},
    "l": {"manner": "lateral_approximant", "place": "alveolar", "voiced": True},
    "m": {"labial": True, "manner": "nasal", "voiced": True},
    "n": {"manner": "nasal", "place": "alveolar", "voiced": True},
    "p": {"labial": True, "manner": "plosive", "voiced": False},
    "r": {"manner": "trill", "place": "alveolar", "voiced": True},
    "s": {"manner": "fricative", "place": "alveolar", "voiced": False},
    "t": {"manner": "plosive", "place": "alveolar", "voiced": False},
    "t͡": {"manner": "plosive", "place": "alveolar", "voiced": False},
    "v": {"dental": True, "labial": True, "manner": "fricative", "voiced": True},
    "w": {"labial": True, "manner": "approximant", "place": "velar", "voiced": True},
    "x": {"manner": "fricative", "place": "velar", "voiced": False},
    "z": {"manner": "fricative", "place": "alveolar", "voiced": True},
    "ç": {"manner": "fricative", "place": "palatal", "voiced": False},
    "ŋ": {"manner": "nasal", "place": "velar", "voiced": True},
    "ɡ": {"manner": "plosive", "place": "velar", "voiced": True},
    "ɲ": {"manner": "nasal", "place": "palatal", "voiced": True},
    "ʃ": {"manner": "fricative", "place": "post-alveolar", "voiced": False},
    "ʒ": {"manner": "fricative", "place": "post-alveolar", "voiced": True},
    "ʔ": {"manner": "plosive", "place": "glottal", "voiced": False},
    "χ": {"manner": "fricative", "place": "uvular", "voiced": False},
    # Modifiers
    "ˈ": {"marker": True},
    "ˌ": {"marker": True},
    "c": {"manner": "plosive", "place": "palatal", "voiced": False},
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
    "trill",
    "uvular",
    "velar",
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
    "mid-low",
    "near-high",
    "round",
}

MODIFIER_COLUMNS = {
    "marker",
}

FEATURES = {"consonant": CONSONANT_COLUMNS, "vowel": VOWEL_COLUMNS, "modifier": MODIFIER_COLUMNS}
