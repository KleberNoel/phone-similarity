VOWELS_SET = {
    "a",
    "a͡",
    "e",
    "e͡",
    "i",
    "o",
    "o͡",
    "u",
    "y",
    "ɪ",
    "ʊ",
}

PHONEME_FEATURES = {
    # Vowels
    "a": {"front": True, "low": True, "round": False},
    "a͡": {"front": True, "low": True, "round": False},
    "e": {"front": True, "mid-high": True, "round": False},
    "e͡": {"front": True, "mid-high": True, "round": False},
    "i": {"front": True, "high": True, "round": False},
    "o": {"back": True, "mid-high": True, "round": True},
    "o͡": {"back": True, "mid-high": True, "round": True},
    "u": {"back": True, "high": True, "round": True},
    "y": {"front": True, "high": True, "round": True},
    "ɪ": {"front": True, "near-high": True, "round": False},
    "ʊ": {"back": True, "near-high": True, "round": True},
    # Consonants
    "b": {"labial": True, "manner": "plosive", "voiced": True},
    "c": {"manner": "plosive", "place": "palatal", "voiced": False},
    "d": {"manner": "plosive", "place": "alveolar", "voiced": True},
    "f": {"dental": True, "labial": True, "manner": "fricative", "voiced": False},
    "h": {"manner": "fricative", "place": "glottal", "voiced": False},
    "j": {"manner": "approximant", "place": "palatal", "voiced": True},
    "k": {"manner": "plosive", "place": "velar", "voiced": False},
    "l": {"manner": "lateral_approximant", "place": "alveolar", "voiced": True},
    "m": {"labial": True, "manner": "nasal", "voiced": True},
    "n": {"manner": "nasal", "place": "alveolar", "voiced": True},
    "p": {"labial": True, "manner": "plosive", "voiced": False},
    "r": {"manner": "trill", "place": "alveolar", "voiced": True},
    "s̺": {"manner": "fricative", "place": "alveolar", "voiced": False},
    "s̻": {"manner": "fricative", "place": "alveolar", "voiced": False},
    "t": {"manner": "plosive", "place": "alveolar", "voiced": False},
    "t͡": {"manner": "plosive", "place": "alveolar", "voiced": False},
    "x": {"manner": "fricative", "place": "velar", "voiced": False},
    "ð": {"manner": "fricative", "place": "dental", "voiced": True},
    "ɟ": {"manner": "plosive", "place": "palatal", "voiced": True},
    "ɡ": {"manner": "plosive", "place": "velar", "voiced": True},
    "ɣ": {"manner": "fricative", "place": "velar", "voiced": True},
    "ɲ": {"manner": "nasal", "place": "palatal", "voiced": True},
    "ɾ": {"manner": "tap", "place": "alveolar", "voiced": True},
    "ʃ": {"manner": "fricative", "place": "post-alveolar", "voiced": False},
    "ʎ": {"manner": "lateral_approximant", "place": "palatal", "voiced": True},
    "β": {"labial": True, "manner": "fricative", "voiced": True},
    "θ": {"manner": "fricative", "place": "dental", "voiced": False},
    # Modifiers
    "ˈ": {"marker": True},
    "ˌ": {"marker": True},
    "s": {"manner": "fricative", "place": "alveolar", "voiced": False},
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
    "tap",
    "trill",
    "velar",
    "voiced",
}

VOWEL_COLUMNS = {
    "back",
    "front",
    "high",
    "low",
    "mid-high",
    "near-high",
    "round",
}

MODIFIER_COLUMNS = {
    "marker",
}

FEATURES = {"consonant": CONSONANT_COLUMNS, "vowel": VOWEL_COLUMNS, "modifier": MODIFIER_COLUMNS}
