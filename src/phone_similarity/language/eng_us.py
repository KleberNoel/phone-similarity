VOWELS_SET = {
    "i",
    "ɪ",
    "e",
    "ɛ",
    "æ",
    "a",
    "ɑ",
    "ɔ",
    "o",
    "ʊ",
    "u",
    "ʌ",
    "ə",
    "ɚ",
    "ɝ",
    "aɪ",
    "aʊ",
    "ɔɪ",
}

PHONEME_FEATURES = {
    # Vowels (monophthongs)
    "i": {"high": True, "front": True, "round": False, "tense": True},
    "ɪ": {"high": True, "front": True, "round": False, "tense": False},
    "e": {"mid": True, "front": True, "round": False, "tense": True},
    "ɛ": {"mid-low": True, "front": True, "round": False, "tense": False},
    "æ": {"low": True, "front": True, "round": False, "tense": False},
    "a": {"low": True, "central": True, "round": False, "tense": False},
    "ɑ": {"low": True, "back": True, "round": False, "tense": True},
    "ɔ": {"mid-low": True, "back": True, "round": True, "tense": False},
    "o": {"mid": True, "back": True, "round": True, "tense": True},
    "ʊ": {"high": True, "back": True, "round": True, "tense": False},
    "u": {"high": True, "back": True, "round": True, "tense": True},
    "ʌ": {"mid-low": True, "back": True, "round": False, "tense": False},  # strut vowel
    # often transcribed as ə in unstressed syllables
    "ə": {"mid": True, "central": True, "round": False, "tense": False},  # schwa
    "ɚ": {
        "mid": True,
        "central": True,
        "round": False,
        "tense": False,
        "rhotacized": True,
    },  # r-colored schwa
    "ɝ": {
        "mid": True,
        "central": True,
        "round": False,
        "tense": True,
        "rhotacized": True,
    },  # stressed r-colored vowel
    # Vowels (diphthongs)
    "aɪ": {"diphthong": True, "start": "a", "end": "ɪ"},
    "aʊ": {"diphthong": True, "start": "a", "end": "ʊ"},
    "ɔɪ": {"diphthong": True, "start": "ɔ", "end": "ɪ"},
    # Consonants
    "p": {"voiced": False, "labial": True, "manner": "plosive"},
    "b": {"voiced": True, "labial": True, "manner": "plosive"},
    "t": {"voiced": False, "place": "alveolar", "manner": "plosive"},
    "d": {"voiced": True, "place": "alveolar", "manner": "plosive"},
    "k": {"voiced": False, "place": "velar", "manner": "plosive"},
    "ɡ": {"voiced": True, "place": "velar", "manner": "plosive"},
    "f": {"voiced": False, "labial": True, "dental": True, "manner": "fricative"},
    "v": {"voiced": True, "labial": True, "dental": True, "manner": "fricative"},
    "θ": {"voiced": False, "place": "dental", "manner": "fricative"},
    "ð": {"voiced": True, "place": "dental", "manner": "fricative"},
    "s": {"voiced": False, "place": "alveolar", "manner": "fricative"},
    "z": {"voiced": True, "place": "alveolar", "manner": "fricative"},
    "ʃ": {"voiced": False, "place": "post-alveolar", "manner": "fricative"},
    "ʒ": {"voiced": True, "place": "post-alveolar", "manner": "fricative"},
    "h": {"voiced": False, "place": "glottal", "manner": "fricative"},
    "tʃ": {"voiced": False, "place": "post-alveolar", "manner": "affricate"},
    "dʒ": {"voiced": True, "place": "post-alveolar", "manner": "affricate"},
    "m": {"voiced": True, "labial": True, "manner": "nasal"},
    "n": {"voiced": True, "place": "alveolar", "manner": "nasal"},
    "ŋ": {"voiced": True, "place": "velar", "manner": "nasal"},
    "ɫ": {"voiced": True, "place": "alveolar", "manner": "lateral_approximant", "velarized": True},
    "ɹ": {"voiced": True, "place": "alveolar", "manner": "approximant"},
    "j": {"voiced": True, "place": "palatal", "manner": "approximant"},
    "w": {"voiced": True, "labial": True, "place": "velar", "manner": "approximant"},
    # Modifiers
    "ˈ": {"marker": True},
    "ˌ": {"marker": True},
}

CONSONANT_COLUMNS = {
    "voiced",
    "labial",
    "dental",
    "alveolar",
    "post-alveolar",
    "palatal",
    "velar",
    "glottal",
    "plosive",
    "fricative",
    "nasal",
    "lateral_approximant",
    "approximant",
    "affricate",
    "velarized",
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
    "tense",
    "rhotacized",
    "diphthong",
    "start",
    "end",
}

MODIFIER_COLUMNS = {
    "marker",
    "long",
    "nasalized",
}

FEATURES = {"consonant": CONSONANT_COLUMNS, "vowel": VOWEL_COLUMNS, "modifier": MODIFIER_COLUMNS}
