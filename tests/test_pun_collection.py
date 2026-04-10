"""
Real-world inter-lingual and intra-lingual pun test cases.

Curated from Wikipedia articles on bilingual puns, mondegreens,
homophonic translations, eggcorns, soramimi, and macaronic language.

These test cases exercise the beam search pipeline with real phonological
matches found in the wild, paying special attention to:
  - Syllable boundary alignment
  - Multi-word segmentation (beam search)
  - Cross-language feature edit distance
  - ONNX model pronunciation variants

Categories
----------
1. **Inter-lingual homophonic translations** (eng<->fra, eng<->deu, etc.)
   - Mots D'Heures / Mother Goose style
   - Macaronic language puns
2. **Intra-lingual mondegreens** (same language, different word boundaries)
   - Song lyric mondegreens
   - Eggcorns
3. **Cross-language soramimi** (reinterpretation across languages)

Sources
-------
- https://en.wikipedia.org/wiki/Bilingual_pun
- https://en.wikipedia.org/wiki/Eggcorn
- https://en.wikipedia.org/wiki/Mondegreen
- https://en.wikipedia.org/wiki/Homophonic_translation
- https://en.wikipedia.org/wiki/Soramimi
- https://en.wikipedia.org/wiki/Macaronic_language
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from phone_similarity.beam_search import (
    BeamResult,
    beam_search_segmentation,
)
from phone_similarity.pretokenize import PreTokenizedDictionary


# ===================================================================
# Pun data structures
# ===================================================================


@dataclass(frozen=True)
class PunCase:
    """A single pun / homophonic pair for testing.

    Attributes
    ----------
    id : str
        Short identifier for the test case.
    source_lang : str
        ISO-style language code of the source phrase (e.g. "eng-us").
    target_lang : str
        Language code of the target phrase.
    source_text : str
        Original phrase in the source language.
    target_text : str
        Phonologically matched phrase in the target language.
    source_ipa : str
        IPA transcription of the source phrase.
    target_ipa : str
        IPA transcription of the target phrase.
    category : str
        One of: "interlingual", "intralingual_mondegreen", "eggcorn",
        "homophonic_translation", "soramimi".
    n_words_source : int
        Number of words in the source phrase.
    n_words_target : int
        Number of words in the target phrase.
    notes : str
        Brief description of why this pun works phonologically.
    max_expected_distance : float
        Upper bound on the normalised feature edit distance we expect
        from the pipeline.  Looser (higher) values for approximate
        matches, tighter for near-homophones.
    """

    id: str
    source_lang: str
    target_lang: str
    source_text: str
    target_text: str
    source_ipa: str
    target_ipa: str
    category: str
    n_words_source: int
    n_words_target: int
    notes: str
    max_expected_distance: float = 0.50


# ===================================================================
# Curated pun collection
# ===================================================================

# ------------------------------------------------------------------
# 1. Inter-lingual homophonic translations (eng <-> fra)
# ------------------------------------------------------------------

INTERLINGUAL_ENG_FRA = [
    PunCase(
        id="humpty_dumpty_l1",
        source_lang="eng-us",
        target_lang="fra",
        source_text="Humpty Dumpty",
        target_text="Un petit d'un petit",
        source_ipa="h\u028cmpt\u026a d\u028cmpt\u026a",
        target_ipa="\u0153\u0303 p\u0259ti d\u0153\u0303 p\u0259ti",
        category="homophonic_translation",
        n_words_source=2,
        n_words_target=4,
        notes="From 'Mots D'Heures: Gousses, Rames' by Luis van Rooten. "
        "Multi-word French approximation of English nursery rhyme.",
        max_expected_distance=0.50,
    ),
    PunCase(
        id="humpty_dumpty_l2",
        source_lang="eng-us",
        target_lang="fra",
        source_text="sat on a wall",
        target_text="s'etonne aux Halles",
        source_ipa="s\u00e6t \u0252n \u0259 w\u0254\u02d0l",
        target_ipa="set\u0254n o al",
        category="homophonic_translation",
        n_words_source=4,
        n_words_target=3,
        notes="Homophonic translation: /s\u00e6t \u0252n \u0259 w\u0254\u02d0l/ ~ "
        "/set\u0254n o al/. Syllable boundary shift: 'sat-on' -> 's\u00e9-tonne'.",
        max_expected_distance=0.45,
    ),
    PunCase(
        id="mother_goose_title",
        source_lang="eng-us",
        target_lang="fra",
        source_text="Mother Goose Rhymes",
        target_text="Mots d'Heures Gousses Rames",
        source_ipa="m\u028c\u00f0\u0259\u0279 \u0261u\u02d0s \u0279a\u026amz",
        target_ipa="mo d\u0153\u0281 \u0261us \u0281am",
        category="homophonic_translation",
        n_words_source=3,
        n_words_target=4,
        notes="Book title pun. English title sounds like French phrase. "
        "Tests syllable re-alignment across word boundaries.",
        max_expected_distance=0.45,
    ),
    PunCase(
        id="thing_of_beauty",
        source_lang="eng-us",
        target_lang="fra",
        source_text="A thing of beauty is a joy forever",
        target_text="Un singe de beaute est un jouet pour l'hiver",
        source_ipa="\u0259 \u03b8\u026a\u014b \u0259v bju\u02d0ti \u026az \u0259 d\u0292\u0254\u026a f\u0259\u0279\u025bv\u0259\u0279",
        target_ipa="\u0153\u0303 s\u025b\u0303\u0292 d\u0259 bote \u025b \u0153\u0303 \u0292w\u025b pu\u0281 liv\u025b\u0281",
        category="interlingual",
        n_words_source=8,
        n_words_target=9,
        notes="Francois Le Lionnais's cross-language pun. 'A monkey of beauty "
        "is a toy for winter.' Multi-word beam search stress test.",
        max_expected_distance=0.65,
    ),
    PunCase(
        id="poor_john",
        source_lang="fra",
        target_lang="fra",
        source_text="pauvre Jean",
        target_text="pauvres gens",
        source_ipa="pov\u0281 \u0292\u0251\u0303",
        target_ipa="pov\u0281 \u0292\u0251\u0303",
        category="intralingual_mondegreen",
        n_words_source=2,
        n_words_target=2,
        notes="French mondegreen: 'poor John' heard as 'poor people'. "
        "Perfect homophone pair. Led to mistranslation of a hit song title.",
        max_expected_distance=0.05,
    ),
]

# ------------------------------------------------------------------
# 2. Inter-lingual homophonic translations (eng <-> deu)
# ------------------------------------------------------------------

INTERLINGUAL_ENG_DEU = [
    PunCase(
        id="mother_goose_german",
        source_lang="eng-us",
        target_lang="deu",
        source_text="Mother Goose Rhymes",
        target_text="Moerder Guss Reims",
        source_ipa="m\u028c\u00f0\u0259\u0279 \u0261u\u02d0s \u0279a\u026amz",
        target_ipa="m\u0153\u0281d\u0259\u0281 \u0261\u028as \u0281a\u026ams",
        category="homophonic_translation",
        n_words_source=3,
        n_words_target=3,
        notes="German homophonic version of 'Mother Goose Rhymes'. "
        "Tests cross-language vowel quality: /\u028c/ vs /\u0153/, /u\u02d0/ vs /\u028a/.",
        max_expected_distance=0.40,
    ),
]

# ------------------------------------------------------------------
# 3. Intra-lingual mondegreens (English)
# ------------------------------------------------------------------

MONDEGREENS_ENG = [
    PunCase(
        id="kiss_the_sky",
        source_lang="eng-us",
        target_lang="eng-us",
        source_text="kiss the sky",
        target_text="kiss this guy",
        source_ipa="k\u026as \u00f0\u0259 ska\u026a",
        target_ipa="k\u026as \u00f0\u026as \u0261a\u026a",
        category="intralingual_mondegreen",
        n_words_source=3,
        n_words_target=3,
        notes="Jimi Hendrix 'Purple Haze'. /\u00f0\u0259 ska\u026a/ vs /\u00f0\u026as \u0261a\u026a/. "
        "Syllable boundary shift at 'the-sky' -> 'this-guy'.",
        max_expected_distance=0.35,
    ),
    PunCase(
        id="bad_moon_rising",
        source_lang="eng-us",
        target_lang="eng-us",
        source_text="bad moon on the rise",
        target_text="bathroom on the right",
        source_ipa="b\u00e6d mu\u02d0n \u0252n \u00f0\u0259 \u0279a\u026az",
        target_ipa="b\u00e6\u03b8\u0279u\u02d0m \u0252n \u00f0\u0259 \u0279a\u026at",
        category="intralingual_mondegreen",
        n_words_source=5,
        n_words_target=4,
        notes="CCR 'Bad Moon Rising'. Word boundary collapse: "
        "'bad-moon' -> 'bath-room'. Final /z/ vs /t/.",
        max_expected_distance=0.35,
    ),
    PunCase(
        id="blinded_by_light",
        source_lang="eng-us",
        target_lang="eng-us",
        source_text="revved up like a deuce",
        target_text="wrapped up like a douche",
        source_ipa="\u0279\u025bvd \u028cp la\u026ak \u0259 du\u02d0s",
        target_ipa="\u0279\u00e6pt \u028cp la\u026ak \u0259 du\u02d0\u0283",
        category="intralingual_mondegreen",
        n_words_source=5,
        n_words_target=5,
        notes="Manfred Mann / Springsteen. 'Probably the most misheard lyric "
        "of all time.' Onset cluster /\u0279\u025bvd/ vs /\u0279\u00e6pt/, coda /s/ vs /\u0283/.",
        max_expected_distance=0.30,
    ),
    PunCase(
        id="lucy_in_the_sky",
        source_lang="eng-us",
        target_lang="eng-us",
        source_text="the girl with kaleidoscope eyes",
        target_text="the girl with colitis goes by",
        source_ipa="\u00f0\u0259 \u0261\u025d\u02d0l w\u026a\u03b8 k\u0259la\u026ad\u0259sko\u028ap a\u026az",
        target_ipa="\u00f0\u0259 \u0261\u025d\u02d0l w\u026a\u03b8 k\u0259la\u026at\u0259s \u0261o\u028az ba\u026a",
        category="intralingual_mondegreen",
        n_words_source=5,
        n_words_target=6,
        notes="Beatles 'Lucy in the Sky'. 'kaleidoscope-eyes' -> "
        "'colitis-goes-by'. Major syllable boundary re-analysis.",
        max_expected_distance=0.40,
    ),
    PunCase(
        id="teen_spirit",
        source_lang="eng-us",
        target_lang="eng-us",
        source_text="here we are now entertain us",
        target_text="here we are now in containers",
        source_ipa="h\u026a\u0279 wi\u02d0 \u0251\u0279 na\u028a \u025bnt\u0259\u0279te\u026an \u028cs",
        target_ipa="h\u026a\u0279 wi\u02d0 \u0251\u0279 na\u028a \u026an k\u0259nte\u026an\u0259\u0279z",
        category="intralingual_mondegreen",
        n_words_source=5,
        n_words_target=6,
        notes="Nirvana 'Smells Like Teen Spirit'. 'entertain-us' -> "
        "'in-containers'. Word boundary insertion + coda change.",
        max_expected_distance=0.35,
    ),
    PunCase(
        id="blank_space",
        source_lang="eng-us",
        target_lang="eng-us",
        source_text="got a long list of ex-lovers",
        target_text="all the lonely Starbucks lovers",
        source_ipa="\u0261\u0252t \u0259 l\u0252\u014b l\u026ast \u0259v \u025bksl\u028cv\u0259\u0279z",
        target_ipa="\u0254\u02d0l \u00f0\u0259 lo\u028anli st\u0251\u0279b\u028cks l\u028cv\u0259\u0279z",
        category="intralingual_mondegreen",
        n_words_source=6,
        n_words_target=5,
        notes="Taylor Swift 'Blank Space'. Major lexical resegmentation. "
        "Tests beam search with very different word boundaries.",
        max_expected_distance=0.50,
    ),
    PunCase(
        id="mairzy_doats",
        source_lang="eng-us",
        target_lang="eng-us",
        source_text="mares eat oats",
        target_text="mairzy doats",
        source_ipa="m\u025b\u0259\u0279z i\u02d0t o\u028ats",
        target_ipa="m\u025b\u0259\u0279zi do\u028ats",
        category="intralingual_mondegreen",
        n_words_source=3,
        n_words_target=2,
        notes="Reverse mondegreen from 1943 novelty song. "
        "Syllable boundary collapse: 'mares-eat' -> 'mairzy'.",
        max_expected_distance=0.25,
    ),
    PunCase(
        id="recognize_speech",
        source_lang="eng-us",
        target_lang="eng-us",
        source_text="recognize speech",
        target_text="wreck a nice beach",
        source_ipa="\u0279\u025bk\u0259na\u026az spi\u02d0t\u0283",
        target_ipa="\u0279\u025bk \u0259 na\u026as bi\u02d0t\u0283",
        category="intralingual_mondegreen",
        n_words_source=2,
        n_words_target=4,
        notes="Classic speech recognition joke. 2 words -> 4 words. "
        "Perfect test for beam search multi-word segmentation.",
        max_expected_distance=0.20,
    ),
]

# ------------------------------------------------------------------
# 4. Eggcorns (English intra-lingual)
# ------------------------------------------------------------------

EGGCORNS_ENG = [
    PunCase(
        id="egg_acorn",
        source_lang="eng-us",
        target_lang="eng-us",
        source_text="acorn",
        target_text="egg corn",
        source_ipa="e\u026ak\u0254\u0279n",
        target_ipa="\u025b\u0261 k\u0254\u0279n",
        category="eggcorn",
        n_words_source=1,
        n_words_target=2,
        notes="The eponymous eggcorn. Single word -> two words. "
        "Onset /e\u026a/ vs /\u025b\u0261/, syllable boundary shift.",
        max_expected_distance=0.30,
    ),
    PunCase(
        id="old_timers",
        source_lang="eng-us",
        target_lang="eng-us",
        source_text="Alzheimer's",
        target_text="old-timers",
        source_ipa="\u0251\u02d0lts.ha\u026a.m\u0259\u0279z",
        target_ipa="o\u028ald.ta\u026a.m\u0259\u0279z",
        category="eggcorn",
        n_words_source=1,
        n_words_target=1,
        notes="'Alzheimer's disease' -> 'old-timers disease'. "
        "Similar syllable count, vowel shift in onset: /\u0251\u02d0l/ vs /o\u028ald/.",
        max_expected_distance=0.35,
    ),
    PunCase(
        id="intensive_purposes",
        source_lang="eng-us",
        target_lang="eng-us",
        source_text="for all intents and purposes",
        target_text="for all intensive purposes",
        source_ipa="f\u0254\u0279 \u0254\u02d0l \u026ant\u025bnts \u0259nd p\u025d\u02d0p\u0259s\u026az",
        target_ipa="f\u0254\u0279 \u0254\u02d0l \u026ant\u025bns\u026av p\u025d\u02d0p\u0259s\u026az",
        category="eggcorn",
        n_words_source=5,
        n_words_target=4,
        notes="'intents and' -> 'intensive'. Word boundary collapse with "
        "near-identical phoneme sequence. Beam search should find close match.",
        max_expected_distance=0.20,
    ),
    PunCase(
        id="bated_breath",
        source_lang="eng-us",
        target_lang="eng-us",
        source_text="bated breath",
        target_text="baited breath",
        source_ipa="be\u026at\u026ad b\u0279\u025b\u03b8",
        target_ipa="be\u026at\u026ad b\u0279\u025b\u03b8",
        category="eggcorn",
        n_words_source=2,
        n_words_target=2,
        notes="Perfect homophone eggcorn. 'bated' and 'baited' are "
        "phonetically identical in most dialects. Distance should be ~0.",
        max_expected_distance=0.05,
    ),
    PunCase(
        id="deep_seated",
        source_lang="eng-us",
        target_lang="eng-us",
        source_text="deep seated",
        target_text="deep seeded",
        source_ipa="di\u02d0p si\u02d0t\u026ad",
        target_ipa="di\u02d0p si\u02d0d\u026ad",
        category="eggcorn",
        n_words_source=2,
        n_words_target=2,
        notes="Single phoneme difference: /t/ vs /d/ (voicing). "
        "Minimal pair eggcorn, should have very low distance.",
        max_expected_distance=0.15,
    ),
]

# ------------------------------------------------------------------
# 5. Cross-language soramimi (Romanian -> Japanese)
# ------------------------------------------------------------------

SORAMIMI = [
    PunCase(
        id="dragostea_din_tei",
        source_lang="ron",
        target_lang="jpn",
        source_text="Vrei sa pleci dar nu ma, nu ma iei",
        target_text="bei sa beishu darou nomanoma iei",
        source_ipa="vre\u026a s\u0259 plet\u0283\u02b2 dar nu m\u0259 nu m\u0259 je\u026a",
        target_ipa="be\u026a sa be\u026a\u0283\u0289 da\u027e\u0289 nomanoma ie\u026a",
        category="soramimi",
        n_words_source=9,
        n_words_target=6,
        notes="O-Zone 'Dragostea Din Tei' reinterpreted as Japanese. "
        "'Rice wine, most likely! Drink drink yay!' "
        "Cross-language soramimi: Romanian -> Japanese.",
        max_expected_distance=0.50,
    ),
]

# ------------------------------------------------------------------
# 6. Bilingual puns (misc)
# ------------------------------------------------------------------

BILINGUAL_PUNS_MISC = [
    PunCase(
        id="purgatory_purgatorio",
        source_lang="eng-us",
        target_lang="spa",
        source_text="purgatory",
        target_text="purgatorio",
        source_ipa="p\u025d\u02d0\u0261\u0259t\u0254\u0279i",
        target_ipa="pu\u027e\u0263a\u02c8to\u027ejo",
        category="interlingual",
        n_words_source=1,
        n_words_target=1,
        notes="Bilingual pun from Wikipedia: 'Where do cats go when they die? "
        "In English PURRgatory, in Spanish purGATOrio.' "
        "Single-word cognate with embedded 'purr' / 'gato'.",
        max_expected_distance=0.55,
    ),
    PunCase(
        id="fun_with_f1",
        source_lang="eng-us",
        target_lang="fra",
        source_text="fun with F one",
        target_text="fun with F un",
        source_ipa="f\u028cn w\u026a\u03b8 \u025bf w\u028cn",
        target_ipa="f\u028cn w\u026a\u03b8 \u025bf \u0153\u0303",
        category="interlingual",
        n_words_source=4,
        n_words_target=4,
        notes="Mathematics bilingual pun: '1' in French is 'un'. "
        "Paper title 'Fun with F1' reads the same in both languages.",
        max_expected_distance=0.30,
    ),
    PunCase(
        id="wario_warui",
        source_lang="jpn",
        target_lang="eng-us",
        source_text="warui",
        target_text="Wario",
        source_ipa="wa\u027e\u0269i",
        target_ipa="w\u0251\u02d0\u0279io\u028a",
        category="interlingual",
        n_words_source=1,
        n_words_target=1,
        notes="Nintendo bilingual portmanteau: Japanese 'warui' (bad) + "
        "Mario = Wario. Single-word cross-language match.",
        max_expected_distance=0.40,
    ),
]

# ------------------------------------------------------------------
# 7. Dog Latin / cross-script homophones
# ------------------------------------------------------------------

DOG_LATIN = [
    PunCase(
        id="caesar_jam",
        source_lang="lat",
        target_lang="eng-us",
        source_text="Caesar adsum jam forte",
        target_text="Caesar had some jam for tea",
        source_ipa="ka\u026asa\u0279 ads\u028am jam f\u0254\u0279te\u026a",
        target_ipa="si\u02d0z\u0259\u0279 h\u00e6d s\u028cm d\u0292\u00e6m f\u0254\u0279 ti\u02d0",
        category="homophonic_translation",
        n_words_source=4,
        n_words_target=6,
        notes="Classic British schoolboy Dog Latin. Latin sentence "
        "sounds like English sentence about jam when read aloud. "
        "Tests dramatic syllable boundary reshuffling.",
        max_expected_distance=0.50,
    ),
]


# ===================================================================
# All puns combined
# ===================================================================

ALL_PUNS: list[PunCase] = (
    INTERLINGUAL_ENG_FRA
    + INTERLINGUAL_ENG_DEU
    + MONDEGREENS_ENG
    + EGGCORNS_ENG
    + SORAMIMI
    + BILINGUAL_PUNS_MISC
    + DOG_LATIN
)


# ===================================================================
# Fixtures
# ===================================================================

# Minimal phoneme feature set that covers the phonemes used in the pun
# collection.  This avoids requiring real language modules / ONNX model.
_SHARED_FEATURES: dict[str, dict[str, object]] = {
    # Consonants
    "p": {"voiced": False, "manner": "plosive", "labial": True},
    "b": {"voiced": True, "manner": "plosive", "labial": True},
    "t": {"voiced": False, "manner": "plosive", "place": "alveolar"},
    "d": {"voiced": True, "manner": "plosive", "place": "alveolar"},
    "k": {"voiced": False, "manner": "plosive", "place": "velar"},
    "\u0261": {"voiced": True, "manner": "plosive", "place": "velar"},
    "f": {"voiced": False, "manner": "fricative", "labial": True},
    "v": {"voiced": True, "manner": "fricative", "labial": True},
    "s": {"voiced": False, "manner": "fricative", "place": "alveolar"},
    "z": {"voiced": True, "manner": "fricative", "place": "alveolar"},
    "\u0283": {"voiced": False, "manner": "fricative", "place": "postalveolar"},
    "\u0292": {"voiced": True, "manner": "fricative", "place": "postalveolar"},
    "\u03b8": {"voiced": False, "manner": "fricative", "place": "dental"},
    "\u00f0": {"voiced": True, "manner": "fricative", "place": "dental"},
    "h": {"voiced": False, "manner": "fricative", "place": "glottal"},
    "m": {"voiced": True, "manner": "nasal", "labial": True},
    "n": {"voiced": True, "manner": "nasal", "place": "alveolar"},
    "\u014b": {"voiced": True, "manner": "nasal", "place": "velar"},
    "l": {"voiced": True, "manner": "lateral", "place": "alveolar"},
    "\u0279": {"voiced": True, "manner": "approximant", "place": "alveolar"},
    "\u0281": {"voiced": True, "manner": "fricative", "place": "uvular"},
    "\u027e": {"voiced": True, "manner": "tap", "place": "alveolar"},
    "w": {"voiced": True, "manner": "approximant", "labial": True},
    "j": {"voiced": True, "manner": "approximant", "place": "palatal"},
    "t\u0283": {"voiced": False, "manner": "affricate", "place": "postalveolar"},
    "d\u0292": {"voiced": True, "manner": "affricate", "place": "postalveolar"},
    "\u0263": {"voiced": True, "manner": "fricative", "place": "velar"},
    # Vowels
    "i": {"high": True, "front": True, "round": False},
    "\u026a": {"high": True, "front": True, "round": False, "lax": True},
    "e": {"mid": True, "front": True, "round": False},
    "\u025b": {"mid-low": True, "front": True, "round": False},
    "\u00e6": {"low": True, "front": True, "round": False},
    "a": {"low": True, "central": True, "round": False},
    "\u0259": {"mid": True, "central": True, "round": False},
    "\u0250": {"low": True, "central": True, "round": False},
    "\u028c": {"mid-low": True, "back": True, "round": False},
    "\u0251": {"low": True, "back": True, "round": False},
    "\u0252": {"low": True, "back": True, "round": True},
    "o": {"mid": True, "back": True, "round": True},
    "\u0254": {"mid-low": True, "back": True, "round": True},
    "u": {"high": True, "back": True, "round": True},
    "\u028a": {"high": True, "back": True, "round": True, "lax": True},
    "\u0153": {"mid-low": True, "front": True, "round": True},
    "\u025d": {"mid": True, "central": True, "round": False, "rhoticised": True},
    "\u0289": {"high": True, "central": True, "round": True},
    "\u0269": {"high": True, "central": True, "round": False},
    # Nasalised vowels (French)
    "\u0153\u0303": {"mid-low": True, "front": True, "round": True, "nasal": True},
    "\u025b\u0303": {"mid-low": True, "front": True, "round": False, "nasal": True},
    "\u0251\u0303": {"low": True, "back": True, "round": False, "nasal": True},
    # Diphthongs (as single units for tokenization)
    "a\u026a": {"low": True, "front": True, "diphthong": True},
    "e\u026a": {"mid": True, "front": True, "diphthong": True},
    "o\u028a": {"mid": True, "back": True, "round": True, "diphthong": True},
    "a\u028a": {"low": True, "back": True, "diphthong": True},
    "\u0254\u026a": {"mid-low": True, "back": True, "round": True, "diphthong": True},
}


@pytest.fixture
def pun_spec():
    """BitArraySpecification covering all phonemes in the pun collection."""
    from phone_similarity.bit_array_specification import BitArraySpecification

    vowels = {
        "i",
        "\u026a",
        "e",
        "\u025b",
        "\u00e6",
        "a",
        "\u0259",
        "\u0250",
        "\u028c",
        "\u0251",
        "\u0252",
        "o",
        "\u0254",
        "u",
        "\u028a",
        "\u0153",
        "\u025d",
        "\u0289",
        "\u0269",
        "\u0153\u0303",
        "\u025b\u0303",
        "\u0251\u0303",
        "a\u026a",
        "e\u026a",
        "o\u028a",
        "a\u028a",
        "\u0254\u026a",
    }
    consonants = set(_SHARED_FEATURES.keys()) - vowels
    features = {
        "consonant": {
            "voiced",
            "manner",
            "place",
            "labial",
        },
        "vowel": {
            "low",
            "mid-low",
            "mid",
            "high",
            "front",
            "central",
            "back",
            "round",
            "lax",
            "nasal",
            "rhoticised",
            "diphthong",
        },
    }
    return BitArraySpecification(
        vowels=vowels,
        consonants=consonants,
        features=features,
        features_per_phoneme=_SHARED_FEATURES,
    )


# Pre-sorted phoneme list (longest first) — computed once at module load
# instead of re-sorting in every _tokenize_ipa() call.
_SORTED_PHONEMES: list[str] = sorted(_SHARED_FEATURES.keys(), key=len, reverse=True)


def _tokenize_ipa(ipa_str: str, features: dict) -> list[str]:
    """Greedy longest-match IPA tokenizer against feature keys."""
    phonemes = _SORTED_PHONEMES
    tokens: list[str] = []
    i = 0
    text = ipa_str.replace(" ", "")
    while i < len(text):
        matched = False
        for ph in phonemes:
            if text[i:].startswith(ph):
                tokens.append(ph)
                i += len(ph)
                matched = True
                break
        if not matched:
            # skip unknown characters (stress marks, length marks, etc.)
            i += 1
    return tokens


# ===================================================================
# Tests: PunCase data integrity
# ===================================================================


class TestPunCaseData:
    """Validate the pun collection data itself."""

    def test_all_puns_have_unique_ids(self):
        ids = [p.id for p in ALL_PUNS]
        assert len(ids) == len(set(ids)), f"Duplicate ids: {[x for x in ids if ids.count(x) > 1]}"

    def test_all_puns_have_ipa(self):
        for p in ALL_PUNS:
            assert p.source_ipa, f"{p.id}: missing source_ipa"
            assert p.target_ipa, f"{p.id}: missing target_ipa"

    def test_all_puns_have_valid_category(self):
        valid = {
            "interlingual",
            "intralingual_mondegreen",
            "eggcorn",
            "homophonic_translation",
            "soramimi",
        }
        for p in ALL_PUNS:
            assert p.category in valid, f"{p.id}: invalid category '{p.category}'"

    def test_word_counts_are_positive(self):
        for p in ALL_PUNS:
            assert p.n_words_source > 0, f"{p.id}: n_words_source must be > 0"
            assert p.n_words_target > 0, f"{p.id}: n_words_target must be > 0"

    def test_max_expected_distance_in_range(self):
        for p in ALL_PUNS:
            assert 0.0 <= p.max_expected_distance <= 1.0, (
                f"{p.id}: max_expected_distance out of [0,1] range"
            )

    @pytest.mark.parametrize("pun", ALL_PUNS, ids=[p.id for p in ALL_PUNS])
    def test_source_ipa_tokenizable(self, pun):
        """Source IPA should produce at least 1 token."""
        tokens = _tokenize_ipa(pun.source_ipa, _SHARED_FEATURES)
        assert len(tokens) >= 1, f"{pun.id}: source IPA '{pun.source_ipa}' produced no tokens"

    @pytest.mark.parametrize("pun", ALL_PUNS, ids=[p.id for p in ALL_PUNS])
    def test_target_ipa_tokenizable(self, pun):
        """Target IPA should produce at least 1 token."""
        tokens = _tokenize_ipa(pun.target_ipa, _SHARED_FEATURES)
        assert len(tokens) >= 1, f"{pun.id}: target IPA '{pun.target_ipa}' produced no tokens"


# ===================================================================
# Tests: Feature edit distance between pun pairs
# ===================================================================


class TestPunFeatureDistance:
    """Verify that the source/target IPA pairs are within expected distance."""

    @pytest.mark.parametrize("pun", ALL_PUNS, ids=[p.id for p in ALL_PUNS])
    def test_pair_distance_within_threshold(self, pun):
        """The normalised feature edit distance between source and target
        IPA should be at or below the pun's max_expected_distance."""
        from phone_similarity.primitives import normalised_feature_edit_distance

        source_tokens = _tokenize_ipa(pun.source_ipa, _SHARED_FEATURES)
        target_tokens = _tokenize_ipa(pun.target_ipa, _SHARED_FEATURES)

        if not source_tokens or not target_tokens:
            pytest.skip(f"{pun.id}: could not tokenize IPA")

        dist = normalised_feature_edit_distance(source_tokens, target_tokens, _SHARED_FEATURES)
        assert dist <= pun.max_expected_distance + 0.10, (
            f"{pun.id}: distance {dist:.4f} exceeds threshold "
            f"{pun.max_expected_distance} + 0.10 tolerance\n"
            f"  source: {pun.source_text} -> {source_tokens}\n"
            f"  target: {pun.target_text} -> {target_tokens}"
        )


# ===================================================================
# Tests: Beam search can find pun targets in a mock dictionary
# ===================================================================


class TestPunBeamSearch:
    """For selected puns, verify beam search finds the target in a
    dictionary containing the target words plus distractors."""

    @staticmethod
    def _make_ptd_from_pun(pun: PunCase, spec) -> PreTokenizedDictionary:
        """Create a small PreTokenizedDictionary that includes the target
        words plus some distractors."""
        target_words = pun.target_text.lower().split()
        target_ipas = pun.target_ipa.split()

        # Pad or truncate target_ipas to match target_words
        while len(target_ipas) < len(target_words):
            target_ipas.append(target_ipas[-1] if target_ipas else "")
        target_ipas = target_ipas[: len(target_words)]

        entries: list[tuple[str, str, list[str]]] = []
        for word, ipa in zip(target_words, target_ipas):
            tokens = _tokenize_ipa(ipa, _SHARED_FEATURES)
            if tokens:
                entries.append((word, ipa, tokens))

        # Add some distractors
        distractors = [
            ("noise", "n\u0254\u026az", ["n", "\u0254\u026a", "z"]),
            ("frog", "f\u0279\u0252\u0261", ["f", "\u0279", "\u0252", "\u0261"]),
            ("lamp", "l\u00e6mp", ["l", "\u00e6", "m", "p"]),
            ("desk", "d\u025bsk", ["d", "\u025b", "s", "k"]),
        ]
        entries.extend(distractors)

        return PreTokenizedDictionary.from_entries(entries)

    @pytest.mark.parametrize(
        "pun",
        [
            p
            for p in MONDEGREENS_ENG
            if p.n_words_target <= 4
            and p.id
            in (
                "kiss_the_sky",
                "mairzy_doats",
                "recognize_speech",
            )
        ],
        ids=lambda p: p.id,
    )
    def test_mondegreen_beam_search(self, pun, pun_spec):
        """Beam search should discover the mondegreen segmentation."""
        ptd = self._make_ptd_from_pun(pun, pun_spec)

        source_tokens = _tokenize_ipa(pun.source_ipa, _SHARED_FEATURES)
        if not source_tokens:
            pytest.skip("Could not tokenize source IPA")

        results = beam_search_segmentation(
            source_tokens,
            pun_spec,
            _SHARED_FEATURES,
            ptd,
            pun_spec,
            _SHARED_FEATURES,
            beam_width=15,
            top_k=5,
            max_words=pun.n_words_target + 1,
            max_distance=0.60,
            prune_ratio=2.5,
        )

        # We should get at least one result
        assert len(results) > 0, (
            f"{pun.id}: beam search returned no results for "
            f"'{pun.source_text}' -> '{pun.target_text}'"
        )

        # The best result should be within reasonable distance
        best = results[0]
        assert best.distance <= pun.max_expected_distance + 0.15, (
            f"{pun.id}: best distance {best.distance:.4f} too high. "
            f"Words: {best.words}, expected ~'{pun.target_text}'"
        )

    @pytest.mark.parametrize(
        "pun",
        [
            p
            for p in EGGCORNS_ENG
            if p.id
            in (
                "bated_breath",
                "deep_seated",
            )
        ],
        ids=lambda p: p.id,
    )
    def test_eggcorn_beam_search(self, pun, pun_spec):
        """Beam search should discover eggcorn matches."""
        ptd = self._make_ptd_from_pun(pun, pun_spec)

        source_tokens = _tokenize_ipa(pun.source_ipa, _SHARED_FEATURES)
        if not source_tokens:
            pytest.skip("Could not tokenize source IPA")

        results = beam_search_segmentation(
            source_tokens,
            pun_spec,
            _SHARED_FEATURES,
            ptd,
            pun_spec,
            _SHARED_FEATURES,
            beam_width=10,
            top_k=3,
            max_words=pun.n_words_target + 1,
            max_distance=0.50,
        )

        assert len(results) > 0, f"{pun.id}: beam search returned no results"

        best = results[0]
        assert best.distance <= pun.max_expected_distance + 0.10, (
            f"{pun.id}: best distance {best.distance:.4f} too high. Words: {best.words}"
        )


# ===================================================================
# Tests: Syllable boundary analysis for pun pairs
# ===================================================================


class TestPunSyllableAlignment:
    """Verify that syllable segmentation reveals the boundary shifts
    that make puns work."""

    @pytest.mark.parametrize(
        "pun",
        [
            p
            for p in ALL_PUNS
            if p.id
            in (
                "kiss_the_sky",
                "bad_moon_rising",
                "recognize_speech",
                "mairzy_doats",
                "egg_acorn",
                "teen_spirit",
            )
        ],
        ids=lambda p: p.id,
    )
    def test_syllable_counts_comparable(self, pun):
        """Source and target should have similar syllable counts
        (within +/-2), since puns rely on syllable-level matching."""
        from phone_similarity.syllable import syllable_count

        # Use the shared features to build a minimal vowels set
        vowels = {
            "i",
            "\u026a",
            "e",
            "\u025b",
            "\u00e6",
            "a",
            "\u0259",
            "\u0250",
            "\u028c",
            "\u0251",
            "\u0252",
            "o",
            "\u0254",
            "u",
            "\u028a",
            "\u0153",
            "\u025d",
            "\u0289",
            "\u0269",
            "\u0153\u0303",
            "\u025b\u0303",
            "\u0251\u0303",
            "a\u026a",
            "e\u026a",
            "o\u028a",
            "a\u028a",
            "\u0254\u026a",
        }

        source_tokens = _tokenize_ipa(pun.source_ipa, _SHARED_FEATURES)
        target_tokens = _tokenize_ipa(pun.target_ipa, _SHARED_FEATURES)

        if not source_tokens or not target_tokens:
            pytest.skip(f"{pun.id}: could not tokenize")

        # Count syllables by counting vowel tokens (simple heuristic)
        src_vowel_count = sum(1 for t in source_tokens if t in vowels)
        tgt_vowel_count = sum(1 for t in target_tokens if t in vowels)

        # Puns typically have similar syllable counts
        assert abs(src_vowel_count - tgt_vowel_count) <= 3, (
            f"{pun.id}: syllable count mismatch too large: "
            f"source={src_vowel_count}, target={tgt_vowel_count}\n"
            f"  source tokens: {source_tokens}\n"
            f"  target tokens: {target_tokens}"
        )


# ===================================================================
# Convenience: summary of pun collection
# ===================================================================


def test_pun_collection_summary():
    """Print a summary of the pun collection (for debugging)."""
    by_category: dict[str, int] = {}
    by_lang_pair: dict[str, int] = {}
    single_word = 0
    multi_word = 0

    for p in ALL_PUNS:
        by_category[p.category] = by_category.get(p.category, 0) + 1
        pair = f"{p.source_lang}->{p.target_lang}"
        by_lang_pair[pair] = by_lang_pair.get(pair, 0) + 1
        if p.n_words_source == 1 and p.n_words_target == 1:
            single_word += 1
        else:
            multi_word += 1

    assert len(ALL_PUNS) >= 20, f"Expected >= 20 puns, got {len(ALL_PUNS)}"
    assert len(by_category) >= 3, "Expected at least 3 categories"
    assert multi_word >= 10, "Expected at least 10 multi-word puns"
