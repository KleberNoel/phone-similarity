#!/usr/bin/env python3
"""
Evaluate phone-similarity on phonetic puns from Will Styler's collection.
Source: https://wstyler.ucsd.edu/puns/

PART 1: Pun-pair normalised edit distance (with best-of-N pronunciation matching)
PART 2: Oronym beam search recall (English→English same-language segmentation)
PART 3: Recovered false negatives via beam search
PART 4: Bonus oronyms discovered via beam search mining
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from itertools import product

from phone_similarity.beam_search import beam_search_segmentation
from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator
from phone_similarity.language import LANGUAGES
from phone_similarity.pretokenize import cached_pretokenize_dictionary

# ── Setup ────────────────────────────────────────────────────────────────────
g2p = CharsiuGraphemeToPhonemeGenerator("eng-us")
eng_spec = LANGUAGES.build_spec("eng_us")
eng = LANGUAGES["eng_us"]
dist_obj = LANGUAGES.build_distance("eng_us")


def ipa_all(word: str, manual: str | None = None) -> list[str]:
    """Return all IPA pronunciations for *word*. Manual override if provided."""
    if manual:
        return [manual]
    clean = word.strip().lower()
    v = g2p.pdict.get(clean)
    if v:
        return [x.strip() for x in v.split(",") if x.strip()]
    return []


def multi_word_best_ned(
    single_ipa: str, words: list[str]
) -> tuple[float | None, tuple[str, ...] | None, str | None]:
    """Best NED across all pronunciation combos (itertools.product) for a multi-word parse.

    Returns (best_distance, best_combo_ipa_tuple, best_joined_ipa).
    """
    word_prons = [ipa_all(w) for w in words]
    if any(not p for p in word_prons):
        return (None, None, None)
    best_d: float | None = None
    best_combo: tuple[str, ...] | None = None
    best_joined: str | None = None
    for combo in product(*word_prons):
        joined = "".join(combo)
        try:
            d = dist_obj.normalised_edit_distance(single_ipa, joined)
            if best_d is None or d < best_d:
                best_d = d
                best_combo = combo
                best_joined = joined
        except Exception:
            continue
    return (best_d, best_combo, best_joined)


def ned_best(ipas_a: list[str], ipas_b: list[str]) -> float | None:
    """Min normalised edit distance across all pronunciation variant pairs."""
    if not ipas_a or not ipas_b:
        return None
    best = None
    for a in ipas_a:
        for b in ipas_b:
            try:
                d = dist_obj.normalised_edit_distance(a, b)
                if best is None or d < best:
                    best = d
            except Exception:
                continue
    return best


def best_ipa_pair(
    ipas_a: list[str], ipas_b: list[str]
) -> tuple[str | None, str | None, float | None]:
    """Return the (ipa_a, ipa_b, distance) triple that minimises NED."""
    if not ipas_a or not ipas_b:
        return (ipas_a[0] if ipas_a else None, ipas_b[0] if ipas_b else None, None)
    best_d = None
    best_a = best_b = None
    for a in ipas_a:
        for b in ipas_b:
            try:
                d = dist_obj.normalised_edit_distance(a, b)
                if best_d is None or d < best_d:
                    best_d, best_a, best_b = d, a, b
            except Exception:
                continue
    return best_a, best_b, best_d


# ── PART 1: Pun pair dataset ────────────────────────────────────────────────
# (label, pun_word, target_word, manual_ipa_pun, manual_ipa_target)

PHONETIC_PUNS: list[tuple[str, str, str, str | None, str | None]] = [
    ("Jalapeño business", "jalapeno", "all up in yo", "ˌhɑləˈpeɪnjoʊ", "ˌɔləˈpɪnjoʊ"),
    ("big pause/paws", "pause", "paws", None, None),
    ("investigator", "investigator", "in vest a gator", None, "ɪnˈvɛstəˌɡeɪtɚ"),
    ("Alpaca lunch", "alpaca", "i'll pack a", None, "aɪlˈpækə"),
    ("stomach pane/pain", "pane", "pain", None, None),
    ("iWitness/eyewitness", "iwitness", "eyewitness", "aɪˈwɪtnəs", None),
    ("horse/hoarse", "horse", "hoarse", None, None),
    ("Luke Warm/lukewarm", "lukewarm", "lukewarm", None, None),
    ("Cheetahs/cheaters", "cheetahs", "cheaters", None, None),
    ("Shitzu/shit zoo", "shih tzu", "shit zoo", "ˈʃitsu", "ˈʃɪtˌzu"),
    ("fillings/feelings", "fillings", "feelings", None, None),
    (
        "soda pressing/so depressing",
        "soda pressing",
        "so depressing",
        "ˈsoʊdəˌpɹɛsɪŋ",
        "soʊdɪˈpɹɛsɪŋ",
    ),
    ("leaf/leave", "leaf", "leave", None, None),
    ("missed steak/mistake", "missed steak", "mistake", "ˌmɪstˈsteɪk", "mɪˈsteɪk"),
    ("Macawbre/macabre", "macawbre", "macabre", "məˈkɔbɹeɪ", None),
    ("Releaved/relieved", "releaved", "relieved", "ɹɪˈlivd", None),
    (
        "case ideas/quesadillas",
        "case ideas",
        "quesadillas",
        "ˈkeɪsaɪˈdiəz",
        "ˌkeɪsəˈdiəz",
    ),
    ("oar deal/ordeal", "oar deal", "ordeal", "ˈɔɹˌdil", None),
    ("Spellfie/selfie", "spellfie", "selfie", "ˈspɛlfi", "ˈsɛlfi"),
    ("hell toupee/to pay", "hell toupee", "hell to pay", "ˌhɛltuˈpeɪ", "ˌhɛltəˈpeɪ"),
    (
        "Neverlands/never lands",
        "neverlands",
        "never lands",
        "ˈnɛvɚˌlændz",
        "ˈnɛvɚˌlændz",
    ),
    (
        "resisting a rest/arrest",
        "resisting a rest",
        "resisting arrest",
        "ɹɪˈzɪstɪŋəˈɹɛst",
        "ɹɪˈzɪstɪŋəˈɹɛst",
    ),
    ("sue chef/sous chef", "sue chef", "sous chef", "ˈsuˌʃɛf", "ˈsuˌʃɛf"),
    (
        "met herbivore/her before",
        "met herbivore",
        "met her before",
        "ˌmɛtˈhɝbɪvɔɹ",
        "ˌmɛtɚˈbɪfɔɹ",
    ),
    ("in tents/intense", "in tents", "intense", "ɪnˈtɛnts", None),
    ("All-Terrier/ulterior", "all terrier", "ulterior", "ˌɔlˈtɛɹiɚ", "ʌlˈtɪɹiɚ"),
    (
        "whisk averse/risk averse",
        "whisk averse",
        "risk averse",
        "ˌwɪskəˈvɝs",
        "ˌɹɪskəˈvɝs",
    ),
    ("syncing/sinking", "syncing", "sinking", "ˈsɪŋkɪŋ", None),
    ("wurst/worst", "wurst", "worst", None, None),
    ("knead/need", "knead", "need", None, None),
    ("vitamin/vite em in", "vitamin", "vite em in", None, "ˈvaɪtəmɪn"),
    ("canarial/venereal", "canarial", "venereal", "kəˈnɛɹiəl", None),
    ("despair/a spare", "despair", "a spare", None, "əˈspɛɹ"),
    ("Czardines/sardines", "czardines", "sardines", "ˈzɑɹdinz", None),
    ("miner/minor", "miner", "minor", None, None),
    ("Daytrogen/nitrogen", "daytrogen", "nitrogen", "ˈdeɪtɹədʒɪn", None),
    ("sighsmograph/seismograph", "sighsmograph", "seismograph", "ˈsaɪzməˌɡɹæf", None),
    (
        "Mettamorphosis/metamorphosis",
        "mettamorphosis",
        "metamorphosis",
        "ˌmɛtəˈmɔɹfəsɪs",
        None,
    ),
    ("bar tender/bartender", "bar tender", "bartender", "ˈbɑɹˌtɛndɚ", None),
    ("Tres/trace", "tres", "trace", "tɹɛs", None),
    (
        "soda lighted/so delighted",
        "soda lighted",
        "so delighted",
        "ˌsoʊdəˈlaɪtɪd",
        "ˌsoʊdɪˈlaɪtɪd",
    ),
    ("Gluten/glutton", "gluten", "glutton", None, None),
    (
        "stow thrones/throw stones",
        "stow thrones",
        "throw stones",
        "ˌstoʊˈθɹoʊnz",
        "ˌθɹoʊˈstoʊnz",
    ),
    ("jester/gesture", "jester", "gesture", None, None),
    ("pigment/figment", "pigment", "figment", None, None),
    ("argon/are gone", "argon", "are gone", None, "ɑɹˈɡɑn"),
    ("four/for", "four", "for", None, None),
    ("Bison/bye son", "bison", "bye son", None, "ˈbaɪsʌn"),
    ("Cah/Car", "cah", "car", "kɑ", "kɑɹ"),
    ("LAN/land", "lan", "land", "læn", None),
    ("Custardy/custody", "custardy", "custody", "ˈkʌstɚdi", None),
    ("in Seine/insane", "in seine", "insane", "ɪnˈseɪn", None),
    ("Medusinal/medicinal", "medusinal", "medicinal", "mɪˈdjusɪnəl", None),
    ("Tome/Tom", "tome", "tom", None, "tɑm"),
    ("apartmint/apartment", "apartmint", "apartment", "əˈpɑɹtmɪnt", None),
    ("Commas/Commons", "commas", "commons", None, None),
    ("Squire/square", "squire", "square", None, None),
    ("propaganda/proper gander", "propaganda", "proper gander", None, "ˌpɹɑpɚˈɡændɚ"),
    (
        "catastrophe/cat has trophy",
        "cat has trophy",
        "catastrophe",
        "ˌkætˈhæztɹoʊfi",
        None,
    ),
    ("Neigh-sayers/naysayers", "neigh sayers", "naysayers", "ˈneɪˌseɪɚz", None),
    ("multi porpoise/purpose", "porpoise", "purpose", None, None),
    (
        "Toadally ribbeting/riveting",
        "toadally ribbeting",
        "totally riveting",
        "ˌtoʊdəliˈɹɪbɪtɪŋ",
        "ˌtoʊɾəliˈɹɪvɪtɪŋ",
    ),
    ("steaks/stakes", "steaks", "stakes", None, None),
    ("apps/ass", "apps", "ass", None, None),
    ("Sham Pain/champagne", "sham pain", "champagne", "ˌʃæmˈpeɪn", None),
    (
        "Ayes for Ewe/eyes for you",
        "ayes for ewe",
        "eyes for you",
        "ˈaɪzfɚˈju",
        "ˈaɪzfɚˈju",
    ),
    ("Al Dante/al dente", "al dante", "al dente", "ˌælˈdɑnteɪ", "ˌælˈdɛnteɪ"),
    ("boy ant/buoyant", "boy ant", "buoyant", "ˈbɔɪˌænt", None),
    ("General Lee/generally", "general lee", "generally", "ˈdʒɛnɚəlˈli", None),
    (
        "diskoalafying/disqualifying",
        "diskoalafying",
        "disqualifying",
        "ˌdɪsˈkoʊɑləˌfaɪɪŋ",
        None,
    ),
    ("Sheik down/shakedown", "sheik down", "shakedown", "ˈʃeɪkˌdaʊn", None),
    ("meat/meet", "meat", "meet", None, None),
    ("fo drizzle/fo shizzle", "fo drizzle", "fo shizzle", "foʊˈdɹɪzəl", "foʊˈʃɪzəl"),
    ("Thor loser/sore loser", "thor loser", "sore loser", "ˈθɔɹˌluzɚ", "ˈsɔɹˌluzɚ"),
    ("OMg/oh my god", "omg", "oh my god", "oʊˈɛmˈdʒi", "oʊˈmaɪˈɡɑd"),
    ("cents/sense", "cents", "sense", None, None),
    ("night mare/nightmare", "night mare", "nightmare", "ˈnaɪtˌmɛɹ", None),
    ("Simbalism/symbolism", "simbalism", "symbolism", "ˈsɪmbəˌlɪzəm", None),
    ("foal/fool", "foal", "fool", None, None),
    (
        "O tempura O moray",
        "o tempura o moray",
        "o tempora o mores",
        "oʊˈtɛmpɚəoʊˈmɔɹeɪ",
        "oʊˈtɛmpɚəoʊˈmɔɹeɪz",
    ),
    ("porpoise/purpose", "porpoise", "purpose", None, None),
    ("Star Bucks/Starbucks", "star bucks", "starbucks", "ˈstɑɹˌbʌks", "ˈstɑɹˌbʌks"),
    (
        "two in tents/too intense",
        "two in tents",
        "too intense",
        "ˌtuɪnˈtɛnts",
        "ˌtuɪnˈtɛns",
    ),
    ("infant tree/infantry", "infant tree", "infantry", "ˈɪnfəntˌtɹi", None),
    ("vowel/bowel", "vowel", "bowel", None, None),
    (
        "Collaboradors/collaborators",
        "collaboradors",
        "collaborators",
        "kəˈlæbɚəˌdɔɹz",
        None,
    ),
    ("Vaal/fall", "vaal", "fall", "vɑl", None),
    ("Tooth hurty/two thirty", "tooth hurty", "two thirty", "ˈtuθˌhɝti", "ˈtuˌθɝti"),
    ("Fresh Prints/Prince", "prints", "prince", None, None),
    (
        "in da skies/in disguise",
        "in da skies",
        "in disguise",
        "ɪndəˈskaɪz",
        "ɪndɪsˈɡaɪz",
    ),
    ("drama dairy/dromedary", "drama dairy", "dromedary", "ˈdɹɑməˌdɛɹi", None),
    ("Nguyen/win", "nguyen", "win", "ˈwɪn", None),
    (
        "Scandinavian/scan the avian",
        "scandinavian",
        "scan the avian",
        None,
        "ˌskændɪˈneɪviən",
    ),
    ("Thai/tie", "thai", "tie", None, None),
    ("pier/peer", "pier", "peer", None, None),
    ("aloha/a lower", "aloha", "a lower", None, "əˈloʊɚ"),
    (
        "super fish oil/superficial",
        "super fish oil",
        "superficial",
        "ˌsupɚˈfɪʃəl",
        None,
    ),
    ("De brie/debris", "de brie", "debris", "dəˈbɹi", None),
    ("tines/times", "tines", "times", None, None),
    ("herd/hurt", "herd", "hurt", None, None),
    ("sea/C", "sea", "c", "si", "si"),
    ("carrion/carry on", "carrion", "carry on", None, "ˈkæɹiˌɑn"),
    ("no bell/Nobel", "no bell", "nobel", "ˌnoʊˈbɛl", None),
    ("scenter/center", "scenter", "center", "ˈsɛntɚ", None),
    ("irrelephant/irrelevant", "irrelephant", "irrelevant", "ˌɪɹˈɛləfənt", None),
    ("ova/over", "ova", "over", None, None),
    ("Roemancer/romancer", "roemancer", "romancer", "ɹoʊˈmænsɚ", None),
    (
        "extroversion/extra virgin",
        "extroversion",
        "extra virgin",
        None,
        "ˌɛkstɹəˈvɝdʒɪn",
    ),
    ("passenger pidgin/pigeon", "pidgin", "pigeon", None, None),
    ("bouillonaire/billionaire", "bouillonaire", "billionaire", "ˌbuljəˈnɛɹ", None),
    ("centsless/senseless", "centsless", "senseless", "ˈsɛntslɪs", None),
    ("punnish mint/punishment", "punnish mint", "punishment", "ˈpʌnɪʃˌmɪnt", None),
    ("Ten tickles/tentacles", "ten tickles", "tentacles", "ˌtɛnˈtɪkəlz", None),
    ("tales/tails", "tales", "tails", None, None),
    ("sphere/fear", "sphere", "fear", None, None),
    ("Pho Queue/fuck you", "pho queue", "fuck you", "ˌfoʊˈkju", "ˌfʌˈkju"),
    ("sine/sign", "sine", "sign", None, None),
    ("reptile dysfunction/erectile", "reptile", "erectile", None, None),
    ("Unique/you sneak", "unique", "you sneak", None, "juˈnik"),
    (
        "Miniappleless Minisoda",
        "miniappleless minisoda",
        "minneapolis minnesota",
        "ˌmɪniˈæpəlɪsˌmɪnɪˈsoʊdə",
        "ˌmɪniˈæpəlɪsˌmɪnɪˈsoʊɾə",
    ),
    ("Romainder/remainder", "romainder", "remainder", "ɹɪˈmeɪndɚ", None),
    ("treble/trouble", "treble", "trouble", None, None),
    ("sea shells/C shells", "sea shells", "c shells", "ˈsiˌʃɛlz", "ˈsiˌʃɛlz"),
    ("Hebrews/he brews", "hebrews", "he brews", None, "hiˈbɹuz"),
    ("Dill emma/dilemma", "dill emma", "dilemma", "ˈdɪlˌɛmə", None),
    ("roverdose/overdose", "roverdose", "overdose", "ˈɹoʊvɚˌdoʊs", None),
    ("curryous/curious", "curryous", "curious", "ˈkɝiəs", None),
    ("ill eagle/illegal", "ill eagle", "illegal", "ˌɪlˈiɡəl", None),
    (
        "staid lion/state line",
        "staid lion",
        "state line",
        "ˌsteɪdˈlaɪən",
        "ˌsteɪtˈlaɪn",
    ),
    (
        "immortal porpoises/immoral purposes",
        "immortal porpoises",
        "immoral purposes",
        "ɪˈmɔɹtəlˈpɔɹpəsɪz",
        "ɪˈmɔɹəlˈpɝpəsɪz",
    ),
    (
        "Lycansubscribe/like and subscribe",
        "lycansubscribe",
        "like and subscribe",
        "ˌlaɪkənˈsʌbskɹaɪb",
        "ˌlaɪkəndˈsʌbskɹaɪb",
    ),
    (
        "Yes Oui Sí Ja/yes we see ya",
        "yes oui si ja",
        "yes we see ya",
        "ˈjɛsˈwiˈsiˈjɑ",
        "ˈjɛsˈwiˈsiˈjɑ",
    ),
    ("loched/locked", "loched", "locked", "lɑkt", None),
    ("Optical Aleutians/illusions", "aleutians", "illusions", "əˈluʃənz", "ɪˈluʒənz"),
    ("a shoe/achoo", "a shoe", "achoo", "əˈʃu", "əˈtʃu"),
    ("secede/succeed", "secede", "succeed", None, None),
    ("hens meet/ends meet", "hens meet", "ends meet", "ˈhɛnzˌmit", "ˈɛndzˌmit"),
    ("Everest/ever rest", "everest", "ever rest", None, "ˈɛvɚˌɹɛst"),
    ("leased/least", "leased", "least", None, None),
    ("ICU/I see you", "icu", "i see you", "ˌaɪˈsiˈju", "ˌaɪˈsiˈju"),
    ("fizzicist/physicist", "fizzicist", "physicist", "ˈfɪzɪsɪst", None),
    ("roll model/role model", "roll", "role", None, None),
    ("no bell prize/Nobel prize", "no bell", "nobel", "ˌnoʊˈbɛl", None),
    (
        "final front ear/frontier",
        "final front ear",
        "final frontier",
        "ˌfaɪnəlˈfɹʌntˌɪɹ",
        "ˌfaɪnəlˌfɹʌnˈtɪɹ",
    ),
    (
        "Labracadabrador",
        "labracadabrador",
        "labrador",
        "ˌlæbɹəkəˈdæbɹədɔɹ",
        "ˈlæbɹəˌdɔɹ",
    ),
    (
        "hatchet counts/count chickens",
        "hatchet your counts",
        "count your chickens",
        "ˈhætʃɪtjɚˈkaʊnts",
        "ˈkaʊntjɚˈtʃɪkɪnz",
    ),
    (
        "Premature edraculation",
        "edraculation",
        "ejaculation",
        "ˌɛdɹækjəˈleɪʃən",
        "ɪˌdʒækjəˈleɪʃən",
    ),
    ("Trajeudi/tragédie", "trajeudi", "tragedie", "tɹæˈʒʌdi", "ˈtɹædʒədi"),
    ("Fed Ex/fed exes", "fed ex", "fed exes", "ˈfɛdˌɛks", "ˈfɛdˌɛksɪz"),
    ("Gyroscope/gyros", "gyros cope", "gyroscope", "ˈdʒaɪɹoʊsˌkoʊp", None),
    (
        "oncologist/on call",
        "oncologist",
        "on call a gist",
        "ɑnˈkɑlədʒɪst",
        "ˌɑnˈkɔləˌdʒɪst",
    ),
    ("XORpheus/Orpheus", "xorpheus", "orpheus", "ˈzɔɹfiəs", "ˈɔɹfiəs"),
    ("dehydrated/de-hydra-ted", "dehydrated", "de hydra ted", None, "ˌdiˈhaɪdɹəˌtɪd"),
    ("in visa ble/invisible", "in visa ble", "invisible", "ɪnˈvɪzəbəl", None),
    ("Sb/stingy (antimony)", "sb", "stingy", "ˌɛsˈbi", "ˈstɪndʒi"),
    ("Bovine/bow vine", "bovine", "bow vine", None, "ˈboʊˌvaɪn"),
    (
        "no bun in ten did/nobody intended",
        "no bun in ten did",
        "nobody intended",
        "ˌnoʊˈbʌnɪnˈtɛnˌdɪd",
        "ˈnoʊbɑdiɪnˈtɛndɪd",
    ),
    (
        "Kaiser Temporariente",
        "temporariente",
        "permanente",
        "ˌtɛmpɚəˈɹiɛnteɪ",
        "ˌpɝməˈnɛnteɪ",
    ),
    (
        "lack of pies/space (spoonerism)",
        "lack of pies",
        "lack of space",
        "ˌlækəvˈpaɪz",
        "ˌlækəvˈspeɪs",
    ),
    ("kicking apps/ass", "apps", "ass", "æps", "æs"),
]

# Semantic pun count (for overall stats only)
N_SEMANTIC = 82

# ── Oronym subset: multi-word puns where beam search should find the parse ──
# (label, single_form_ipa, expected_multi_words)
# The IPA is the "collapsed" single-string pronunciation; beam search should
# find the multi-word parse from the English dictionary.
ORONYMS: list[tuple[str, str, list[str]]] = [
    ("resisting a rest/arrest", "ɹɪˈzɪstɪŋəˈɹɛst", ["resisting", "a", "rest"]),
    ("in tents/intense", "ɪnˈtɛns", ["in", "tents"]),
    ("two in tents/too intense", "tuɪnˈtɛns", ["to", "in", "tents"]),
    ("infant tree/infantry", "ˈɪnfəntɹi", ["infant", "tree"]),
    ("in da skies/in disguise", "ɪndɪsˈɡaɪz", ["in", "disguise"]),
    ("Sham Pain/champagne", "ʃæmˈpeɪn", ["sham", "pain"]),
    ("ill eagle/illegal", "ɪˈliɡəl", ["ill", "eagle"]),
    ("staid lion/state line", "steɪtˈlaɪn", ["staid", "lion"]),
    (
        "immortal porpoises/immoral purposes",
        "ɪˈmɔɹəlˈpɝpəsɪz",
        ["immortal", "porpoises"],
    ),
    ("ten tickles/tentacles", "ˈtɛntəkəlz", ["ten", "tickles"]),
    (
        "no bun in ten did/nobody intended",
        "ˈnoʊbɑdiɪnˈtɛndɪd",
        ["no", "bun", "in", "ten", "did"],
    ),
    ("soda pressing/so depressing", "soʊdɪˈpɹɛsɪŋ", ["soda", "pressing"]),
    ("soda lighted/so delighted", "soʊdɪˈlaɪtɪd", ["soda", "lighted"]),
    ("no bell/Nobel", "noʊˈbɛl", ["no", "bell"]),
    ("Hebrews/he brews", "hiˈbɹuz", ["he", "brews"]),
    ("Dill emma/dilemma", "dɪˈlɛmə", ["dill", "emma"]),
    ("a shoe/achoo", "əˈtʃu", ["a", "shoe"]),
    ("carrion/carry on", "ˈkæɹiˌɑn", ["carry", "on"]),
    ("catastrophe/cat has trophy", "kəˈtæstɹəfi", ["cat", "has", "trophy"]),
    ("aloha/a lower", "əˈloʊɚ", ["a", "lower"]),
    ("super fish oil/superficial", "ˌsupɚˈfɪʃəl", ["super", "fish", "oil"]),
    ("therapist (bonus)", "ˈθɛɹəpɪst", ["the", "rapist"]),
    ("ice cream (classic)", "aɪsˈkɹim", ["i", "scream"]),
]

# ── Bonus oronym mining seeds ───────────────────────────────────────────────
ORONYM_SEEDS: list[tuple[str, str]] = [
    ("therapist", "ˈθɛɹəpɪst"),
    ("nowhere", "ˈnoʊˌwɛɹ"),
    ("atonement", "əˈtoʊnmənt"),
    ("understand", "ˌʌndɚˈstænd"),
    ("tonight", "təˈnaɪt"),
    ("together", "təˈɡɛðɚ"),
    ("disease", "dɪˈziz"),
    ("announce", "əˈnaʊns"),
    ("carpet", "ˈkɑɹpɪt"),
    ("example", "ɪɡˈzæmpəl"),
    ("apartment", "əˈpɑɹtmənt"),
    ("assault", "əˈsɔlt"),
    ("inspire", "ɪnˈspaɪɹ"),
    ("explain", "ɪkˈspleɪn"),
    ("adore", "əˈdɔɹ"),
    ("catastrophe", "kəˈtæstɹəfi"),
    ("intense", "ɪnˈtɛns"),
    ("illegal", "ɪˈliɡəl"),
    ("champagne", "ʃæmˈpeɪn"),
    ("infantry", "ˈɪnfəntɹi"),
    ("arcade", "ɑɹˈkeɪd"),
    ("disguise", "dɪsˈɡaɪz"),
    ("dialogue", "ˈdaɪəˌlɔɡ"),
    ("diploma", "dɪˈploʊmə"),
    ("forfeit", "ˈfɔɹfɪt"),
    ("paradise", "ˈpæɹəˌdaɪs"),
    ("season", "ˈsizən"),
    ("selfish", "ˈsɛlfɪʃ"),
    ("warship", "ˈwɔɹˌʃɪp"),
    ("kidnap", "ˈkɪdˌnæp"),
]


print("Source: https://wstyler.ucsd.edu/puns/")
print(f"Total puns classified: {len(PHONETIC_PUNS) + N_SEMANTIC}")
print(f"  Phonetic: {len(PHONETIC_PUNS)}")
print(f"  Semantic: {N_SEMANTIC}")

THRESHOLD = 0.35
tp = fn = 0
results = []

for label, pw, tw, m_pw, m_tw in PHONETIC_PUNS:
    ipas_a = ipa_all(pw, m_pw)
    ipas_b = ipa_all(tw, m_tw)
    best_a, best_b, d = best_ipa_pair(ipas_a, ipas_b)

    hit = d is not None and d <= THRESHOLD
    if hit:
        tp += 1
    else:
        fn += 1
    results.append((label, pw, tw, best_a, best_b, d, hit))

results.sort(key=lambda r: (r[5] is None, r[5] if r[5] is not None else 999))

print(f"\n{'=' * 90}")
print(f"PART 1: PUN-PAIR DISTANCES  (best-of-N matching, threshold={THRESHOLD})")
print(f"{'=' * 90}")
evaluated = 0
for label, pw, tw, ipa_a, ipa_b, d, hit in results:
    s = "✓" if hit else "✗"
    ds = f"{d:.3f}" if d is not None else "N/A  "
    ipas = f"/{ipa_a or '?'}/ vs /{ipa_b or '?'}/"
    print(f"  {s} d={ds}  {label:50s} {ipas}")
    if d is not None:
        evaluated += 1

no_ipa = len(PHONETIC_PUNS) - evaluated
precision = tp / (tp + 0) if tp > 0 else 0.0  # no FP mechanism
recall_eval = tp / evaluated if evaluated > 0 else 0.0
recall_all = tp / len(PHONETIC_PUNS)

print(f"\n  Phonetic puns total:          {len(PHONETIC_PUNS)}")
print(f"  Successfully evaluated:       {evaluated}")
print(f"  No IPA available:             {no_ipa}")
print(f"  True positives (d≤{THRESHOLD}):     {tp}")
print(f"  False negatives (d>{THRESHOLD}):     {fn}")
print(f"  Precision (of flagged):       {precision:.3f}")
print(f"  Recall (over evaluated):      {recall_eval:.3f}")
print(f"  Recall (over all phonetic):   {recall_all:.3f}")

# Identify false negatives for Part 3
false_negatives = [
    (label, pw, tw, ipa_a, ipa_b, d)
    for label, pw, tw, ipa_a, ipa_b, d, hit in results
    if not hit
]


print(f"\n{'=' * 90}")
print("PART 2a: ORONYM PRODUCT-OF-VARIANTS  (itertools.product over pronunciations)")
print(f"{'=' * 90}")

product_tp = 0
product_total = 0
threshold = 0.35
for label, source_ipa, expected_words in ORONYMS:
    product_total += 1
    d, combo, joined = multi_word_best_ned(source_ipa, expected_words)
    if d is not None and d <= threshold:
        product_tp += 1
        s = "✓"
    else:
        s = "✗"
    words_str = " + ".join(expected_words)
    if d is not None and combo is not None:
        combo_str = " + ".join(combo)
        print(f"  {s} d={d:.3f}  {label:45s}  /{joined}/")
        if combo_str != words_str:
            print(f"           prons: {combo_str}")
    else:
        missing = [w for w in expected_words if not ipa_all(w)]
        print(f"  {s} d=N/A   {label:45s}  missing: {missing}")

print(
    f"\n  Product-match recall (d≤{threshold}): {product_tp}/{product_total} = {product_tp / product_total:.3f}"
)


print(f"\n{'=' * 90}")
print("Building English PreTokenizedDictionary...")
print(f"{'=' * 90}")
t0 = time.time()
ptd = cached_pretokenize_dictionary(
    lambda: g2p.pdict,
    eng_spec,
    lang="eng_us",
    min_tokens=2,
)
print(f"  PTD built in {time.time() - t0:.1f}s  ({len(ptd)} entries)")


print(f"\n{'=' * 90}")
print(f"PART 2: ORONYM BEAM SEARCH RECALL")
print(f"{'=' * 90}")

oronym_tp = 0
oronym_fn = 0
oronym_results = []

for label, source_ipa, expected_words in ORONYMS:
    source_tokens = eng_spec.ipa_tokenizer(source_ipa)
    if not source_tokens:
        oronym_results.append((label, expected_words, [], False, "no tokens"))
        oronym_fn += 1
        continue

    try:
        beam_results = beam_search_segmentation(
            source_tokens,
            source_features=eng.PHONEME_FEATURES,
            target_ptd=ptd,
            target_spec=eng_spec,
            target_features=eng.PHONEME_FEATURES,
            beam_width=30,
            top_k=20,
            max_words=5,
            max_distance=0.40,
            min_target_tokens=1,
        )
    except Exception as e:
        oronym_results.append((label, expected_words, [], False, str(e)))
        oronym_fn += 1
        continue

    # Check if any result contains the expected multi-word parse (fuzzy word match)
    expected_set = set(w.lower() for w in expected_words)
    found = False
    best_match = None
    for br in beam_results:
        result_set = set(w.lower() for w in br.words)
        # Check overlap — allow partial match (at least 2 of the expected words)
        overlap = expected_set & result_set
        if len(overlap) >= min(2, len(expected_set)):
            found = True
            best_match = br
            break

    if found:
        oronym_tp += 1
    else:
        oronym_fn += 1

    top_results = beam_results[:5] if beam_results else []
    oronym_results.append(
        (
            label,
            expected_words,
            top_results,
            found,
            (
                f"best: {best_match.words} d={best_match.distance:.3f}"
                if best_match
                else "not found"
            ),
        )
    )

for label, expected, top, found, note in oronym_results:
    s = "✓" if found else "✗"
    exp_str = " + ".join(expected)
    print(f"\n  {s} {label}")
    print(f"    expected: {exp_str}")
    print(f"    {note}")
    if top:
        for br in top[:3]:
            print(
                f"      {' + '.join(br.words):35s} /{br.glued_ipa}/  d={br.distance:.3f}"
            )

oronym_total = len(ORONYMS)
oronym_recall = oronym_tp / oronym_total if oronym_total > 0 else 0
print(f"\n  Oronym recall: {oronym_tp}/{oronym_total} = {oronym_recall:.3f}")


print(f"\n{'=' * 90}")
print("PART 3: FALSE NEGATIVE RECOVERY VIA BEAM SEARCH")
print(f"{'=' * 90}")

recovered = 0
for label, pw, tw, ipa_a, ipa_b, d in false_negatives:
    # Try beam search on the target IPA to see if alternative parsing helps
    test_ipa = ipa_a or ipa_b
    if not test_ipa:
        print(f"\n  ✗ {label:50s}  no IPA to test")
        continue

    source_tokens = eng_spec.ipa_tokenizer(test_ipa)
    if not source_tokens:
        print(f"\n  ✗ {label:50s}  no tokens")
        continue

    try:
        beam_results = beam_search_segmentation(
            source_tokens,
            source_features=eng.PHONEME_FEATURES,
            target_ptd=ptd,
            target_spec=eng_spec,
            target_features=eng.PHONEME_FEATURES,
            beam_width=30,
            top_k=10,
            max_words=4,
            max_distance=0.45,
            min_target_tokens=1,
        )
    except Exception as e:
        print(f"\n  ✗ {label:50s}  error: {e}")
        continue

    if beam_results:
        best = beam_results[0]
        is_better = best.distance < (d if d is not None else 999)
        marker = "↑" if is_better else "→"
        if is_better:
            recovered += 1
        was = f"d={d:.3f}" if d is not None else "d=N/A"
        print(f"\n  {marker} {label:50s}  (was {was})")
        for br in beam_results[:3]:
            print(
                f"      {' + '.join(br.words):35s} /{br.glued_ipa}/  d={br.distance:.3f}"
            )
    else:
        print(f"\n  ✗ {label:50s}  no beam results")

print(f"\n  Recovered: {recovered}/{len(false_negatives)}")


print(f"\n{'=' * 90}")
print("PART 4: BONUS ORONYM MINING")
print(f"{'=' * 90}")

for word, ipa in ORONYM_SEEDS:
    source_tokens = eng_spec.ipa_tokenizer(ipa)
    if not source_tokens:
        continue

    try:
        beam_results = beam_search_segmentation(
            source_tokens,
            source_features=eng.PHONEME_FEATURES,
            target_ptd=ptd,
            target_spec=eng_spec,
            target_features=eng.PHONEME_FEATURES,
            beam_width=30,
            top_k=10,
            max_words=4,
            max_distance=0.35,
            min_target_tokens=1,
        )
    except Exception:
        continue

    # Filter out single-word results that are just the word itself
    multi = [br for br in beam_results if len(br.words) > 1]
    if multi:
        print(f"\n  {word} /{ipa}/:")
        for br in multi[:5]:
            print(
                f"    {' + '.join(br.words):35s} /{br.glued_ipa}/  d={br.distance:.3f}"
            )
