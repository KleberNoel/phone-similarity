"""Real-world pun / homophonic pair tests exercising beam search and feature distance."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from phone_similarity.beam_search import beam_search_segmentation
from phone_similarity.pretokenize import PreTokenizedDictionary


@dataclass(frozen=True)
class PunCase:
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


INTERLINGUAL_ENG_FRA = [
    PunCase(
        id="humpty_dumpty_l1",
        source_lang="eng-us",
        target_lang="fra",
        source_text="Humpty Dumpty",
        target_text="Un petit d'un petit",
        source_ipa="hʌmptɪ dʌmptɪ",
        target_ipa="œ̃ pəti dœ̃ pəti",
        category="homophonic_translation",
        n_words_source=2,
        n_words_target=4,
        notes="From 'Mots D'Heures: Gousses, Rames'.",
        max_expected_distance=0.50,
    ),
    PunCase(
        id="poor_john",
        source_lang="fra",
        target_lang="fra",
        source_text="pauvre Jean",
        target_text="pauvres gens",
        source_ipa="povʁ ʒɑ̃",
        target_ipa="povʁ ʒɑ̃",
        category="intralingual_mondegreen",
        n_words_source=2,
        n_words_target=2,
        notes="French mondegreen: perfect homophone pair.",
        max_expected_distance=0.05,
    ),
]

MONDEGREENS_ENG = [
    PunCase(
        id="recognize_speech",
        source_lang="eng-us",
        target_lang="eng-us",
        source_text="recognize speech",
        target_text="wreck a nice beach",
        source_ipa="ɹɛkənaɪz spiːtʃ",
        target_ipa="ɹɛk ə naɪs biːtʃ",
        category="intralingual_mondegreen",
        n_words_source=2,
        n_words_target=4,
        notes="Classic speech recognition joke. 2 words -> 4 words.",
        max_expected_distance=0.20,
    ),
]

EGGCORNS_ENG = [
    PunCase(
        id="bated_breath",
        source_lang="eng-us",
        target_lang="eng-us",
        source_text="bated breath",
        target_text="baited breath",
        source_ipa="beɪtɪd bɹɛθ",
        target_ipa="beɪtɪd bɹɛθ",
        category="eggcorn",
        n_words_source=2,
        n_words_target=2,
        notes="Perfect homophone eggcorn.",
        max_expected_distance=0.05,
    ),
    PunCase(
        id="egg_acorn",
        source_lang="eng-us",
        target_lang="eng-us",
        source_text="acorn",
        target_text="egg corn",
        source_ipa="eɪkɔɹn",
        target_ipa="ɛɡ kɔɹn",
        category="eggcorn",
        n_words_source=1,
        n_words_target=2,
        notes="The eponymous eggcorn.",
        max_expected_distance=0.30,
    ),
]

SORAMIMI = [
    PunCase(
        id="dragostea_din_tei",
        source_lang="ron",
        target_lang="jpn",
        source_text="Vrei sa pleci dar nu ma, nu ma iei",
        target_text="bei sa beishu darou nomanoma iei",
        source_ipa="vreɪ sə pletʃʲ dar nu mə nu mə jeɪ",
        target_ipa="beɪ sa beɪʃɨ daɾɨ nomanoma ieɪ",
        category="soramimi",
        n_words_source=9,
        n_words_target=6,
        notes="O-Zone 'Dragostea Din Tei' reinterpreted as Japanese.",
        max_expected_distance=0.50,
    ),
]

BILINGUAL_PUNS_MISC = [
    PunCase(
        id="wario_warui",
        source_lang="jpn",
        target_lang="eng-us",
        source_text="warui",
        target_text="Wario",
        source_ipa="waɾɯi",
        target_ipa="wɑːɹioʊ",
        category="interlingual",
        n_words_source=1,
        n_words_target=1,
        notes="Nintendo bilingual portmanteau.",
        max_expected_distance=0.40,
    ),
]

ALL_PUNS: list[PunCase] = (
    INTERLINGUAL_ENG_FRA + MONDEGREENS_ENG + EGGCORNS_ENG + SORAMIMI + BILINGUAL_PUNS_MISC
)

_SHARED_FEATURES: dict[str, dict[str, object]] = {
    "p": {"voiced": False, "manner": "plosive", "labial": True},
    "b": {"voiced": True, "manner": "plosive", "labial": True},
    "t": {"voiced": False, "manner": "plosive", "place": "alveolar"},
    "d": {"voiced": True, "manner": "plosive", "place": "alveolar"},
    "k": {"voiced": False, "manner": "plosive", "place": "velar"},
    "ɡ": {"voiced": True, "manner": "plosive", "place": "velar"},
    "f": {"voiced": False, "manner": "fricative", "labial": True},
    "v": {"voiced": True, "manner": "fricative", "labial": True},
    "s": {"voiced": False, "manner": "fricative", "place": "alveolar"},
    "z": {"voiced": True, "manner": "fricative", "place": "alveolar"},
    "ʃ": {"voiced": False, "manner": "fricative", "place": "postalveolar"},
    "ʒ": {"voiced": True, "manner": "fricative", "place": "postalveolar"},
    "θ": {"voiced": False, "manner": "fricative", "place": "dental"},
    "ð": {"voiced": True, "manner": "fricative", "place": "dental"},
    "h": {"voiced": False, "manner": "fricative", "place": "glottal"},
    "m": {"voiced": True, "manner": "nasal", "labial": True},
    "n": {"voiced": True, "manner": "nasal", "place": "alveolar"},
    "ŋ": {"voiced": True, "manner": "nasal", "place": "velar"},
    "l": {"voiced": True, "manner": "lateral", "place": "alveolar"},
    "ɹ": {"voiced": True, "manner": "approximant", "place": "alveolar"},
    "ʁ": {"voiced": True, "manner": "fricative", "place": "uvular"},
    "ɾ": {"voiced": True, "manner": "tap", "place": "alveolar"},
    "w": {"voiced": True, "manner": "approximant", "labial": True},
    "j": {"voiced": True, "manner": "approximant", "place": "palatal"},
    "tʃ": {"voiced": False, "manner": "affricate", "place": "postalveolar"},
    "dʒ": {"voiced": True, "manner": "affricate", "place": "postalveolar"},
    "i": {"high": True, "front": True, "round": False},
    "ɪ": {"high": True, "front": True, "round": False, "lax": True},
    "e": {"mid": True, "front": True, "round": False},
    "ɛ": {"mid-low": True, "front": True, "round": False},
    "æ": {"low": True, "front": True, "round": False},
    "a": {"low": True, "central": True, "round": False},
    "ə": {"mid": True, "central": True, "round": False},
    "ɐ": {"low": True, "central": True, "round": False},
    "ʌ": {"mid-low": True, "back": True, "round": False},
    "ɑ": {"low": True, "back": True, "round": False},
    "ɒ": {"low": True, "back": True, "round": True},
    "o": {"mid": True, "back": True, "round": True},
    "ɔ": {"mid-low": True, "back": True, "round": True},
    "u": {"high": True, "back": True, "round": True},
    "ʊ": {"high": True, "back": True, "round": True, "lax": True},
    "œ": {"mid-low": True, "front": True, "round": True},
    "ɜ": {"mid": True, "central": True, "round": False},
    "ɨ": {"high": True, "central": True, "round": True},
    "ɯ": {"high": True, "central": True, "round": False},
    "œ̃": {"mid-low": True, "front": True, "round": True, "nasal": True},
    "ɛ̃": {"mid-low": True, "front": True, "round": False, "nasal": True},
    "ɑ̃": {"low": True, "back": True, "round": False, "nasal": True},
    "aɪ": {"low": True, "front": True, "diphthong": True},
    "eɪ": {"mid": True, "front": True, "diphthong": True},
    "oʊ": {"mid": True, "back": True, "round": True, "diphthong": True},
    "aʊ": {"low": True, "back": True, "diphthong": True},
    "ɔɪ": {"mid-low": True, "back": True, "round": True, "diphthong": True},
}

_SORTED_PHONEMES: list[str] = sorted(_SHARED_FEATURES.keys(), key=len, reverse=True)


def _tokenize_ipa(ipa_str: str, features: dict) -> list[str]:
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
            i += 1
    return tokens


_FIVE_PUNS = [
    next(p for p in ALL_PUNS if p.category == "homophonic_translation"),
    next(p for p in ALL_PUNS if p.category == "intralingual_mondegreen"),
    next(p for p in ALL_PUNS if p.category == "eggcorn"),
    next(p for p in ALL_PUNS if p.category == "soramimi"),
    next(p for p in ALL_PUNS if p.category == "interlingual"),
]


@pytest.fixture
def pun_spec():
    from phone_similarity.bit_array_specification import BitArraySpecification

    vowels = {
        "i",
        "ɪ",
        "e",
        "ɛ",
        "æ",
        "a",
        "ə",
        "ɐ",
        "ʌ",
        "ɑ",
        "ɒ",
        "o",
        "ɔ",
        "u",
        "ʊ",
        "œ",
        "ɜ",
        "ɨ",
        "ɯ",
        "œ̃",
        "ɛ̃",
        "ɑ̃",
        "aɪ",
        "eɪ",
        "oʊ",
        "aʊ",
        "ɔɪ",
    }
    consonants = set(_SHARED_FEATURES.keys()) - vowels
    features = {
        "consonant": {"voiced", "manner", "place", "labial"},
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


class TestPunCaseData:
    def test_pun_data_integrity(self):
        valid_categories = {
            "interlingual",
            "intralingual_mondegreen",
            "eggcorn",
            "homophonic_translation",
            "soramimi",
        }
        ids = [p.id for p in ALL_PUNS]
        assert len(ids) == len(set(ids))
        for p in ALL_PUNS:
            assert p.source_ipa
            assert p.target_ipa
            assert p.category in valid_categories
            assert p.n_words_source > 0
            assert p.n_words_target > 0
            assert 0.0 <= p.max_expected_distance <= 1.0


class TestPunFeatureDistance:
    @pytest.mark.parametrize("pun", _FIVE_PUNS, ids=[p.id for p in _FIVE_PUNS])
    def test_pair_distance_within_threshold(self, pun):
        from phone_similarity.primitives import normalised_feature_edit_distance

        source_tokens = _tokenize_ipa(pun.source_ipa, _SHARED_FEATURES)
        target_tokens = _tokenize_ipa(pun.target_ipa, _SHARED_FEATURES)
        if not source_tokens or not target_tokens:
            pytest.skip(f"{pun.id}: could not tokenize IPA")
        dist = normalised_feature_edit_distance(source_tokens, target_tokens, _SHARED_FEATURES)
        assert dist <= pun.max_expected_distance + 0.10, (
            f"{pun.id}: distance {dist:.4f} exceeds threshold {pun.max_expected_distance} + 0.10"
        )


class TestPunBeamSearch:
    @staticmethod
    def _make_ptd_from_pun(pun: PunCase, spec) -> PreTokenizedDictionary:
        target_words = pun.target_text.lower().split()
        target_ipas = pun.target_ipa.split()
        while len(target_ipas) < len(target_words):
            target_ipas.append(target_ipas[-1] if target_ipas else "")
        target_ipas = target_ipas[: len(target_words)]
        entries: list[tuple[str, str, list[str]]] = []
        for word, ipa in zip(target_words, target_ipas, strict=False):
            tokens = _tokenize_ipa(ipa, _SHARED_FEATURES)
            if tokens:
                entries.append((word, ipa, tokens))
        distractors = [
            ("noise", "nɔɪz", ["n", "ɔɪ", "z"]),
            ("frog", "fɹɒɡ", ["f", "ɹ", "ɒ", "ɡ"]),
            ("lamp", "læmp", ["l", "æ", "m", "p"]),
        ]
        entries.extend(distractors)
        return PreTokenizedDictionary.from_entries(entries)

    def test_mondegreen_beam_search(self, pun_spec):
        pun = next(p for p in MONDEGREENS_ENG if p.id == "recognize_speech")
        ptd = self._make_ptd_from_pun(pun, pun_spec)
        source_tokens = _tokenize_ipa(pun.source_ipa, _SHARED_FEATURES)
        if not source_tokens:
            pytest.skip("Could not tokenize source IPA")
        results = beam_search_segmentation(
            source_tokens,
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
        assert len(results) > 0
        assert results[0].distance <= pun.max_expected_distance + 0.15

    def test_eggcorn_beam_search(self, pun_spec):
        pun = next(p for p in EGGCORNS_ENG if p.id == "bated_breath")
        ptd = self._make_ptd_from_pun(pun, pun_spec)
        source_tokens = _tokenize_ipa(pun.source_ipa, _SHARED_FEATURES)
        if not source_tokens:
            pytest.skip("Could not tokenize source IPA")
        results = beam_search_segmentation(
            source_tokens,
            _SHARED_FEATURES,
            ptd,
            pun_spec,
            _SHARED_FEATURES,
            beam_width=10,
            top_k=3,
            max_words=pun.n_words_target + 1,
            max_distance=0.50,
        )
        assert len(results) > 0
        assert results[0].distance <= pun.max_expected_distance + 0.10
