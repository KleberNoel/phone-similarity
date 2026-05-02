"""Tests for stress preservation, extraction, syllabification, and helpers."""

import pytest

from phone_similarity.clean_phones import (
    PRESERVE_ALL,
    PRESERVE_LENGTH,
    PRESERVE_STRESS,
    STRIP_ALL,
    CleanConfig,
    clean_phones,
    extract_stress_marks,
)
from phone_similarity.syllable import (
    Syllable,
    stress_pattern,
    stressed_syllable,
    syllabify,
)


class TestCleanConfig:
    def test_default_strips_all(self):
        cfg = CleanConfig()
        assert cfg.strip_stress is True
        assert cfg.strip_length is True
        assert cfg.strip_liaison is True
        assert cfg.strip_all is True

    def test_prebuilt_preserve_stress(self):
        assert PRESERVE_STRESS.strip_stress is False
        assert PRESERVE_STRESS.strip_length is True

    def test_prebuilt_preserve_all(self):
        assert PRESERVE_ALL.strip_stress is False
        assert PRESERVE_ALL.strip_length is False
        assert PRESERVE_ALL.strip_liaison is False


class TestCleanPhonesBackwardCompat:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("ˈhɛloʊ", "hɛloʊ"),
            ("kæt", "kæt"),
            ("", ""),
        ],
    )
    def test_default_strips_all(self, raw, expected):
        assert clean_phones(raw) == expected


class TestCleanPhonesPreserveStress:
    def test_preserve_stress_keeps_markers(self):
        assert clean_phones("ˈhɛloʊ", preserve_stress=True) == "ˈhɛloʊ"

    def test_preserve_stress_still_strips_length(self):
        result = clean_phones("ˈhɛːloʊ", preserve_stress=True)
        assert "ˈ" in result
        assert "ː" not in result
        assert result == "ˈhɛloʊ"


class TestCleanPhonesConfig:
    def test_preserve_length_strips_stress(self):
        result = clean_phones("ˈaːb", config=PRESERVE_LENGTH)
        assert "ˈ" not in result
        assert "ː" in result
        assert result == "aːb"

    def test_preserve_all_keeps_everything(self):
        raw = "ˈhɛːˌloʊ‿ˑz"
        result = clean_phones(raw, config=PRESERVE_ALL)
        assert "ˈ" in result
        assert "ˌ" in result
        assert "ː" in result
        assert "ˑ" in result
        assert "‿" in result

    def test_strip_all_config_matches_default(self):
        raw = "ˈhɛːˌloʊ"
        assert clean_phones(raw) == clean_phones(raw, config=STRIP_ALL)


class TestExtractStressMarks:
    def test_single_primary(self):
        assert extract_stress_marks("ˈhɛloʊ") == [(0, "primary")]

    def test_primary_and_secondary(self):
        assert extract_stress_marks("ˈhɛˌloʊ") == [(0, "primary"), (2, "secondary")]

    def test_no_stress(self):
        assert extract_stress_marks("hɛloʊ") == []

    def test_stress_mid_word(self):
        assert extract_stress_marks("bəˈnænə") == [(2, "primary")]


class TestSyllableStress:
    def test_primary_stress(self):
        syl = Syllable(onset=("k",), nucleus=("æ",), coda=("t",), stress="primary")
        assert syl.stress == "primary"
        assert syl.is_stressed is True

    def test_rime_property(self):
        syl = Syllable(onset=("k",), nucleus=("æ",), coda=("t",))
        assert syl.rime == ("æ", "t")


class TestSyllabifyWithStress:
    @pytest.fixture()
    def vowels(self):
        return frozenset({"æ", "ɛ", "ɪ", "ə", "oʊ", "ʌ", "i", "a", "u"})

    def test_primary_stress_on_first_syllable(self, vowels):
        tokens = ["h", "æ", "p", "i"]
        marks = [(0, "primary")]
        syls = syllabify(tokens, vowels, stress_marks=marks)
        assert syls[0].stress == "primary"
        assert syls[1].stress is None

    def test_primary_wins_over_secondary(self, vowels):
        tokens = ["k", "æ", "t"]
        marks = [(0, "secondary"), (1, "primary")]
        syls = syllabify(tokens, vowels, stress_marks=marks)
        assert syls[0].stress == "primary"

    def test_multi_syllable_stress(self, vowels):
        tokens = ["b", "æ", "n", "ə", "n", "ə"]
        marks = [(0, "primary"), (4, "secondary")]
        syls = syllabify(tokens, vowels, stress_marks=marks)
        assert syls[0].stress == "primary"
        assert syls[2].stress == "secondary"


class TestStressedSyllable:
    def test_finds_primary(self):
        syls = [
            Syllable(("b",), ("ə",), (), stress=None),
            Syllable(("n",), ("æ",), ("n",), stress="primary"),
            Syllable((), ("ə",), (), stress=None),
        ]
        assert stressed_syllable(syls) is syls[1]

    def test_returns_none_when_absent(self):
        syls = [Syllable(("k",), ("æ",), ("t",), stress=None)]
        assert stressed_syllable(syls) is None


class TestStressPattern:
    def test_basic_pattern(self):
        syls = [
            Syllable(("b",), ("ɪ",), (), stress="secondary"),
            Syllable(("n",), ("æ",), (), stress=None),
            Syllable(("n",), ("ə",), (), stress="primary"),
        ]
        assert stress_pattern(syls) == "201"


class TestStressPipeline:
    @pytest.fixture()
    def eng_vowels(self):
        return frozenset({"æ", "ɛ", "ɪ", "ə", "oʊ", "ʌ", "i", "u", "a", "ɑ", "ɔ", "aɪ"})

    def test_hello_pipeline(self, eng_vowels):
        raw = "ˈhɛloʊ"
        marks = extract_stress_marks(raw)
        assert marks == [(0, "primary")]
        cleaned = clean_phones(raw)
        assert cleaned == "hɛloʊ"
        tokens = ["h", "ɛ", "l", "oʊ"]
        syls = syllabify(tokens, eng_vowels, stress_marks=marks)
        assert len(syls) == 2
        assert syls[0].stress == "primary"
        assert stress_pattern(syls) == "10"

    def test_banana_pipeline(self, eng_vowels):
        raw = "bəˈnænə"
        marks = extract_stress_marks(raw)
        assert marks == [(2, "primary")]
        cleaned = clean_phones(raw)
        assert cleaned == "bənænə"
        tokens = ["b", "ə", "n", "æ", "n", "ə"]
        syls = syllabify(tokens, eng_vowels, stress_marks=marks)
        assert len(syls) == 3
        assert syls[1].stress == "primary"
        assert stress_pattern(syls) == "010"
