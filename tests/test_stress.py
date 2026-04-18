"""Tests for stress preservation mode.

Covers ``CleanConfig``, ``clean_phones`` with ``preserve_stress``,
``extract_stress_marks``, stress-aware ``syllabify()``, and the
``stressed_syllable()`` / ``stress_pattern()`` helpers.
"""

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
    syllable_count,
)

# -----------------------------------------------------------------------
# CleanConfig
# -----------------------------------------------------------------------


class TestCleanConfig:
    """Verify CleanConfig immutability and strip_all property."""

    def test_default_strips_all(self):
        cfg = CleanConfig()
        assert cfg.strip_stress is True
        assert cfg.strip_length is True
        assert cfg.strip_liaison is True
        assert cfg.strip_all is True

    def test_preserve_stress_not_strip_all(self):
        cfg = CleanConfig(strip_stress=False)
        assert cfg.strip_all is False

    def test_prebuilt_strip_all(self):
        assert STRIP_ALL.strip_all is True

    def test_prebuilt_preserve_stress(self):
        assert PRESERVE_STRESS.strip_stress is False
        assert PRESERVE_STRESS.strip_length is True

    def test_prebuilt_preserve_length(self):
        assert PRESERVE_LENGTH.strip_length is False
        assert PRESERVE_LENGTH.strip_stress is True

    def test_prebuilt_preserve_all(self):
        assert PRESERVE_ALL.strip_stress is False
        assert PRESERVE_ALL.strip_length is False
        assert PRESERVE_ALL.strip_liaison is False


# -----------------------------------------------------------------------
# clean_phones backward compatibility
# -----------------------------------------------------------------------


class TestCleanPhonesBackwardCompat:
    """Default behaviour must be identical to the original implementation."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("ˈhɛloʊ", "hɛloʊ"),
            ("ˌsɛkənˌdɛɹi", "sɛkəndɛɹi"),
            ("aːbˑc‿d", "abcd"),
            ("kæt", "kæt"),  # no markers → pass through
            ("", ""),
        ],
    )
    def test_default_strips_all(self, raw, expected):
        assert clean_phones(raw) == expected

    def test_nfkd_normalisation(self):
        # U+00E9 (precomposed) → e + combining acute
        result = clean_phones("\u00e9")
        assert "e" in result


# -----------------------------------------------------------------------
# clean_phones with preserve_stress flag
# -----------------------------------------------------------------------


class TestCleanPhonesPreserveStress:
    """Test the preserve_stress convenience parameter."""

    def test_preserve_stress_keeps_markers(self):
        assert clean_phones("ˈhɛloʊ", preserve_stress=True) == "ˈhɛloʊ"

    def test_preserve_stress_still_strips_length(self):
        result = clean_phones("ˈhɛːloʊ", preserve_stress=True)
        assert "ˈ" in result
        assert "ː" not in result
        assert result == "ˈhɛloʊ"

    def test_preserve_stress_still_strips_liaison(self):
        result = clean_phones("ˈa‿b", preserve_stress=True)
        assert "ˈ" in result
        assert "‿" not in result

    def test_secondary_stress_preserved(self):
        result = clean_phones("ˌhɛˈloʊ", preserve_stress=True)
        assert "ˌ" in result
        assert "ˈ" in result


# -----------------------------------------------------------------------
# clean_phones with CleanConfig
# -----------------------------------------------------------------------


class TestCleanPhonesConfig:
    """Test fine-grained CleanConfig."""

    def test_config_overrides_preserve_stress(self):
        # config takes precedence over preserve_stress
        cfg = CleanConfig(strip_stress=True)
        result = clean_phones("ˈhɛloʊ", preserve_stress=True, config=cfg)
        assert "ˈ" not in result

    def test_preserve_length_strips_stress(self):
        result = clean_phones("ˈaːb", config=PRESERVE_LENGTH)
        assert "ˈ" not in result
        assert "ː" in result
        assert result == "aːb"

    def test_preserve_all_keeps_everything(self):
        raw = "ˈhɛːˌloʊ‿ˑz"
        result = clean_phones(raw, config=PRESERVE_ALL)
        # All suprasegmentals retained (only NFKD applied)
        assert "ˈ" in result
        assert "ˌ" in result
        assert "ː" in result
        assert "ˑ" in result
        assert "‿" in result

    def test_strip_all_config_matches_default(self):
        raw = "ˈhɛːˌloʊ"
        assert clean_phones(raw) == clean_phones(raw, config=STRIP_ALL)

    def test_no_nfkd(self):
        cfg = CleanConfig(nfkd=False, strip_stress=False, strip_length=False, strip_liaison=False)
        raw = "\u00e9"  # precomposed e-acute
        assert clean_phones(raw, config=cfg) == "\u00e9"

    def test_finnish_length_contrastive(self):
        """Finnish: length is contrastive, stress is predictable (initial)."""
        cfg = CleanConfig(strip_stress=True, strip_length=False)
        result = clean_phones("ˈkɑːtːo", config=cfg)
        assert "ˈ" not in result
        assert "ː" in result
        assert result == "kɑːtːo"


# -----------------------------------------------------------------------
# extract_stress_marks
# -----------------------------------------------------------------------


class TestExtractStressMarks:
    """Verify stress marker extraction with cleaned index mapping."""

    def test_single_primary(self):
        marks = extract_stress_marks("ˈhɛloʊ")
        assert marks == [(0, "primary")]

    def test_primary_and_secondary(self):
        marks = extract_stress_marks("ˈhɛˌloʊ")
        assert marks == [(0, "primary"), (2, "secondary")]

    def test_no_stress(self):
        marks = extract_stress_marks("hɛloʊ")
        assert marks == []

    def test_stress_mid_word(self):
        # ˈ before the 3rd clean character
        marks = extract_stress_marks("bəˈnænə")
        assert marks == [(2, "primary")]

    def test_multiple_secondary(self):
        marks = extract_stress_marks("ˌɪntɹəˌdʌkʃən")
        assert len(marks) == 2
        assert marks[0] == (0, "secondary")
        assert marks[1][1] == "secondary"

    def test_adjacent_markers(self):
        # pathological: two markers in a row
        marks = extract_stress_marks("ˈˌab")
        assert marks == [(0, "primary"), (0, "secondary")]


# -----------------------------------------------------------------------
# Syllable stress field
# -----------------------------------------------------------------------


class TestSyllableStress:
    """Verify the stress field on Syllable dataclass."""

    def test_default_stress_is_none(self):
        syl = Syllable(onset=("k",), nucleus=("æ",), coda=("t",))
        assert syl.stress is None
        assert syl.is_stressed is False

    def test_primary_stress(self):
        syl = Syllable(onset=("k",), nucleus=("æ",), coda=("t",), stress="primary")
        assert syl.stress == "primary"
        assert syl.is_stressed is True

    def test_secondary_stress(self):
        syl = Syllable(onset=(), nucleus=("ə",), coda=(), stress="secondary")
        assert syl.stress == "secondary"
        assert syl.is_stressed is True

    def test_rime_property(self):
        syl = Syllable(onset=("k",), nucleus=("æ",), coda=("t",))
        assert syl.rime == ("æ", "t")

    def test_rime_no_coda(self):
        syl = Syllable(onset=("k",), nucleus=("æ",), coda=())
        assert syl.rime == ("æ",)


# -----------------------------------------------------------------------
# syllabify with stress_marks
# -----------------------------------------------------------------------


class TestSyllabifyWithStress:
    """Test stress assignment through syllabify()."""

    @pytest.fixture()
    def vowels(self):
        return frozenset({"æ", "ɛ", "ɪ", "ə", "oʊ", "ʌ", "i", "a", "u"})

    def test_primary_stress_on_first_syllable(self, vowels):
        # "ˈhæpi" → h.æ.p.i → [Syllable(h, æ, p), Syllable(_, i, _)]
        tokens = ["h", "æ", "p", "i"]
        marks = [(0, "primary")]  # stress at token 0 → first syllable
        syls = syllabify(tokens, vowels, stress_marks=marks)
        assert syls[0].stress == "primary"
        assert syls[1].stress is None

    def test_secondary_stress_on_second_syllable(self, vowels):
        tokens = ["b", "ə", "n", "æ", "n", "ə"]
        marks = [(3, "secondary")]  # stress at token 3 → second syllable (næ)
        syls = syllabify(tokens, vowels, stress_marks=marks)
        assert syls[0].stress is None
        assert syls[1].stress == "secondary"

    def test_no_marks_no_stress(self, vowels):
        tokens = ["k", "æ", "t"]
        syls = syllabify(tokens, vowels)
        for syl in syls:
            assert syl.stress is None

    def test_empty_marks_no_stress(self, vowels):
        tokens = ["k", "æ", "t"]
        syls = syllabify(tokens, vowels, stress_marks=[])
        for syl in syls:
            assert syl.stress is None

    def test_primary_wins_over_secondary(self, vowels):
        # Both markers on same syllable — primary should win
        tokens = ["k", "æ", "t"]
        marks = [(0, "secondary"), (1, "primary")]
        syls = syllabify(tokens, vowels, stress_marks=marks)
        assert syls[0].stress == "primary"

    def test_multi_syllable_stress(self, vowels):
        # "ˈbænəˌnə" → ban.ə.nə
        tokens = ["b", "æ", "n", "ə", "n", "ə"]
        marks = [(0, "primary"), (4, "secondary")]
        syls = syllabify(tokens, vowels, stress_marks=marks)
        assert syls[0].stress == "primary"
        assert syls[2].stress == "secondary"


# -----------------------------------------------------------------------
# stressed_syllable helper
# -----------------------------------------------------------------------


class TestStressedSyllable:
    """Test the stressed_syllable() lookup function."""

    def test_finds_primary(self):
        syls = [
            Syllable(("b",), ("ə",), (), stress=None),
            Syllable(("n",), ("æ",), ("n",), stress="primary"),
            Syllable((), ("ə",), (), stress=None),
        ]
        result = stressed_syllable(syls)
        assert result is syls[1]

    def test_finds_secondary(self):
        syls = [
            Syllable((), ("ɪ",), ("n",), stress="secondary"),
            Syllable(("t",), ("ɛ",), ("n",), stress=None),
        ]
        result = stressed_syllable(syls, kind="secondary")
        assert result is syls[0]

    def test_returns_none_when_absent(self):
        syls = [
            Syllable(("k",), ("æ",), ("t",), stress=None),
        ]
        assert stressed_syllable(syls) is None

    def test_empty_list(self):
        assert stressed_syllable([]) is None


# -----------------------------------------------------------------------
# stress_pattern helper
# -----------------------------------------------------------------------


class TestStressPattern:
    """Test numeric stress pattern generation."""

    def test_basic_pattern(self):
        syls = [
            Syllable(("b",), ("ɪ",), (), stress="secondary"),
            Syllable(("n",), ("æ",), (), stress=None),
            Syllable(("n",), ("ə",), (), stress="primary"),
        ]
        assert stress_pattern(syls) == "201"

    def test_all_unstressed(self):
        syls = [
            Syllable(("k",), ("æ",), ("t",)),
            Syllable((), ("ə",), ()),
        ]
        assert stress_pattern(syls) == "00"

    def test_single_stressed(self):
        syls = [Syllable(("k",), ("æ",), ("t",), stress="primary")]
        assert stress_pattern(syls) == "1"

    def test_empty(self):
        assert stress_pattern([]) == ""


# -----------------------------------------------------------------------
# syllable_count
# -----------------------------------------------------------------------


class TestSyllableCount:
    """Test syllable_count convenience function."""

    def test_count(self):
        syls = [
            Syllable(("b",), ("æ",), ()),
            Syllable(("n",), ("ə",), ()),
            Syllable(("n",), ("ə",), ()),
        ]
        assert syllable_count(syls) == 3

    def test_empty(self):
        assert syllable_count([]) == 0


# -----------------------------------------------------------------------
# Integration: extract_stress_marks + syllabify pipeline
# -----------------------------------------------------------------------


class TestStressPipeline:
    """End-to-end tests: raw IPA → extract marks → clean → tokenize → syllabify."""

    @pytest.fixture()
    def eng_vowels(self):
        return frozenset({"æ", "ɛ", "ɪ", "ə", "oʊ", "ʌ", "i", "u", "a", "ɑ", "ɔ", "aɪ"})

    def test_hello_pipeline(self, eng_vowels):
        """ˈhɛloʊ → extract marks → clean → tokenize → syllabify with stress."""
        raw = "ˈhɛloʊ"
        marks = extract_stress_marks(raw)
        assert marks == [(0, "primary")]

        cleaned = clean_phones(raw)
        assert cleaned == "hɛloʊ"

        # Manual tokenization for test simplicity
        tokens = ["h", "ɛ", "l", "oʊ"]
        syls = syllabify(tokens, eng_vowels, stress_marks=marks)

        assert len(syls) == 2
        assert syls[0].stress == "primary"
        assert syls[1].stress is None
        assert stress_pattern(syls) == "10"

    def test_banana_pipeline(self, eng_vowels):
        """bəˈnænə → second syllable stressed."""
        raw = "bəˈnænə"
        marks = extract_stress_marks(raw)
        assert marks == [(2, "primary")]

        cleaned = clean_phones(raw)
        assert cleaned == "bənænə"

        tokens = ["b", "ə", "n", "æ", "n", "ə"]
        syls = syllabify(tokens, eng_vowels, stress_marks=marks)

        assert len(syls) == 3
        assert syls[0].stress is None
        assert syls[1].stress == "primary"
        assert syls[2].stress is None
        assert stress_pattern(syls) == "010"

    def test_understand_pipeline(self, eng_vowels):
        """ˌʌndɹ̩ˈstænd → secondary on first, primary on last."""
        raw = "ˌʌndəˈstænd"
        marks = extract_stress_marks(raw)
        assert marks[0] == (0, "secondary")
        assert marks[1][1] == "primary"

        clean_phones(raw)
        tokens = ["ʌ", "n", "d", "ə", "s", "t", "æ", "n", "d"]
        syls = syllabify(tokens, eng_vowels, stress_marks=marks)

        primary = stressed_syllable(syls)
        assert primary is not None
        assert "æ" in primary.nucleus

        secondary = stressed_syllable(syls, kind="secondary")
        assert secondary is not None
        assert "ʌ" in secondary.nucleus

    def test_preserve_stress_clean_phones(self, eng_vowels):
        """clean_phones with preserve_stress=True retains markers."""
        raw = "ˈkæt"
        cleaned = clean_phones(raw, preserve_stress=True)
        assert cleaned == "ˈkæt"

        # And the default strips it
        cleaned_default = clean_phones(raw)
        assert cleaned_default == "kæt"
