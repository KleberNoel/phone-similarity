"""Tests for the syllable segmenter.

Covers the ``SonorityScale``, ``MaxOnsetSegmenter``, ``Syllable``
dataclass, the high-level ``syllabify()`` and ``batch_syllabify()``
functions, and Cython dispatch when available.
"""

import pytest

from phone_similarity.syllable import (
    RANK_FRICATIVE,
    RANK_GLIDE,
    RANK_LIQUID,
    RANK_NASAL,
    RANK_STOP,
    RANK_VOWEL,
    MaxOnsetSegmenter,
    SonorityScale,
    Syllable,
    batch_syllabify,
    syllabify,
)

# -----------------------------------------------------------------------
# SonorityScale
# -----------------------------------------------------------------------


class TestSonorityScale:
    """Verify sonority rank derivation from universal features."""

    @pytest.fixture()
    def scale(self):
        return SonorityScale()

    def test_sonority_ranks(self, scale):
        """All phoneme classes have correct sonority ranks."""
        expected = {
            "a": RANK_VOWEL,
            "i": RANK_VOWEL,
            "u": RANK_VOWEL,
            "e": RANK_VOWEL,
            "o": RANK_VOWEL,
            "j": RANK_GLIDE,
            "w": RANK_GLIDE,
            "l": RANK_LIQUID,
            "r": RANK_LIQUID,
            "ɹ": RANK_LIQUID,
            "ɾ": RANK_LIQUID,
            "m": RANK_NASAL,
            "n": RANK_NASAL,
            "ŋ": RANK_NASAL,
            "f": RANK_FRICATIVE,
            "v": RANK_FRICATIVE,
            "s": RANK_FRICATIVE,
            "z": RANK_FRICATIVE,
            "ʃ": RANK_FRICATIVE,
            "θ": RANK_FRICATIVE,
            "p": RANK_STOP,
            "b": RANK_STOP,
            "t": RANK_STOP,
            "d": RANK_STOP,
            "k": RANK_STOP,
            "g": RANK_STOP,
        }
        for phoneme, rank in expected.items():
            assert scale.rank(phoneme) == rank, (
                f"{phoneme!r}: expected rank {rank}, got {scale.rank(phoneme)}"
            )

    def test_rank_tokens(self, scale):
        tokens = ["p", "a", "t"]
        ranks = scale.rank_tokens(tokens)
        assert ranks == [RANK_STOP, RANK_VOWEL, RANK_STOP]

    def test_extra_ranks_override(self):
        scale = SonorityScale(extra_ranks={"ʔ": 0})
        assert scale.rank("ʔ") == 0

    def test_build_rank_map(self, scale):
        rm = scale.build_rank_map(["p", "a", "n"])
        assert rm["p"] == RANK_STOP
        assert rm["a"] == RANK_VOWEL
        assert rm["n"] == RANK_NASAL


# -----------------------------------------------------------------------
# Syllable dataclass
# -----------------------------------------------------------------------


class TestSyllable:
    def test_phonemes_property(self):
        s = Syllable(onset=("k",), nucleus=("æ",), coda=("t",))
        assert s.phonemes == ("k", "æ", "t")

    def test_len(self):
        s = Syllable(onset=("s", "t", "r"), nucleus=("ɪ",), coda=("ŋ", "z"))
        assert len(s) == 6

    def test_empty_components(self):
        s = Syllable(onset=(), nucleus=("a",), coda=())
        assert s.phonemes == ("a",)
        assert len(s) == 1


# -----------------------------------------------------------------------
# MaxOnsetSegmenter — core splitting logic
# -----------------------------------------------------------------------


class TestMaxOnsetSegmenter:
    @pytest.fixture()
    def seg(self):
        return MaxOnsetSegmenter()

    @pytest.fixture()
    def seg_no_sibilant(self):
        return MaxOnsetSegmenter(sibilant_appendix=False)

    @pytest.fixture()
    def scale(self):
        return SonorityScale()

    def _syllabify(self, tokens, vowels, seg, scale):
        son = scale.rank_tokens(tokens)
        return seg.syllabify(tokens, frozenset(vowels), son)

    def test_cv_syllable(self, seg, scale):
        # "pa" → [Syllable(onset=('p',), nucleus=('a',), coda=())]
        result = self._syllabify(["p", "a"], {"a"}, seg, scale)
        assert len(result) == 1
        assert result[0] == Syllable(onset=("p",), nucleus=("a",), coda=())

    def test_cvc_syllable(self, seg, scale):
        # "pat" → [Syllable(onset=('p',), nucleus=('a',), coda=('t',))]
        result = (
            self._syllabify(["p", "a", "t"], {"a"}, {"a"}, seg, scale)
            if False
            else self._syllabify(["p", "a", "t"], {"a"}, seg, scale)
        )
        assert len(result) == 1
        assert result[0] == Syllable(onset=("p",), nucleus=("a",), coda=("t",))

    def test_v_syllable(self, seg, scale):
        # "a" → [Syllable(onset=(), nucleus=('a',), coda=())]
        result = self._syllabify(["a"], {"a"}, seg, scale)
        assert len(result) == 1
        assert result[0] == Syllable(onset=(), nucleus=("a",), coda=())

    def test_vc_syllable(self, seg, scale):
        # "an" → [Syllable(onset=(), nucleus=('a',), coda=('n',))]
        result = self._syllabify(["a", "n"], {"a"}, seg, scale)
        assert len(result) == 1
        assert result[0] == Syllable(onset=(), nucleus=("a",), coda=("n",))

    def test_cvcv_mop_split(self, seg, scale):
        # "pata" → pa.ta (MOP: intervocalic /t/ goes to onset of 2nd syllable)
        result = self._syllabify(["p", "a", "t", "a"], {"a"}, seg, scale)
        assert len(result) == 2
        assert result[0] == Syllable(onset=("p",), nucleus=("a",), coda=())
        assert result[1] == Syllable(onset=("t",), nucleus=("a",), coda=())

    def test_cvccv_cluster_split(self, seg, scale):
        # "panta" → pan.ta (MOP: /n/ has higher sonority than /t/, so coda)
        result = self._syllabify(["p", "a", "n", "t", "a"], {"a"}, seg, scale)
        assert len(result) == 2
        assert result[0] == Syllable(onset=("p",), nucleus=("a",), coda=("n",))
        assert result[1] == Syllable(onset=("t",), nucleus=("a",), coda=())

    def test_ccvcv_onset_cluster(self, seg, scale):
        # "plata" → pla.ta (MOP: /pl/ is valid onset, rising sonority)
        result = self._syllabify(["p", "l", "a", "t", "a"], {"a"}, seg, scale)
        assert len(result) == 2
        assert result[0] == Syllable(onset=("p", "l"), nucleus=("a",), coda=())
        assert result[1] == Syllable(onset=("t",), nucleus=("a",), coda=())

    def test_sibilant_appendix(self, seg, scale):
        # "astra" → as.tra (with sibilant appendix: /s/ stays in coda
        # because split_cluster gives /str/ as onset with appendix)
        # Actually: s(2) t(1) r(4) — SSP: t<r ✓, s>t ✗ but sibilant appendix
        # allows /s/ to attach → onset = /str/, coda = empty
        result = self._syllabify(["a", "s", "t", "r", "a"], {"a"}, seg, scale)
        assert len(result) == 2
        assert result[0] == Syllable(onset=(), nucleus=("a",), coda=())
        assert result[1] == Syllable(onset=("s", "t", "r"), nucleus=("a",), coda=())

    def test_sibilant_appendix_disabled(self, seg_no_sibilant, scale):
        # Without appendix: "astra" → as.tra (/s/ stays in coda)
        result = self._syllabify(
            ["a", "s", "t", "r", "a"],
            {"a"},
            seg_no_sibilant,
            scale,
        )
        assert len(result) == 2
        assert result[0] == Syllable(onset=(), nucleus=("a",), coda=("s",))
        assert result[1] == Syllable(onset=("t", "r"), nucleus=("a",), coda=())

    def test_no_vowels(self, seg, scale):
        # Degenerate case: all consonants
        result = self._syllabify(["p", "s", "t"], set(), seg, scale)
        assert len(result) == 1
        assert result[0].nucleus == ()
        assert result[0].onset == ("p", "s", "t")

    def test_empty_input(self, seg, scale):
        result = self._syllabify([], {"a"}, seg, scale)
        assert result == []

    def test_diphthong_nucleus(self, seg, scale):
        # "bai" where both a and i are vowels → single syllable with
        # two-vowel nucleus
        result = self._syllabify(["b", "a", "i"], {"a", "i"}, seg, scale)
        assert len(result) == 1
        assert result[0] == Syllable(onset=("b",), nucleus=("a", "i"), coda=())

    def test_three_syllable_word(self, seg, scale):
        # "banana" → ba.na.na
        tokens = ["b", "a", "n", "a", "n", "a"]
        result = self._syllabify(tokens, {"a"}, seg, scale)
        assert len(result) == 3
        assert result[0] == Syllable(onset=("b",), nucleus=("a",), coda=())
        assert result[1] == Syllable(onset=("n",), nucleus=("a",), coda=())
        assert result[2] == Syllable(onset=("n",), nucleus=("a",), coda=())


# -----------------------------------------------------------------------
# High-level syllabify() + batch_syllabify()
# -----------------------------------------------------------------------


class TestSyllabifyFunction:
    """Test the public ``syllabify()`` entry point."""

    VOWELS = frozenset({"a", "e", "i", "o", "u", "æ", "ɪ", "ɛ", "ʌ", "ɑ", "ɔ", "ʊ"})

    def test_simple_word(self):
        # "kæt" → single syllable
        result = syllabify(["k", "æ", "t"], self.VOWELS)
        assert len(result) == 1
        assert result[0].onset == ("k",)
        assert result[0].nucleus == ("æ",)
        assert result[0].coda == ("t",)

    def test_two_syllable_word(self):
        # "pɛpɪ" → pe.pi
        result = syllabify(["p", "ɛ", "p", "ɪ"], self.VOWELS)
        assert len(result) == 2
        assert result[0] == Syllable(onset=("p",), nucleus=("ɛ",), coda=())
        assert result[1] == Syllable(onset=("p",), nucleus=("ɪ",), coda=())

    def test_complex_onset(self):
        # "stɹɪŋ" → single syllable with /stɹ/ onset (sibilant appendix)
        result = syllabify(["s", "t", "ɹ", "ɪ", "ŋ"], self.VOWELS)
        assert len(result) == 1
        assert result[0].onset == ("s", "t", "ɹ")
        assert result[0].nucleus == ("ɪ",)
        assert result[0].coda == ("ŋ",)

    def test_returns_syllable_objects(self):
        result = syllabify(["p", "a"], frozenset({"a"}))
        assert all(isinstance(s, Syllable) for s in result)

    def test_empty_tokens(self):
        assert syllabify([], frozenset({"a"})) == []


class TestBatchSyllabify:
    """Test ``batch_syllabify()``."""

    VOWELS = frozenset({"a", "e", "i", "o", "u"})

    def test_batch_basic(self):
        token_lists = [
            ["p", "a"],
            ["p", "a", "t", "a"],
            ["a"],
        ]
        results = batch_syllabify(token_lists, self.VOWELS)
        assert len(results) == 3
        assert len(results[0]) == 1  # "pa" = 1 syllable
        assert len(results[1]) == 2  # "pata" = 2 syllables
        assert len(results[2]) == 1  # "a" = 1 syllable

    def test_batch_empty(self):
        assert batch_syllabify([], frozenset({"a"})) == []

    def test_batch_returns_syllable_objects(self):
        results = batch_syllabify([["p", "a"]], self.VOWELS)
        assert all(isinstance(s, Syllable) for s in results[0])


# -----------------------------------------------------------------------
# Integration with language modules
# -----------------------------------------------------------------------


class TestLanguageIntegration:
    """Test syllabification with real language data."""

    def test_english_water(self):
        """Syllabify an English-like IPA transcription."""
        # "wɔtɚ" → wo.tɚ (treating ɚ as vowel)
        vowels = frozenset({"ɔ", "ɚ", "ə", "a", "e", "i", "o", "u", "ɪ", "æ"})
        tokens = ["w", "ɔ", "t", "ɚ"]
        result = syllabify(tokens, vowels)
        assert len(result) == 2
        assert result[0].onset == ("w",)
        assert result[0].nucleus == ("ɔ",)
        assert result[1].onset == ("t",)
        assert result[1].nucleus == ("ɚ",)

    def test_french_bonjour(self):
        """Syllabify a French-like IPA transcription."""
        # "bɔ̃ʒuʁ" → bɔ̃.ʒuʁ (simplified)
        vowels = frozenset({"ɔ", "u", "a", "e", "i", "o"})
        tokens = ["b", "ɔ", "ʒ", "u", "ʁ"]
        result = syllabify(tokens, vowels)
        assert len(result) == 2
        assert result[0].onset == ("b",)
        assert result[0].nucleus == ("ɔ",)
        assert result[1].onset == ("ʒ",)
        assert result[1].nucleus == ("u",)
        assert result[1].coda == ("ʁ",)

    def test_german_strasse(self):
        """Syllabify a German-like IPA: ʃtʁaːsə → ʃtʁaː.sə"""
        vowels = frozenset({"a", "aː", "ə", "e", "i", "o", "u"})
        tokens = ["ʃ", "t", "ʁ", "aː", "s", "ə"]
        result = syllabify(tokens, vowels)
        assert len(result) == 2
        assert result[0].nucleus == ("aː",)
        assert result[1].onset == ("s",)
        assert result[1].nucleus == ("ə",)
