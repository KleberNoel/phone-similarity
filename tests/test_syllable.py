"""Tests for the syllable segmenter."""

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


class TestSonorityScale:
    @pytest.fixture()
    def scale(self):
        return SonorityScale()

    def test_sonority_ranks(self, scale):
        expected = {
            "a": RANK_VOWEL,
            "i": RANK_VOWEL,
            "u": RANK_VOWEL,
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
            "p": RANK_STOP,
            "b": RANK_STOP,
            "t": RANK_STOP,
            "d": RANK_STOP,
        }
        for phoneme, rank in expected.items():
            assert scale.rank(phoneme) == rank

    def test_rank_tokens(self, scale):
        ranks = scale.rank_tokens(["p", "a", "t"])
        assert ranks == [RANK_STOP, RANK_VOWEL, RANK_STOP]

    def test_extra_ranks_override(self):
        scale = SonorityScale(extra_ranks={"ʔ": 0})
        assert scale.rank("ʔ") == 0


class TestSyllable:
    def test_phonemes_property(self):
        s = Syllable(onset=("k",), nucleus=("æ",), coda=("t",))
        assert s.phonemes == ("k", "æ", "t")

    def test_len(self):
        s = Syllable(onset=("s", "t", "r"), nucleus=("ɪ",), coda=("ŋ", "z"))
        assert len(s) == 6


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
        result = self._syllabify(["p", "a"], {"a"}, seg, scale)
        assert len(result) == 1
        assert result[0] == Syllable(onset=("p",), nucleus=("a",), coda=())

    def test_cvc_syllable(self, seg, scale):
        result = self._syllabify(["p", "a", "t"], {"a"}, seg, scale)
        assert len(result) == 1
        assert result[0] == Syllable(onset=("p",), nucleus=("a",), coda=("t",))

    def test_cvcv_mop_split(self, seg, scale):
        result = self._syllabify(["p", "a", "t", "a"], {"a"}, seg, scale)
        assert len(result) == 2
        assert result[0] == Syllable(onset=("p",), nucleus=("a",), coda=())
        assert result[1] == Syllable(onset=("t",), nucleus=("a",), coda=())

    def test_cvccv_cluster_split(self, seg, scale):
        result = self._syllabify(["p", "a", "n", "t", "a"], {"a"}, seg, scale)
        assert len(result) == 2
        assert result[0] == Syllable(onset=("p",), nucleus=("a",), coda=("n",))
        assert result[1] == Syllable(onset=("t",), nucleus=("a",), coda=())

    def test_ccvcv_onset_cluster(self, seg, scale):
        result = self._syllabify(["p", "l", "a", "t", "a"], {"a"}, seg, scale)
        assert len(result) == 2
        assert result[0] == Syllable(onset=("p", "l"), nucleus=("a",), coda=())
        assert result[1] == Syllable(onset=("t",), nucleus=("a",), coda=())

    def test_sibilant_appendix(self, seg, scale):
        result = self._syllabify(["a", "s", "t", "r", "a"], {"a"}, seg, scale)
        assert len(result) == 2
        assert result[0] == Syllable(onset=(), nucleus=("a",), coda=())
        assert result[1] == Syllable(onset=("s", "t", "r"), nucleus=("a",), coda=())

    def test_sibilant_appendix_disabled(self, seg_no_sibilant, scale):
        result = self._syllabify(["a", "s", "t", "r", "a"], {"a"}, seg_no_sibilant, scale)
        assert len(result) == 2
        assert result[0] == Syllable(onset=(), nucleus=("a",), coda=("s",))
        assert result[1] == Syllable(onset=("t", "r"), nucleus=("a",), coda=())

    def test_empty_input(self, seg, scale):
        assert self._syllabify([], {"a"}, seg, scale) == []

    def test_diphthong_nucleus(self, seg, scale):
        result = self._syllabify(["b", "a", "i"], {"a", "i"}, seg, scale)
        assert len(result) == 1
        assert result[0] == Syllable(onset=("b",), nucleus=("a", "i"), coda=())

    def test_three_syllable_word(self, seg, scale):
        tokens = ["b", "a", "n", "a", "n", "a"]
        result = self._syllabify(tokens, {"a"}, seg, scale)
        assert len(result) == 3


class TestSyllabifyFunction:
    VOWELS = frozenset({"a", "e", "i", "o", "u", "æ", "ɪ", "ɛ", "ʌ", "ɑ", "ɔ", "ʊ"})

    def test_simple_word(self):
        result = syllabify(["k", "æ", "t"], self.VOWELS)
        assert len(result) == 1
        assert result[0].onset == ("k",)
        assert result[0].nucleus == ("æ",)
        assert result[0].coda == ("t",)

    def test_two_syllable_word(self):
        result = syllabify(["p", "ɛ", "p", "ɪ"], self.VOWELS)
        assert len(result) == 2
        assert result[0] == Syllable(onset=("p",), nucleus=("ɛ",), coda=())
        assert result[1] == Syllable(onset=("p",), nucleus=("ɪ",), coda=())

    def test_complex_onset(self):
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
    VOWELS = frozenset({"a", "e", "i", "o", "u"})

    def test_batch_basic(self):
        token_lists = [["p", "a"], ["p", "a", "t", "a"], ["a"]]
        results = batch_syllabify(token_lists, self.VOWELS)
        assert len(results) == 3
        assert len(results[0]) == 1
        assert len(results[1]) == 2
        assert len(results[2]) == 1

    def test_batch_empty(self):
        assert batch_syllabify([], frozenset({"a"})) == []


class TestLanguageIntegration:
    def test_english_water(self):
        vowels = frozenset({"ɔ", "ɚ", "ə", "a", "e", "i", "o", "u", "ɪ", "æ"})
        tokens = ["w", "ɔ", "t", "ɚ"]
        result = syllabify(tokens, vowels)
        assert len(result) == 2
        assert result[0].onset == ("w",)
        assert result[0].nucleus == ("ɔ",)
        assert result[1].onset == ("t",)
        assert result[1].nucleus == ("ɚ",)

    def test_french_bonjour(self):
        vowels = frozenset({"ɔ", "u", "a", "e", "i", "o"})
        tokens = ["b", "ɔ", "ʒ", "u", "ʁ"]
        result = syllabify(tokens, vowels)
        assert len(result) == 2
        assert result[0].onset == ("b",)
        assert result[1].onset == ("ʒ",)
        assert result[1].nucleus == ("u",)

    def test_german_strasse(self):
        vowels = frozenset({"a", "aː", "ə", "e", "i", "o", "u"})
        tokens = ["ʃ", "t", "ʁ", "aː", "s", "ə"]
        result = syllabify(tokens, vowels)
        assert len(result) == 2
        assert result[0].nucleus == ("aː",)
        assert result[1].onset == ("s",)
        assert result[1].nucleus == ("ə",)
