"""
Tests for longer-word languages (agglutinative, polysynthetic, compounds).

Languages like Turkish, Finnish, German, Hungarian, Swahili, and
polysynthetic languages (e.g. Inuktitut) produce very long phonological
words. This suite verifies that:

1. Edit distance scales sensibly with word length.
2. Normalised distance stays in [0, 1] even for very long words.
3. Syllabification handles long words correctly.
4. Similar long words are closer than dissimilar long words.
"""

import pytest

from phone_similarity.primitives import (
    feature_edit_distance,
    normalised_feature_edit_distance,
)
from phone_similarity.syllable import syllabify
from phone_similarity.universal_features import UniversalFeatureEncoder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feats(phonemes):
    return {p: UniversalFeatureEncoder.feature_dict(p) for p in set(phonemes)}


_GENERIC_VOWELS = frozenset("ɑaeiouæɛɪɔʊəɤüøyɯʌɜɵ")


# ---------------------------------------------------------------------------
# Long Turkish words (agglutinative)
# ---------------------------------------------------------------------------


class TestTurkishLongWords:
    """Turkish agglutination produces very long words via suffixation."""

    # çekoslovakyalılaştıramadıklarımızdan (one of the longest Turkish words)
    # Simplified IPA segments:
    LONG_WORD = list("tʃekoslovɑkjɑlɯlɑʃtɯɾɑmɑdɯklɑɾɯmɯzdɑn")
    # Shorter related form: çekoslovakyalı
    SHORT_WORD = list("tʃekoslovɑkjɑlɯ")
    # Completely unrelated: ev (house)
    TINY_WORD = list("ev")

    def test_normalised_in_range(self):
        feats = _feats(self.LONG_WORD + self.SHORT_WORD)
        d = normalised_feature_edit_distance(self.LONG_WORD, self.SHORT_WORD, feats)
        assert 0.0 <= d <= 1.0

    def test_long_identical(self):
        feats = _feats(self.LONG_WORD)
        assert normalised_feature_edit_distance(self.LONG_WORD, self.LONG_WORD, feats) == 0.0

    def test_related_closer_than_unrelated(self):
        """A morphologically related long form should be closer than an unrelated short word."""
        feats = _feats(self.LONG_WORD + self.SHORT_WORD + self.TINY_WORD)
        d_related = normalised_feature_edit_distance(self.LONG_WORD, self.SHORT_WORD, feats)
        d_unrelated = normalised_feature_edit_distance(self.LONG_WORD, self.TINY_WORD, feats)
        assert d_related < d_unrelated

    def test_raw_distance_scales_with_length(self):
        """Raw (non-normalised) edit distance should be larger for more edits."""
        feats = _feats(self.LONG_WORD + self.SHORT_WORD + self.TINY_WORD)
        d_short = feature_edit_distance(self.SHORT_WORD, self.TINY_WORD, feats)
        d_long = feature_edit_distance(self.LONG_WORD, self.TINY_WORD, feats)
        assert d_long > d_short


# ---------------------------------------------------------------------------
# German compounds
# ---------------------------------------------------------------------------


class TestGermanCompounds:
    """German compound words can be very long."""

    # Donaudampfschifffahrtsgesellschaftskapitän
    # Simplified IPA:
    COMPOUND = list("doːnaʊdɑmpfʃɪfːɑːɾtsɡəzɛlʃɑftskapitɛːn")
    # Shorter part: Donau (Danube)
    PART = list("doːnaʊ")
    # Unrelated: Brot (bread)
    UNRELATED = list("bʁoːt")

    def test_normalised_in_range(self):
        feats = _feats(self.COMPOUND + self.PART)
        d = normalised_feature_edit_distance(self.COMPOUND, self.PART, feats)
        assert 0.0 <= d <= 1.0

    def test_compound_prefix_closer(self):
        """The prefix portion (Donau) should be closer to the compound than an unrelated word."""
        feats = _feats(self.COMPOUND + self.PART + self.UNRELATED)
        d_prefix = normalised_feature_edit_distance(self.COMPOUND, self.PART, feats)
        d_other = normalised_feature_edit_distance(self.COMPOUND, self.UNRELATED, feats)
        assert d_prefix < d_other

    def test_long_word_symmetry(self):
        feats = _feats(self.COMPOUND + self.PART)
        d_ab = feature_edit_distance(self.COMPOUND, self.PART, feats)
        d_ba = feature_edit_distance(self.PART, self.COMPOUND, feats)
        assert d_ab == pytest.approx(d_ba)


# ---------------------------------------------------------------------------
# Hungarian agglutination
# ---------------------------------------------------------------------------


class TestHungarianLongWords:
    """Hungarian is agglutinative with vowel harmony."""

    # megszentségteleníthetetlenségeskedéseitekért
    # Simplified IPA:
    LONG = list("mɛɡsɛntsːeːɡtɛlɛniːthɛtɛtlɛnʃeːɡɛʃkɛdeːʃɛitɛkeːɾt")
    SHORT = list("mɛɡsɛnt")
    UNRELATED = list("hɑːz")  # ház (house)

    def test_normalised_in_range(self):
        feats = _feats(self.LONG + self.SHORT)
        d = normalised_feature_edit_distance(self.LONG, self.SHORT, feats)
        assert 0.0 <= d <= 1.0

    def test_prefix_closer(self):
        feats = _feats(self.LONG + self.SHORT + self.UNRELATED)
        d_prefix = normalised_feature_edit_distance(self.LONG, self.SHORT, feats)
        d_other = normalised_feature_edit_distance(self.LONG, self.UNRELATED, feats)
        assert d_prefix < d_other


# ---------------------------------------------------------------------------
# Swahili (Bantu agglutination)
# ---------------------------------------------------------------------------


class TestSwahiliLongWords:
    """Swahili/Bantu languages build long verb forms via prefix/suffix stacking."""

    # hatutakaopendana (we will not love each other)
    LONG_VERB = list("hɑtutɑkɑopendɑnɑ")
    # penda (love - root)
    ROOT = list("pendɑ")
    # nyumba (house - unrelated)
    UNRELATED = list("ɲumbɑ")

    def test_normalised_in_range(self):
        feats = _feats(self.LONG_VERB + self.ROOT)
        d = normalised_feature_edit_distance(self.LONG_VERB, self.ROOT, feats)
        assert 0.0 <= d <= 1.0

    def test_root_closer_than_unrelated(self):
        """The verb root should be detectable as closer to the inflected form."""
        feats = _feats(self.LONG_VERB + self.ROOT + self.UNRELATED)
        d_root = normalised_feature_edit_distance(self.LONG_VERB, self.ROOT, feats)
        d_other = normalised_feature_edit_distance(self.LONG_VERB, self.UNRELATED, feats)
        assert d_root < d_other


# ---------------------------------------------------------------------------
# Syllabification of long words
# ---------------------------------------------------------------------------


class TestLongWordSyllabification:
    """Syllabifier should handle long words without errors and produce
    reasonable syllable counts."""

    @pytest.mark.parametrize(
        "label, tokens, min_syllables",
        [
            (
                "Turkish long word",
                list("tʃekoslovɑkjɑlɯlɑʃtɯɾɑmɑdɯklɑɾɯmɯzdɑn"),
                5,
            ),
            (
                "German compound",
                list("doːnaʊdɑmpfʃɪfːɑːɾtsɡəzɛlʃɑftskapitɛːn"),
                4,
            ),
            (
                "Hungarian long word",
                list("mɛɡsɛntsːeːɡtɛlɛniːthɛtɛtlɛnʃeːɡɛʃkɛdeːʃɛitɛkeːɾt"),
                5,
            ),
            (
                "Swahili verb",
                list("hɑtutɑkɑopendɑnɑ"),
                5,
            ),
        ],
    )
    def test_minimum_syllable_count(self, label, tokens, min_syllables):
        syls = syllabify(tokens, _GENERIC_VOWELS)
        assert len(syls) >= min_syllables, (
            f"{label}: expected >= {min_syllables} syllables, got {len(syls)}"
        )

    def test_empty_word(self):
        syls = syllabify([], _GENERIC_VOWELS)
        assert len(syls) == 0

    def test_single_vowel(self):
        syls = syllabify(["ɑ"], _GENERIC_VOWELS)
        assert len(syls) == 1


# ---------------------------------------------------------------------------
# Stress: edit distance should not be dominated by word length
# ---------------------------------------------------------------------------


class TestNormalisedDistanceScaling:
    """Normalised distance should remain meaningful regardless of word length."""

    def test_identical_long_words(self):
        """Identical long words should have distance 0."""
        w = list("tʃekoslovɑkjɑlɯlɑʃtɯɾɑmɑdɯklɑɾɯmɯzdɑn")
        feats = _feats(w)
        assert normalised_feature_edit_distance(w, w, feats) == 0.0

    def test_one_phoneme_change_in_long_word(self):
        """Changing one phoneme in a long word should produce a small normalised distance."""
        w1 = list("tʃekoslovɑkjɑlɯlɑʃtɯɾɑmɑdɯklɑɾɯmɯzdɑn")
        w2 = list(w1)
        w2[0] = "s"  # change tʃ -> s
        feats = _feats(w1 + w2)
        d = normalised_feature_edit_distance(w1, w2, feats)
        assert 0 < d < 0.1, "One change in a long word should give a small normalised distance"

    def test_completely_different_long_words(self):
        """Two completely different long words should have high distance."""
        w1 = list("tʃekoslovɑkjɑlɯlɑʃtɯɾɑmɑ")
        w2 = list("ɲumbɑɲumbɑɲumbɑɲumbɑɲumbɑ")
        feats = _feats(w1 + w2)
        d = normalised_feature_edit_distance(w1, w2, feats)
        assert d > 0.2, "Completely different long words should have substantial distance"
        assert d <= 1.0
