"""Cross-language phonological distance and syllabification tests.

Covers: Finnish length contrasts, Mandarin/Thai/Vietnamese/Yoruba tones,
Turkish/German/Hungarian/Swahili long words.
"""

import pytest

from phone_similarity.primitives import (
    feature_edit_distance,
    normalised_feature_edit_distance,
)
from phone_similarity.syllable import syllabify
from phone_similarity.universal_features import (
    UniversalFeatureEncoder,
    universal_phoneme_distance,
)

_V = frozenset("ɑaeiouæɛɪɔʊəɤüøyɯʌɜɵ")


def _f(phones):
    return {p: UniversalFeatureEncoder.feature_dict(p) for p in set(phones)}


# -- data: (label, base, variant) where variant adds length or changes tone --

LENGTH_PAIRS = [
    ("Finnish V: tuli/tuuli", list("tuli"), list("tuuli")),
    ("Finnish C: kɑtu/kɑttu", list("kɑtu"), list("kɑttu")),
    ("Thai V: kan/kaan", list("kan"), list("kaan")),
]

TONE_SEQS = [
    ("Mandarin high/fall", ["m", "a", "˥˥"], ["m", "a", "˥˩"]),
    ("Thai rice/white", ["kʰ", "aː", "w", "˨˩"], ["kʰ", "aː", "w", "˦˥"]),
    ("Vietnamese ghost/but", ["m", "a", "˧"], ["m", "a", "˨˩"]),
    ("Yoruba husband/hoe", ["ɔ", "˧", "k", "ɔ", "˥"], ["ɔ", "˩", "k", "ɔ", "˩"]),
]

LONG_WORDS = [
    (
        "Turkish",
        list("tʃekoslovɑkjɑlɯlɑʃtɯɾɑmɑdɯklɑɾɯmɯzdɑn"),
        list("tʃekoslovɑkjɑlɯ"),
        list("ev"),
    ),
    ("German", list("doːnaʊdɑmpfʃɪfːɑːɾtsɡəzɛlʃɑftskapitɛːn"), list("doːnaʊ"), list("bʁoːt")),
    (
        "Hungarian",
        list("mɛɡsɛntsːeːɡtɛlɛniːthɛtɛtlɛnʃeːɡɛʃkɛdeːʃɛitɛkeːɾt"),
        list("mɛɡsɛnt"),
        list("hɑːz"),
    ),
    ("Swahili", list("hɑtutɑkɑopendɑnɑ"), list("pendɑ"), list("ɲumbɑ")),
]

SYLLABLE_DATA = [
    ("Finnish tɑ.lo", list("tɑlo"), 2),
    ("Finnish tuu.li", list("tuuli"), 2),
    ("Finnish kɑt.tu", list("kɑttu"), 2),
    ("Mandarin zhōngguó", ["ʈʂ", "o", "ŋ", "k", "u", "o"], 2),
    ("Turkish long", list("tʃekoslovɑkjɑlɯlɑʃtɯɾɑmɑdɯklɑɾɯmɯzdɑn"), 5),
    ("German compound", list("doːnaʊdɑmpfʃɪfːɑːɾtsɡəzɛlʃɑftskapitɛːn"), 4),
    ("Swahili verb", list("hɑtutɑkɑopendɑnɑ"), 5),
]


class TestLengthContrasts:
    @pytest.mark.parametrize("label, base, long", LENGTH_PAIRS)
    def test_nonzero(self, label, base, long):
        assert normalised_feature_edit_distance(base, long, _f(base + long)) > 0

    def test_combined_gt_single(self):
        """Both long V + geminate C > either alone."""
        b, lv, gc, both = list("kɑtu"), list("kɑɑtu"), list("kɑttu"), list("kɑɑttu")
        f = _f(b + both)
        assert feature_edit_distance(b, both, f) > feature_edit_distance(b, lv, f)
        assert feature_edit_distance(b, both, f) > feature_edit_distance(b, gc, f)

    def test_ipa_length_mark(self):
        assert 0 < universal_phoneme_distance("ɑ", "ɑː") < universal_phoneme_distance("t", "k")


class TestTones:
    @pytest.mark.parametrize("label, a, b", TONE_SEQS)
    def test_computable(self, label, a, b):
        assert normalised_feature_edit_distance(a, b, _f(a + b)) >= 0

    def test_identical_zero(self):
        s = ["m", "a", "˥˥"]
        assert normalised_feature_edit_distance(s, s, _f(s)) == 0.0


class TestLongWords:
    @pytest.mark.parametrize("lang, full, prefix, unrelated", LONG_WORDS)
    def test_normalised_range(self, lang, full, prefix, unrelated):
        assert 0.0 <= normalised_feature_edit_distance(full, prefix, _f(full + prefix)) <= 1.0

    @pytest.mark.parametrize("lang, full, prefix, unrelated", LONG_WORDS)
    def test_prefix_closer(self, lang, full, prefix, unrelated):
        f = _f(full + prefix + unrelated)
        assert normalised_feature_edit_distance(
            full, prefix, f
        ) < normalised_feature_edit_distance(full, unrelated, f)

    def test_single_change_small(self):
        w = list("tʃekoslovɑkjɑlɯlɑʃtɯɾɑmɑdɯklɑɾɯmɯzdɑn")
        w2 = list(w)
        w2[0] = "s"
        assert 0 < normalised_feature_edit_distance(w, w2, _f(w + w2)) < 0.1


class TestSyllabification:
    @pytest.mark.parametrize("label, tokens, min_syls", SYLLABLE_DATA)
    def test_min_syllables(self, label, tokens, min_syls):
        assert len(syllabify(tokens, _V)) >= min_syls
