"""Tests for the universal Panphon-based feature system."""

from __future__ import annotations

import pytest

from phone_similarity.universal_features import (
    PANPHON_FEATURE_NAMES,
    UniversalFeatureEncoder,
    encode_phoneme,
    merge_inventories,
    phoneme_feature_dict,
    universal_phoneme_distance,
)


class TestEncode:
    def test_known_phoneme_p(self):
        vec = encode_phoneme("p")
        assert isinstance(vec, tuple)
        assert len(vec) == 24
        assert vec[0] == -1
        assert vec[1] == -1
        assert vec[2] == 1

    def test_known_phoneme_a(self):
        vec = encode_phoneme("a")
        assert len(vec) == 24
        assert vec[0] == 1
        assert vec[1] == 1
        assert vec[2] == -1

    def test_different_phonemes_differ(self):
        assert encode_phoneme("p") != encode_phoneme("b")

    def test_zero_vector_for_unknown(self):
        vec = encode_phoneme("@")
        assert all(v == 0 for v in vec)

    def test_diacritic_stripping_fallback(self):
        vec = encode_phoneme("p\u0361")
        assert len(vec) == 24
        assert vec[2] == 1


class TestFeatureDict:
    def test_keys_are_feature_names(self):
        fd = phoneme_feature_dict("a")
        assert set(fd.keys()) == set(PANPHON_FEATURE_NAMES)

    def test_values_are_int(self):
        fd = phoneme_feature_dict("t")
        assert all(isinstance(v, int) for v in fd.values())
        assert all(v in (-1, 0, 1) for v in fd.values())


class TestUniversalPhonemeDistance:
    def test_same_phoneme_zero(self):
        assert universal_phoneme_distance("p", "p") == 0.0

    def test_different_phonemes_positive(self):
        assert universal_phoneme_distance("p", "a") > 0.0

    def test_within_unit_interval(self):
        d = universal_phoneme_distance("s", "z")
        assert 0.0 <= d <= 1.0

    def test_voicing_contrast_small(self):
        assert universal_phoneme_distance("p", "b") < 0.2

    def test_maximally_different(self):
        assert universal_phoneme_distance("a", "k") > 0.3


class TestConvertInventory:
    def test_converts_sample_inventory(self):
        inv = {
            "p": {"voiced": False, "place": "bilabial"},
            "b": {"voiced": True, "place": "bilabial"},
        }
        result = UniversalFeatureEncoder.convert_inventory(inv)
        assert set(result.keys()) == {"p", "b"}
        assert set(result["p"].keys()) == set(PANPHON_FEATURE_NAMES)
        assert result["p"]["voi"] == -1
        assert result["b"]["voi"] == 1


class TestMergeInventories:
    def test_shared_phoneme_gets_single_representation(self):
        inv_a = {"e": {"height": "mid", "front": True}}
        inv_b = {"e": {"long": False, "nasal": False}}
        merged = merge_inventories(inv_a, inv_b)
        assert "e" in merged
        assert set(merged["e"].keys()) == set(PANPHON_FEATURE_NAMES)

    def test_disjoint_inventories_union(self):
        merged = merge_inventories({"p": {"voiced": False}}, {"ʃ": {"voiced": False}})
        assert "p" in merged
        assert "ʃ" in merged

    def test_no_inventories(self):
        assert merge_inventories() == {}

    def test_real_language_merge_consistency(self):
        from phone_similarity.language import LANGUAGES

        eng = LANGUAGES["eng_us"].PHONEME_FEATURES
        fra = LANGUAGES["fra"].PHONEME_FEATURES
        merged_ab = merge_inventories(eng, fra)
        merged_ba = merge_inventories(fra, eng)
        shared = set(eng) & set(fra)
        assert len(shared) > 0
        for ph in shared:
            assert merged_ab[ph] == merged_ba[ph]


class TestCrossLanguageIntegration:
    @pytest.fixture(scope="class")
    def eng_fra_specs(self):
        from phone_similarity.bit_array_specification import BitArraySpecification
        from phone_similarity.language import LANGUAGES

        specs = {}
        features = {}
        for mod_name in ("eng_us", "fra"):
            lang = LANGUAGES[mod_name]
            consonants = set(lang.PHONEME_FEATURES.keys()) - lang.VOWELS_SET
            specs[mod_name] = BitArraySpecification(
                vowels=lang.VOWELS_SET,
                consonants=consonants,
                features=lang.FEATURES,
                features_per_phoneme=lang.PHONEME_FEATURES,
            )
            features[mod_name] = lang.PHONEME_FEATURES
        return specs, features

    def test_compare_cross_language_edit(self, eng_fra_specs):
        from phone_similarity.cross_language import compare_cross_language

        specs, features = eng_fra_specs
        ipa = {"eng_us": "wɔːtəɹ", "fra": "o"}
        results = compare_cross_language(ipa, specs, features, metric="edit")
        assert len(results) == 1
        key = ("eng_us", "fra")
        assert key in results
        assert 0.0 <= results[key] <= 1.0

    def test_intersecting_spec_tokenizes_both_languages(self, eng_fra_specs):
        from phone_similarity.intersecting_bit_array_specification import (
            IntersectingBitArraySpecification,
        )

        specs, _features = eng_fra_specs
        merged = IntersectingBitArraySpecification([specs["eng_us"], specs["fra"]])
        assert merged._vowels == specs["eng_us"]._vowels | specs["fra"]._vowels
        assert merged._consonants == specs["eng_us"]._consonants | specs["fra"]._consonants
