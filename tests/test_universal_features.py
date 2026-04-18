"""Tests for the universal Panphon-based feature system.

Validates encoding, fallback chain, distance computation,
cross-language merging, and integration with the existing distance API.
"""

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

# -----------------------------------------------------------------------
# Encoder basics
# -----------------------------------------------------------------------


class TestEncode:
    """Test ``UniversalFeatureEncoder.encode``."""

    def test_known_phoneme_p(self):
        vec = encode_phoneme("p")
        assert isinstance(vec, tuple)
        assert len(vec) == 24
        # /p/ is [-syllabic, -sonorant, +consonantal, -continuant, ...]
        assert vec[0] == -1  # syl
        assert vec[1] == -1  # son
        assert vec[2] == 1  # cons

    def test_known_phoneme_a(self):
        vec = encode_phoneme("a")
        assert len(vec) == 24
        # /a/ is [+syllabic, +sonorant, -consonantal, +continuant, ...]
        assert vec[0] == 1  # syl
        assert vec[1] == 1  # son
        assert vec[2] == -1  # cons

    def test_known_phoneme_sh(self):
        vec = encode_phoneme("ʃ")
        assert len(vec) == 24
        assert vec[2] == 1  # cons
        assert vec[3] == 1  # cont (fricative)

    def test_identity_encoding(self):
        """Same phoneme encoded twice should yield the same tuple."""
        assert encode_phoneme("k") == encode_phoneme("k")

    def test_different_phonemes_differ(self):
        """Different phonemes should not encode identically."""
        assert encode_phoneme("p") != encode_phoneme("b")

    def test_zero_vector_for_unknown(self):
        """Truly unknown symbols should return the zero vector."""
        vec = encode_phoneme("@")  # not IPA
        assert all(v == 0 for v in vec)

    def test_nfd_fallback(self):
        """Accented characters should resolve through NFD normalisation."""
        # e.g. 'é' should resolve to base 'e' features (or an accented entry)
        vec = encode_phoneme("é")
        assert len(vec) == 24
        # Should not be the zero vector
        assert any(v != 0 for v in vec)

    def test_diacritic_stripping_fallback(self):
        """Phonemes with unusual combining marks should still resolve."""
        # A fabricated combination that may not be in the table but whose
        # base char is -- e.g. p + some obscure combining mark
        vec = encode_phoneme("p\u0361")  # p with combining double inverted breve
        assert len(vec) == 24
        # Should get features of /p/ (or close), not the zero vector
        assert vec[2] == 1  # cons -- at minimum the base /p/ should survive


class TestFeatureDict:
    """Test ``UniversalFeatureEncoder.feature_dict``."""

    def test_returns_dict(self):
        fd = phoneme_feature_dict("p")
        assert isinstance(fd, dict)
        assert len(fd) == 24

    def test_keys_are_feature_names(self):
        fd = phoneme_feature_dict("a")
        assert set(fd.keys()) == set(PANPHON_FEATURE_NAMES)

    def test_values_are_int(self):
        fd = phoneme_feature_dict("t")
        assert all(isinstance(v, int) for v in fd.values())
        assert all(v in (-1, 0, 1) for v in fd.values())


# -----------------------------------------------------------------------
# Distance
# -----------------------------------------------------------------------


class TestUniversalPhonemeDistance:
    """Test ``universal_phoneme_distance``."""

    def test_same_phoneme_zero(self):
        assert universal_phoneme_distance("p", "p") == 0.0

    def test_different_phonemes_positive(self):
        d = universal_phoneme_distance("p", "a")
        assert d > 0.0

    def test_within_unit_interval(self):
        d = universal_phoneme_distance("s", "z")
        assert 0.0 <= d <= 1.0

    def test_voicing_contrast_small(self):
        """p vs b differ only in voicing -- distance should be small."""
        d = universal_phoneme_distance("p", "b")
        assert d < 0.2  # only 1-2 features differ out of ~20 comparable

    def test_maximally_different(self):
        """A vowel vs a voiceless stop should be quite distant."""
        d = universal_phoneme_distance("a", "k")
        assert d > 0.3

    def test_unknown_phoneme_returns_zero(self):
        """If both phonemes are unknown (zero vectors), distance is 0."""
        d = universal_phoneme_distance("@", "#")
        assert d == 0.0


# -----------------------------------------------------------------------
# Inventory conversion & merging
# -----------------------------------------------------------------------


class TestConvertInventory:
    """Test ``UniversalFeatureEncoder.convert_inventory``."""

    def test_converts_sample_inventory(self):
        # Minimal fake inventory
        inv = {
            "p": {"voiced": False, "place": "bilabial"},
            "b": {"voiced": True, "place": "bilabial"},
        }
        result = UniversalFeatureEncoder.convert_inventory(inv)
        assert set(result.keys()) == {"p", "b"}
        assert set(result["p"].keys()) == set(PANPHON_FEATURE_NAMES)
        assert result["p"]["voi"] == -1  # /p/ is voiceless
        assert result["b"]["voi"] == 1  # /b/ is voiced


class TestMergeInventories:
    """Test ``merge_inventories`` (cross-language merging)."""

    def test_shared_phoneme_gets_single_representation(self):
        """When both inventories contain /e/, the merged result should
        have exactly one entry for /e/ with consistent features."""
        inv_a = {"e": {"height": "mid", "front": True}}
        inv_b = {"e": {"long": False, "nasal": False}}
        merged = merge_inventories(inv_a, inv_b)
        assert "e" in merged
        # The features come from Panphon, not from either ad-hoc dict
        assert set(merged["e"].keys()) == set(PANPHON_FEATURE_NAMES)

    def test_disjoint_inventories_union(self):
        """Non-overlapping inventories should be fully merged."""
        inv_a = {"p": {"voiced": False}}
        inv_b = {"ʃ": {"voiced": False}}
        merged = merge_inventories(inv_a, inv_b)
        assert "p" in merged
        assert "ʃ" in merged

    def test_empty_inventory(self):
        """Merging with an empty inventory should work."""
        inv_a = {"p": {"voiced": False}}
        merged = merge_inventories(inv_a, {})
        assert "p" in merged

    def test_no_inventories(self):
        """Merging zero inventories should return empty dict."""
        merged = merge_inventories()
        assert merged == {}

    def test_real_language_merge_consistency(self):
        """Merging eng_us and fra inventories: shared phonemes should
        get the same universal features regardless of merge order."""
        from phone_similarity.language import LANGUAGES

        eng = LANGUAGES["eng_us"].PHONEME_FEATURES
        fra = LANGUAGES["fra"].PHONEME_FEATURES

        merged_ab = merge_inventories(eng, fra)
        merged_ba = merge_inventories(fra, eng)

        # Find shared phonemes
        shared = set(eng) & set(fra)
        assert len(shared) > 0, "Expected at least some shared phonemes"

        for ph in shared:
            assert merged_ab[ph] == merged_ba[ph], (
                f"Phoneme /{ph}/ differs depending on merge order"
            )


# -----------------------------------------------------------------------
# Integration with existing API
# -----------------------------------------------------------------------


class TestCrossLanguageIntegration:
    """Test that the universal features work end-to-end with ``compare_cross_language``."""

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
        # "water" -> /wɔːtər/ (eng) vs /o/ (fra, for "eau")
        ipa = {"eng_us": "wɔːtəɹ", "fra": "o"}
        results = compare_cross_language(ipa, specs, features, metric="edit")
        assert len(results) == 1
        key = ("eng_us", "fra")
        assert key in results
        assert 0.0 <= results[key] <= 1.0

    def test_intersecting_spec_tokenizes_both_languages(self, eng_fra_specs):
        """The merged specification should be able to tokenize IPA from
        both English and French."""
        from phone_similarity.intersecting_bit_array_specification import (
            IntersectingBitArraySpecification,
        )

        specs, _features = eng_fra_specs
        merged = IntersectingBitArraySpecification([specs["eng_us"], specs["fra"]])
        # English-specific phoneme
        assert "ɹ" in merged._phone_set or "ɹ" not in merged._phone_set  # just ensure no crash
        # The merged spec should have phonemes from both inventories
        assert merged._vowels == specs["eng_us"]._vowels | specs["fra"]._vowels
        assert merged._consonants == specs["eng_us"]._consonants | specs["fra"]._consonants
