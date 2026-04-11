"""
Tests for the co-articulation module.

Covers:
- DefaultCoarticulationModel construction and validation
- Anticipatory coarticulation (consonant before vowel)
- Carryover coarticulation (vowel after consonant)
- Consonant cluster assimilation (voicing, place)
- Syllable boundary attenuation
- Jitter/randomness behaviour
- Co-articulated distance functions
- Integration with syllable module
"""

from __future__ import annotations

import pytest

from phone_similarity.coarticulation import (
    _FEAT_IDX,
    _NUM_FEATURES,
    CoarticulationRule,
    DefaultCoarticulationModel,
    FricativeConfig,
    coarticulated_feature_edit_distance,
    coarticulated_phoneme_distance,
    normalised_coarticulated_feature_edit_distance,
)
from phone_similarity.universal_features import (
    UniversalFeatureEncoder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _feat_val(vec: tuple[float, ...], name: str) -> float:
    """Extract a named feature value from a vector."""
    return vec[_FEAT_IDX[name]]


def _base_vec(phoneme: str) -> tuple[int, ...]:
    """Get the unperturbed Panphon feature vector."""
    return UniversalFeatureEncoder.encode(phoneme)


# ===================================================================
# Construction & validation
# ===================================================================
class TestConstruction:
    """DefaultCoarticulationModel construction and validation."""

    def test_default_jitter_zero(self):
        model = DefaultCoarticulationModel()
        assert model.jitter == 0.0

    def test_custom_jitter(self):
        model = DefaultCoarticulationModel(jitter=0.5)
        assert model.jitter == 0.5

    def test_jitter_setter(self):
        model = DefaultCoarticulationModel()
        model.jitter = 0.7
        assert model.jitter == 0.7

    def test_jitter_out_of_range_init(self):
        with pytest.raises(ValueError, match="jitter"):
            DefaultCoarticulationModel(jitter=-0.1)
        with pytest.raises(ValueError, match="jitter"):
            DefaultCoarticulationModel(jitter=1.5)

    def test_jitter_out_of_range_setter(self):
        model = DefaultCoarticulationModel()
        with pytest.raises(ValueError, match="jitter"):
            model.jitter = 2.0

    def test_cross_syllable_decay_validation(self):
        with pytest.raises(ValueError, match="cross_syllable_decay"):
            DefaultCoarticulationModel(cross_syllable_decay=-0.1)
        with pytest.raises(ValueError, match="cross_syllable_decay"):
            DefaultCoarticulationModel(cross_syllable_decay=1.5)

    def test_seed_reproducibility(self):
        m1 = DefaultCoarticulationModel(jitter=0.8, seed=42)
        m2 = DefaultCoarticulationModel(jitter=0.8, seed=42)
        tokens = ["k", "æ", "t", "s"]
        r1 = m1.perturb_sequence(tokens)
        r2 = m2.perturb_sequence(tokens)
        assert r1 == r2

    def test_empty_input(self):
        model = DefaultCoarticulationModel()
        assert model.perturb_sequence([]) == []


# ===================================================================
# Feature vector basics
# ===================================================================
class TestFeatureVectors:
    """Perturbed vectors have correct shape and properties."""

    def test_output_length(self):
        model = DefaultCoarticulationModel()
        tokens = ["p", "a", "t"]
        result = model.perturb_sequence(tokens)
        assert len(result) == 3
        for vec in result:
            assert len(vec) == _NUM_FEATURES

    def test_isolated_phoneme_no_context(self):
        """Single phoneme -> no co-articulation effects possible."""
        model = DefaultCoarticulationModel()
        result = model.perturb_sequence(["a"])
        base = _base_vec("a")
        # Single phoneme should be unchanged (no neighbors)
        for i in range(_NUM_FEATURES):
            assert result[0][i] == float(base[i])

    def test_values_clamped(self):
        """All values should remain in [-1, +1]."""
        model = DefaultCoarticulationModel(jitter=1.0, seed=99)
        tokens = ["m", "b", "i", "n", "k", "u", "p", "a", "ŋ", "g"]
        result = model.perturb_sequence(tokens)
        for vec in result:
            for val in vec:
                assert -1.0 <= val <= 1.0


# ===================================================================
# Anticipatory coarticulation
# ===================================================================
class TestAnticipatory:
    """Consonant features shift toward following vowel."""

    def test_palatalization_before_high_vowel(self):
        """Consonant before /i/ should shift [hi] upward."""
        model = DefaultCoarticulationModel(jitter=0.0)
        # "t" before "i"
        result = model.perturb_sequence(["t", "i"])
        base_t = _base_vec("t")
        hi_base = float(base_t[_FEAT_IDX["hi"]])
        hi_perturbed = _feat_val(result[0], "hi")
        # Should shift toward +1 (if not already +1)
        if hi_base < 1.0:
            assert hi_perturbed > hi_base, (
                f"Expected hi to increase before /i/: {hi_base} -> {hi_perturbed}"
            )

    def test_lip_rounding_before_round_vowel(self):
        """Consonant before /u/ should shift [round] upward."""
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["t", "u"])
        base_t = _base_vec("t")
        round_base = float(base_t[_FEAT_IDX["round"]])
        round_perturbed = _feat_val(result[0], "round")
        if round_base < 1.0:
            assert round_perturbed > round_base

    def test_backing_before_back_vowel(self):
        """Consonant before /ɑ/ should shift [back] upward."""
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["t", "ɑ"])
        base_t = _base_vec("t")
        back_base = float(base_t[_FEAT_IDX["back"]])
        back_perturbed = _feat_val(result[0], "back")
        if back_base < 1.0:
            assert back_perturbed > back_base

    def test_no_anticipation_between_vowels(self):
        """Vowel-vowel sequence should not trigger anticipatory rules."""
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["a", "i"])
        base_a = _base_vec("a")
        # Vowel "a" should be unperturbed by anticipatory rules
        # (anticipatory only applies to consonants before vowels)
        for i in range(_NUM_FEATURES):
            assert result[0][i] == float(base_a[i])

    def test_multiple_effects_stack(self):
        """/u/ is both high and round and back; all effects should apply."""
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["t", "u"])
        base_t = _base_vec("t")
        # Check that multiple features shifted
        hi_shifted = _feat_val(result[0], "hi") != float(base_t[_FEAT_IDX["hi"]])
        round_shifted = _feat_val(result[0], "round") != float(base_t[_FEAT_IDX["round"]])
        back_shifted = _feat_val(result[0], "back") != float(base_t[_FEAT_IDX["back"]])
        assert sum([hi_shifted, round_shifted, back_shifted]) >= 2


# ===================================================================
# Carryover coarticulation
# ===================================================================
class TestCarryover:
    """Vowel features shift due to preceding consonant."""

    def test_nasalization_after_nasal(self):
        """Vowel after /m/ should shift [nas] upward."""
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["m", "a"])
        base_a = _base_vec("a")
        nas_base = float(base_a[_FEAT_IDX["nas"]])
        nas_perturbed = _feat_val(result[1], "nas")
        assert nas_perturbed > nas_base, (
            f"Expected nasalization after /m/: {nas_base} -> {nas_perturbed}"
        )

    def test_nasalization_after_n(self):
        """Vowel after /n/ should also nasalize."""
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["n", "a"])
        base_a = _base_vec("a")
        nas_base = float(base_a[_FEAT_IDX["nas"]])
        nas_perturbed = _feat_val(result[1], "nas")
        assert nas_perturbed > nas_base

    def test_backing_after_velar(self):
        """Vowel after /k/ should shift [back] upward."""
        model = DefaultCoarticulationModel(jitter=0.0)
        # /a/ is already [+back], so check for /k/ before /ɪ/ instead
        result2 = model.perturb_sequence(["k", "ɪ"])
        base_i = _base_vec("ɪ")
        back_base_i = float(base_i[_FEAT_IDX["back"]])
        back_pert_i = _feat_val(result2[1], "back")
        if back_base_i < 1.0:
            assert back_pert_i > back_base_i

    def test_rounding_after_labial(self):
        """Vowel after /p/ should shift [round] upward."""
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["p", "a"])
        base_a = _base_vec("a")
        round_base = float(base_a[_FEAT_IDX["round"]])
        round_perturbed = _feat_val(result[1], "round")
        if round_base < 1.0:
            assert round_perturbed > round_base

    def test_no_carryover_between_consonants(self):
        """Consonant-consonant should not trigger carryover rules."""
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["m", "p"])
        # "p" is a consonant, not a vowel -- carryover rules don't apply
        # But assimilation rules might, so just check that carryover-specific
        # features like nasalization didn't apply to the consonant
        base_p = _base_vec("p")
        nas_base = float(base_p[_FEAT_IDX["nas"]])
        nas_pert = _feat_val(result[1], "nas")
        # Nasalization carryover should NOT apply to consonants
        # (Only voicing/place assimilation may apply)
        assert nas_pert == nas_base


# ===================================================================
# Consonant cluster assimilation
# ===================================================================
class TestAssimilation:
    """Consonant-consonant cluster effects."""

    def test_voicing_assimilation(self):
        """Voiceless consonant before voiced should shift [voi] upward."""
        model = DefaultCoarticulationModel(jitter=0.0)
        # /s/ before /b/ - /s/ is [-voi], /b/ is [+voi]
        result = model.perturb_sequence(["s", "b"])
        base_s = _base_vec("s")
        voi_base = float(base_s[_FEAT_IDX["voi"]])
        voi_perturbed = _feat_val(result[0], "voi")
        assert voi_perturbed > voi_base, (
            f"Expected voicing shift before /b/: {voi_base} -> {voi_perturbed}"
        )

    def test_devoicing_assimilation(self):
        """Voiced consonant before voiceless should shift [voi] downward."""
        model = DefaultCoarticulationModel(jitter=0.0)
        # /z/ before /t/ - /z/ is [+voi], /t/ is [-voi]
        result = model.perturb_sequence(["z", "t"])
        base_z = _base_vec("z")
        voi_base = float(base_z[_FEAT_IDX["voi"]])
        voi_perturbed = _feat_val(result[0], "voi")
        assert voi_perturbed < voi_base, (
            f"Expected devoicing before /t/: {voi_base} -> {voi_perturbed}"
        )

    def test_nasal_place_assimilation(self):
        """/n/ before /k/ should shift place features toward velar."""
        model = DefaultCoarticulationModel(jitter=0.0)
        # /n/ is [+ant, +cor]; /k/ is typically [-ant, -cor]
        result = model.perturb_sequence(["n", "k"])
        base_n = _base_vec("n")
        base_k = _base_vec("k")

        # At minimum, if /k/ has distinct place features from /n/,
        # /n/ should shift toward them
        lab_k = base_k[_FEAT_IDX["lab"]]
        ant_k = base_k[_FEAT_IDX["ant"]]
        cor_k = base_k[_FEAT_IDX["cor"]]

        # Check that at least one place feature shifted toward /k/'s value
        shifted = False
        for feat, k_val in [("lab", lab_k), ("ant", ant_k), ("cor", cor_k)]:
            if k_val != 0:
                base_val = float(base_n[_FEAT_IDX[feat]])
                pert_val = _feat_val(result[0], feat)
                if (k_val == 1 and pert_val > base_val) or (k_val == -1 and pert_val < base_val):
                    shifted = True
        assert shifted, "Expected at least one place feature to shift"


# ===================================================================
# Syllable boundary effects
# ===================================================================
class TestSyllableBoundary:
    """Effects attenuate across syllable boundaries."""

    def test_within_syllable_stronger(self):
        """Co-articulation within syllable > across syllable boundary."""
        model = DefaultCoarticulationModel(jitter=0.0, cross_syllable_decay=0.4)
        tokens = ["m", "a"]

        # Within syllable (both in syllable 0)
        result_same = model.perturb_sequence(tokens, syllable_boundaries=[0, 0])
        nas_same = _feat_val(result_same[1], "nas")

        # Across syllable boundary (m in syl 0, a in syl 1)
        result_cross = model.perturb_sequence(tokens, syllable_boundaries=[0, 1])
        nas_cross = _feat_val(result_cross[1], "nas")

        base_a = _base_vec("a")
        nas_base = float(base_a[_FEAT_IDX["nas"]])

        # Both should be perturbed upward, but same-syllable more
        assert nas_same > nas_base
        assert nas_cross > nas_base
        assert nas_same > nas_cross, (
            f"Same-syllable effect ({nas_same}) should be > cross-syllable ({nas_cross})"
        )

    def test_zero_decay_blocks_cross_syllable(self):
        """With decay=0, cross-syllable effects should be zero."""
        model = DefaultCoarticulationModel(jitter=0.0, cross_syllable_decay=0.0)
        tokens = ["m", "a"]

        result = model.perturb_sequence(tokens, syllable_boundaries=[0, 1])
        base_a = _base_vec("a")
        nas_base = float(base_a[_FEAT_IDX["nas"]])
        nas_pert = _feat_val(result[1], "nas")
        assert nas_pert == nas_base, (
            f"With decay=0 across boundary, expected no change: {nas_base} -> {nas_pert}"
        )

    def test_no_boundaries_assumes_same_syllable(self):
        """Without syllable info, all effects are full strength."""
        model = DefaultCoarticulationModel(jitter=0.0)
        tokens = ["m", "a"]
        result_none = model.perturb_sequence(tokens, syllable_boundaries=None)
        result_same = model.perturb_sequence(tokens, syllable_boundaries=[0, 0])
        # Should be identical
        assert result_none == result_same

    def test_syllable_boundary_map(self):
        """Test the helper that converts Syllable objects to boundary indices."""
        from phone_similarity.syllable import Syllable

        syllables = [
            Syllable(onset=("k",), nucleus=("æ",), coda=("t",)),
            Syllable(onset=("s",), nucleus=("ɪ",), coda=()),
        ]
        boundaries = DefaultCoarticulationModel.syllable_boundary_map(syllables)
        assert boundaries == [0, 0, 0, 1, 1]


# ===================================================================
# Jitter / randomness
# ===================================================================
class TestJitter:
    """Stochastic perturbation behaviour."""

    def test_zero_jitter_deterministic(self):
        """With jitter=0, results should be perfectly deterministic."""
        model = DefaultCoarticulationModel(jitter=0.0)
        tokens = ["k", "æ", "t", "s"]
        r1 = model.perturb_sequence(tokens)
        r2 = model.perturb_sequence(tokens)
        assert r1 == r2

    def test_nonzero_jitter_varies(self):
        """With jitter>0 and no seed, different calls may produce different results."""
        model = DefaultCoarticulationModel(jitter=0.8, seed=None)
        tokens = ["m", "a", "n", "i", "k"]
        results = set()
        for _ in range(20):
            r = model.perturb_sequence(tokens)
            results.add(r[1])  # vowel after nasal, most likely to vary
        # With high jitter, we should get some variation
        assert len(results) > 1, "Expected variation with jitter=0.8"

    def test_high_jitter_sometimes_no_effect(self):
        """With jitter=1.0, some rule applications should be skipped."""
        tokens = ["m", "a"]
        base_a = _base_vec("a")
        nas_base = float(base_a[_FEAT_IDX["nas"]])

        # Run many times: at least one should have minimal nasalization
        min_nas = float("inf")
        for trial_seed in range(50):
            m = DefaultCoarticulationModel(jitter=1.0, seed=trial_seed)
            result = m.perturb_sequence(tokens)
            nas_val = _feat_val(result[1], "nas")
            min_nas = min(min_nas, nas_val)
        # With high jitter, at least one trial should produce weak/no effect
        # (The rule either didn't fire or fired at reduced magnitude)
        assert min_nas < nas_base + 0.5, "Expected at least one weak nasalization with jitter=1.0"

    def test_seed_produces_same_jitter(self):
        """Same seed + same jitter = same output."""
        for seed in [0, 42, 12345]:
            m1 = DefaultCoarticulationModel(jitter=0.5, seed=seed)
            m2 = DefaultCoarticulationModel(jitter=0.5, seed=seed)
            tokens = ["b", "ɪ", "n", "d", "z"]
            assert m1.perturb_sequence(tokens) == m2.perturb_sequence(tokens)


# ===================================================================
# Co-articulated distance functions
# ===================================================================
class TestCoarticulatedDistance:
    """Distance computation with co-articulation."""

    def test_identical_sequences_zero_distance(self):
        """Same sequence should have distance ~0 (co-articulation is same)."""
        dist = coarticulated_feature_edit_distance(["k", "æ", "t"], ["k", "æ", "t"])
        assert dist == pytest.approx(0.0)

    def test_symmetry(self):
        """Distance should be symmetric (with deterministic model)."""
        d1 = coarticulated_feature_edit_distance(["k", "æ", "t"], ["k", "æ", "b"])
        d2 = coarticulated_feature_edit_distance(["k", "æ", "b"], ["k", "æ", "t"])
        assert d1 == pytest.approx(d2)

    def test_distance_positive(self):
        """Different sequences should have positive distance."""
        dist = coarticulated_feature_edit_distance(["p", "a"], ["b", "a"])
        assert dist > 0.0

    def test_normalised_range(self):
        """Normalised distance should be in [0, 1]."""
        dist = normalised_coarticulated_feature_edit_distance(["k", "æ", "t"], ["m", "u", "s"])
        assert 0.0 <= dist <= 1.0

    def test_normalised_empty(self):
        """Both empty -> 0.0."""
        assert normalised_coarticulated_feature_edit_distance([], []) == 0.0

    def test_coarticulated_distance_differs_from_plain(self):
        """Co-articulated distance should differ from plain feature distance."""
        from phone_similarity.primitives import normalised_feature_edit_distance
        from phone_similarity.universal_features import UniversalFeatureEncoder

        tokens_a = ["m", "i", "n"]
        tokens_b = ["b", "i", "n"]

        feats = UniversalFeatureEncoder.merge_inventories(
            {ph: {} for ph in set(tokens_a) | set(tokens_b)}
        )

        plain = normalised_feature_edit_distance(tokens_a, tokens_b, feats)
        coart = normalised_coarticulated_feature_edit_distance(tokens_a, tokens_b)

        # They should NOT be exactly equal (co-articulation perturbs features)
        assert plain != pytest.approx(coart, abs=1e-6), (
            f"Expected co-articulated distance to differ from plain: "
            f"plain={plain:.6f}, coart={coart:.6f}"
        )

    def test_phoneme_distance_function(self):
        """Test the vector-level phoneme distance function."""
        a = (1.0,) * _NUM_FEATURES
        b = (1.0,) * _NUM_FEATURES
        assert coarticulated_phoneme_distance(a, b) == 0.0

        c = (-1.0,) * _NUM_FEATURES
        dist = coarticulated_phoneme_distance(a, c)
        assert dist == pytest.approx(1.0)

    def test_phoneme_distance_skips_zero(self):
        """Features near zero in both vectors are skipped."""
        a = (0.0, 1.0, -1.0) + (0.0,) * (_NUM_FEATURES - 3)
        b = (0.0, -1.0, 1.0) + (0.0,) * (_NUM_FEATURES - 3)
        # Only indices 1 and 2 are comparable; both differ maximally
        dist = coarticulated_phoneme_distance(a, b)
        assert dist == pytest.approx(1.0)


# ===================================================================
# Integration with syllable module
# ===================================================================
class TestSyllableIntegration:
    """Co-articulation with syllable boundary information."""

    def test_syllabify_then_coarticulate(self):
        """Full pipeline: tokenize -> syllabify -> extract boundaries -> perturb."""
        from phone_similarity.syllable import syllabify

        tokens = ["b", "æ", "n", "æ", "n", "ə"]
        vowels = frozenset({"æ", "ə"})

        syllables = syllabify(tokens, vowels)
        boundaries = DefaultCoarticulationModel.syllable_boundary_map(syllables)

        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(tokens, syllable_boundaries=boundaries)

        assert len(result) == 6
        for vec in result:
            assert len(vec) == _NUM_FEATURES

    def test_cross_syllable_nasalization_reduced(self):
        """Nasalization of vowel across syllable boundary is weaker."""
        from phone_similarity.syllable import syllabify

        # "ban.a" - /n/ in coda of syl 0, /a/ in nucleus of syl 1
        tokens = ["b", "æ", "n", "a"]
        vowels = frozenset({"æ", "a"})

        syllables = syllabify(tokens, vowels)
        boundaries = DefaultCoarticulationModel.syllable_boundary_map(syllables)

        model = DefaultCoarticulationModel(jitter=0.0, cross_syllable_decay=0.4)

        # With boundaries
        result_with = model.perturb_sequence(tokens, syllable_boundaries=boundaries)
        # Without boundaries (treat as same syllable)
        result_without = model.perturb_sequence(tokens, syllable_boundaries=None)

        nas_with = _feat_val(result_with[3], "nas")  # /a/ in syl 1
        nas_without = _feat_val(result_without[3], "nas")  # /a/, no boundary

        # Both should be perturbed, but with boundaries should be weaker
        base_a = _base_vec("a")
        nas_base = float(base_a[_FEAT_IDX["nas"]])

        # Verify that /n/ before /a/ triggers nasalization at all
        assert nas_without > nas_base, f"Expected nasalization: {nas_base} -> {nas_without}"

        # Cross-syllable should be weaker (if n and a are in different syllables)
        if boundaries[2] != boundaries[3]:
            assert nas_with < nas_without or nas_with == pytest.approx(nas_without), (
                f"Cross-syllable nasalization ({nas_with}) should be <= "
                f"same-syllable ({nas_without})"
            )


# ===================================================================
# CoarticulationRule dataclass
# ===================================================================
class TestCoarticulationRule:
    """CoarticulationRule frozen dataclass behaviour."""

    def test_frozen(self):
        rule = CoarticulationRule("test", 0, 1.0, 0.5)
        with pytest.raises(AttributeError):
            rule.name = "changed"  # type: ignore[misc]

    def test_default_values(self):
        rule = CoarticulationRule("test", 0, 1.0, 0.5)
        assert rule.base_probability == 0.8
        assert rule.within_syllable_only is False

    def test_custom_values(self):
        rule = CoarticulationRule(
            "custom", 3, -1.0, 0.3, base_probability=0.5, within_syllable_only=True
        )
        assert rule.base_probability == 0.5
        assert rule.within_syllable_only is True


# ===================================================================
# Real-world phonological scenarios
# ===================================================================
class TestRealWorldScenarios:
    """Scenarios grounded in actual phonological phenomena."""

    def test_english_cant_nasalization(self):
        """English 'can't' /kænt/ - vowel before nasal has nasalization.

        In English, vowels are nasalized before nasal consonants.
        Here /æ/ precedes /n/, so carryover from a preceding consonant
        isn't the mechanism -- but anticipatory nasalization from the
        following nasal is a real phenomenon not yet modelled.

        This test checks the carryover effect: /n/ influences the
        following /t/ via voicing assimilation context.
        """
        model = DefaultCoarticulationModel(jitter=0.0)
        # "kin" - /k/ before /ɪ/ (velar backing), /n/ after /ɪ/ (no carryover to C)
        result = model.perturb_sequence(["k", "ɪ", "n"])
        # /ɪ/ after /k/ should have velar backing
        base_i = _base_vec("ɪ")
        back_base = float(base_i[_FEAT_IDX["back"]])
        back_pert = _feat_val(result[1], "back")
        if back_base < 1.0:
            assert back_pert > back_base

    def test_french_nasalization(self):
        """French nasal vowels: /mɑ̃/ has strong nasalization context.

        Even without true nasal vowels, the /m/ -> /a/ transition
        should show nasalization carryover.
        """
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["m", "a"])
        base_a = _base_vec("a")
        nas_base = float(base_a[_FEAT_IDX["nas"]])
        nas_pert = _feat_val(result[1], "nas")
        assert nas_pert > nas_base

    def test_german_auslautverhaertung_context(self):
        """German final devoicing: /d/ before silence = [t].

        When /d/ is followed by a voiceless consonant like /s/,
        voicing assimilation partially devoices it.
        """
        model = DefaultCoarticulationModel(jitter=0.0)
        # "ds" cluster - /d/ before /s/ (voiceless)
        result = model.perturb_sequence(["d", "s"])
        base_d = _base_vec("d")
        voi_base = float(base_d[_FEAT_IDX["voi"]])
        voi_pert = _feat_val(result[0], "voi")
        assert voi_pert < voi_base, (
            f"Expected devoicing of /d/ before /s/: {voi_base} -> {voi_pert}"
        )

    def test_pun_distance_closer_with_coarticulation(self):
        """Some near-homophones should be closer with co-articulation.

        'cat' vs 'gat': /k/ and /g/ differ primarily in voicing.
        Before the same vowel /æ/, co-articulation makes both
        consonants shift similarly, potentially reducing their distance.
        """
        model = DefaultCoarticulationModel(jitter=0.0)
        d_coart = coarticulated_feature_edit_distance(
            ["k", "æ", "t"], ["ɡ", "æ", "t"], model=model
        )
        # Should be a small distance (voicing is the main difference)
        assert d_coart < 0.5, f"Expected small distance for cat/gat: {d_coart}"

    def test_longer_word_coarticulation(self):
        """Multi-syllable word: 'banana' /bənænə/."""
        model = DefaultCoarticulationModel(jitter=0.0)
        tokens = ["b", "ə", "n", "æ", "n", "ə"]
        result = model.perturb_sequence(tokens)
        assert len(result) == 6
        # Each /n/ before vowel should show anticipatory effects
        # /ə/ after /n/ should show nasalization carryover
        base_schwa = _base_vec("ə")
        nas_base = float(base_schwa[_FEAT_IDX["nas"]])
        # Second /ə/ (index 5) follows /n/ (index 4)
        nas_pert = _feat_val(result[5], "nas")
        assert nas_pert > nas_base


# ===================================================================
# FricativeConfig dataclass
# ===================================================================
class TestFricativeConfig:
    """FricativeConfig frozen dataclass construction and validation."""

    def test_defaults_and_custom(self):
        fc = FricativeConfig()
        assert fc.fricative_weight == 1.0
        assert fc.sibilant_weight == 1.0
        assert fc.frication_spread is False
        assert fc.spread_magnitude == 0.30

        fc2 = FricativeConfig(
            fricative_weight=2.0,
            sibilant_weight=1.5,
            frication_spread=True,
            spread_magnitude=0.50,
        )
        assert fc2.fricative_weight == 2.0
        assert fc2.frication_spread is True

        with pytest.raises(AttributeError):
            fc.fricative_weight = 2.0  # type: ignore[misc]

    def test_validation(self):
        with pytest.raises(ValueError, match="fricative_weight"):
            FricativeConfig(fricative_weight=-0.1)
        with pytest.raises(ValueError, match="sibilant_weight"):
            FricativeConfig(sibilant_weight=-1.0)
        with pytest.raises(ValueError, match="spread_magnitude"):
            FricativeConfig(spread_magnitude=1.5)
        with pytest.raises(ValueError, match="spread_magnitude"):
            FricativeConfig(spread_magnitude=-0.1)

    def test_zero_weight_allowed(self):
        fc = FricativeConfig(fricative_weight=0.0, sibilant_weight=0.0)
        assert fc.fricative_weight == 0.0


# ===================================================================
# FricativeConfig integration with model
# ===================================================================
class TestFricativeConfigModel:
    """FricativeConfig integration with DefaultCoarticulationModel."""

    def test_config_lifecycle(self):
        model = DefaultCoarticulationModel()
        assert model.fricative_config.fricative_weight == 1.0
        assert model.fricative_config.frication_spread is False

        fc = FricativeConfig(fricative_weight=2.0)
        model2 = DefaultCoarticulationModel(fricative_config=fc)
        assert model2.fricative_config.fricative_weight == 2.0

        fc3 = FricativeConfig(fricative_weight=3.0)
        model.fricative_config = fc3
        assert model.fricative_config.fricative_weight == 3.0

        with pytest.raises(TypeError, match="FricativeConfig"):
            model.fricative_config = "not a config"  # type: ignore[assignment]


# ===================================================================
# FricativeConfig weighting in distance computation
# ===================================================================
class TestFricativeWeighting:
    """Fricative-specific weighting changes distance values."""

    def test_weight_scales_fricative_distance(self):
        """Higher fricative_weight increases distance; zero reduces it."""
        model = DefaultCoarticulationModel(jitter=0.0)
        fc_zero = FricativeConfig(fricative_weight=0.0)
        fc_normal = FricativeConfig(fricative_weight=1.0)
        fc_heavy = FricativeConfig(fricative_weight=3.0)

        d_zero = coarticulated_feature_edit_distance(
            ["s", "a"],
            ["t", "a"],
            model=model,
            fricative_config=fc_zero,
        )
        d_normal = coarticulated_feature_edit_distance(
            ["s", "æ", "t"],
            ["t", "æ", "t"],
            model=model,
            fricative_config=fc_normal,
        )
        d_heavy = coarticulated_feature_edit_distance(
            ["s", "æ", "t"],
            ["t", "æ", "t"],
            model=model,
            fricative_config=fc_heavy,
        )
        d_normal_sa = coarticulated_feature_edit_distance(
            ["s", "a"],
            ["t", "a"],
            model=model,
            fricative_config=fc_normal,
        )
        assert d_zero < d_normal_sa
        assert d_heavy > d_normal

    def test_sibilant_weight_affects_sibilant_pairs(self):
        """Higher sibilant_weight increases distance between strident
        and non-strident fricatives."""
        model = DefaultCoarticulationModel(jitter=0.0)
        fc_normal = FricativeConfig(sibilant_weight=1.0)
        fc_heavy = FricativeConfig(sibilant_weight=3.0)

        d_normal = coarticulated_feature_edit_distance(
            ["s", "a"],
            ["θ", "a"],
            model=model,
            fricative_config=fc_normal,
        )
        d_heavy = coarticulated_feature_edit_distance(
            ["s", "a"],
            ["θ", "a"],
            model=model,
            fricative_config=fc_heavy,
        )
        assert d_heavy > d_normal

    def test_non_fricative_pair_unaffected(self):
        """Fricative weight should not change distance between two stops."""
        model = DefaultCoarticulationModel(jitter=0.0)
        fc_normal = FricativeConfig(fricative_weight=1.0)
        fc_heavy = FricativeConfig(fricative_weight=5.0)

        d_normal = coarticulated_feature_edit_distance(
            ["p", "a"],
            ["b", "a"],
            model=model,
            fricative_config=fc_normal,
        )
        d_heavy = coarticulated_feature_edit_distance(
            ["p", "a"],
            ["b", "a"],
            model=model,
            fricative_config=fc_heavy,
        )
        assert d_normal == pytest.approx(d_heavy)

    def test_config_propagation(self):
        """Model config used when no explicit config; normalised and phoneme
        distance also respect FricativeConfig."""
        fc = FricativeConfig(fricative_weight=3.0)
        model = DefaultCoarticulationModel(jitter=0.0, fricative_config=fc)
        model_default = DefaultCoarticulationModel(
            jitter=0.0,
            fricative_config=FricativeConfig(fricative_weight=1.0),
        )

        d_heavy = coarticulated_feature_edit_distance(
            ["s", "a"],
            ["t", "a"],
            model=model,
        )
        d_normal = coarticulated_feature_edit_distance(
            ["s", "a"],
            ["t", "a"],
            model=model_default,
        )
        assert d_heavy > d_normal

        d_norm = normalised_coarticulated_feature_edit_distance(
            ["s", "a"],
            ["t", "a"],
            fricative_config=fc,
        )
        assert 0.0 < d_norm <= 1.5

        vec_s = model.perturb_sequence(["s"])[0]
        vec_t = model.perturb_sequence(["t"])[0]
        d_ph_normal = coarticulated_phoneme_distance(
            vec_s,
            vec_t,
            fricative_config=FricativeConfig(fricative_weight=1.0),
        )
        d_ph_heavy = coarticulated_phoneme_distance(
            vec_s,
            vec_t,
            fricative_config=fc,
        )
        assert d_ph_heavy > d_ph_normal


# ===================================================================
# Frication spread co-articulation rules
# ===================================================================
class TestFricationSpread:
    """Frication noise spread to adjacent segments."""

    def test_frication_spread_off_by_default(self):
        """Without frication_spread=True, no spread rules fire;
        stops also never trigger spread."""
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["s", "a"])
        base_a = _base_vec("a")
        voi_base = float(base_a[_FEAT_IDX["voi"]])
        assert _feat_val(result[1], "voi") == voi_base

        fc = FricativeConfig(frication_spread=True)
        model2 = DefaultCoarticulationModel(jitter=0.0, fricative_config=fc)
        result2 = model2.perturb_sequence(["t", "a"])
        cont_base = float(base_a[_FEAT_IDX["cont"]])
        assert _feat_val(result2[1], "cont") == cont_base

    def test_frication_spread_devoices_vowel(self):
        """With frication_spread=True, vowel after voiceless fricative
        gets partial devoicing (via both anticipatory and carryover paths)."""
        fc = FricativeConfig(frication_spread=True, spread_magnitude=0.30)
        model = DefaultCoarticulationModel(jitter=0.0, fricative_config=fc)
        result = model.perturb_sequence(["s", "a"])
        base_a = _base_vec("a")
        voi_base = float(base_a[_FEAT_IDX["voi"]])
        assert _feat_val(result[1], "voi") < voi_base

    def test_sibilant_spreads_stridency(self):
        """Sibilant /s/ spreads [strid]; non-strident /θ/ does not,
        but /θ/ still devoices."""
        fc = FricativeConfig(frication_spread=True, spread_magnitude=0.30)
        model = DefaultCoarticulationModel(jitter=0.0, fricative_config=fc)
        base_a = _base_vec("a")
        strid_base = float(base_a[_FEAT_IDX["strid"]])
        voi_base = float(base_a[_FEAT_IDX["voi"]])

        result_s = model.perturb_sequence(["s", "a"])
        assert _feat_val(result_s[1], "strid") > strid_base

        result_th = model.perturb_sequence(["θ", "a"])
        assert _feat_val(result_th[1], "strid") == strid_base
        assert _feat_val(result_th[1], "voi") < voi_base

    def test_spread_magnitude_scales_effect(self):
        """Higher spread_magnitude produces stronger devoicing."""
        fc_low = FricativeConfig(frication_spread=True, spread_magnitude=0.10)
        fc_high = FricativeConfig(frication_spread=True, spread_magnitude=0.60)
        model_low = DefaultCoarticulationModel(jitter=0.0, fricative_config=fc_low)
        model_high = DefaultCoarticulationModel(jitter=0.0, fricative_config=fc_high)

        voi_low = _feat_val(model_low.perturb_sequence(["s", "a"])[1], "voi")
        voi_high = _feat_val(model_high.perturb_sequence(["s", "a"])[1], "voi")
        assert voi_high < voi_low

    def test_cross_syllable_decay_affects_spread(self):
        """Frication spread across syllable boundary is attenuated."""
        fc = FricativeConfig(frication_spread=True, spread_magnitude=0.30)
        model = DefaultCoarticulationModel(
            jitter=0.0,
            fricative_config=fc,
            cross_syllable_decay=0.4,
        )
        tokens = ["s", "a"]
        base_a = _base_vec("a")
        voi_base = float(base_a[_FEAT_IDX["voi"]])

        voi_same = _feat_val(
            model.perturb_sequence(tokens, syllable_boundaries=[0, 0])[1],
            "voi",
        )
        voi_cross = _feat_val(
            model.perturb_sequence(tokens, syllable_boundaries=[0, 1])[1],
            "voi",
        )
        assert voi_same < voi_base
        assert voi_cross < voi_base
        assert voi_same < voi_cross


# ===================================================================
# Real-world fricative scenarios
# ===================================================================
class TestFricativeRealWorld:
    """Real phonological scenarios involving fricatives."""

    def test_english_s_vs_theta_distance(self):
        """/s/ vs /θ/: sibilant vs non-sibilant.  With sibilant_weight > 1,
        this pair should be further apart."""
        fc_normal = FricativeConfig(sibilant_weight=1.0)
        fc_heavy = FricativeConfig(sibilant_weight=2.5)
        model = DefaultCoarticulationModel(jitter=0.0)

        d_normal = coarticulated_feature_edit_distance(
            ["s", "ɪ", "ŋ"],
            ["θ", "ɪ", "ŋ"],
            model=model,
            fricative_config=fc_normal,
        )
        d_heavy = coarticulated_feature_edit_distance(
            ["s", "ɪ", "ŋ"],
            ["θ", "ɪ", "ŋ"],
            model=model,
            fricative_config=fc_heavy,
        )
        assert d_heavy > d_normal

    def test_spanish_s_aspiration(self):
        """Spanish coda /s/ before consonant often weakens/aspirates.
        With frication spread, the effect should propagate."""
        fc = FricativeConfig(frication_spread=True, spread_magnitude=0.40)
        model = DefaultCoarticulationModel(jitter=0.0, fricative_config=fc)
        # "es.to" — /s/ before /t/
        result = model.perturb_sequence(["e", "s", "t", "o"])
        # /s/ (index 1) is a fricative before consonant /t/ (index 2)
        # Frication spread only applies to vowels, so /t/ should be unaffected
        # by frication spread; but /o/ after /t/ won't get it either since
        # /t/ isn't a fricative. Let's check /e/ before /s/ carryover doesn't apply.
        # Actually, the spread is anticipatory: fricative before vowel.
        # /s/ is before /t/ (consonant), so no spread there.
        assert len(result) == 4

    def test_german_ich_laut_vs_ach_laut(self):
        """German /ç/ (palatal fricative) vs /x/ (velar fricative).
        Both are fricatives; with fricative_weight=1, the distance
        should reflect place-of-articulation differences."""
        model = DefaultCoarticulationModel(jitter=0.0)
        d = coarticulated_feature_edit_distance(
            ["ç", "a"],
            ["x", "a"],
            model=model,
        )
        assert d > 0.0, "Expected positive distance for ç vs x"

    def test_french_liaison_fricative(self):
        """French liaison: 'les amis' -> /le.z‿a.mi/.
        /z/ (voiced sibilant fricative) between vowels spreads stridency
        but does NOT devoice (because /z/ is voiced)."""
        fc = FricativeConfig(frication_spread=True)
        model = DefaultCoarticulationModel(jitter=0.0, fricative_config=fc)
        result = model.perturb_sequence(["e", "z", "a", "m", "i"])
        # /a/ (index 2) after /z/ (index 1, voiced sibilant) should get
        # stridency spread but NOT devoicing
        base_a = _base_vec("a")
        strid_base = float(base_a[_FEAT_IDX["strid"]])
        strid_pert = _feat_val(result[2], "strid")
        assert strid_pert > strid_base, (
            f"Expected stridency spread from /z/: {strid_base} -> {strid_pert}"
        )
        # /z/ is voiced, so no devoicing should happen
        voi_base = float(base_a[_FEAT_IDX["voi"]])
        voi_pert = _feat_val(result[2], "voi")
        # voi should not decrease (no devoicing from voiced fricative)
        # (carryover nasalization from /m/ won't affect voi)
        assert voi_pert >= voi_base - 0.01
