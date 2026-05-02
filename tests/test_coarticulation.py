"""Tests for the co-articulation module."""

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
from phone_similarity.universal_features import UniversalFeatureEncoder


def _feat_val(vec: tuple[float, ...], name: str) -> float:
    return vec[_FEAT_IDX[name]]


def _base_vec(phoneme: str) -> tuple[int, ...]:
    return UniversalFeatureEncoder.encode(phoneme)


class TestConstruction:
    def test_default_jitter_zero(self):
        assert DefaultCoarticulationModel().jitter == 0.0

    def test_jitter_out_of_range_init(self):
        with pytest.raises(ValueError, match="jitter"):
            DefaultCoarticulationModel(jitter=-0.1)
        with pytest.raises(ValueError, match="jitter"):
            DefaultCoarticulationModel(jitter=1.5)

    def test_cross_syllable_decay_validation(self):
        with pytest.raises(ValueError, match="cross_syllable_decay"):
            DefaultCoarticulationModel(cross_syllable_decay=-0.1)
        with pytest.raises(ValueError, match="cross_syllable_decay"):
            DefaultCoarticulationModel(cross_syllable_decay=1.5)

    def test_seed_reproducibility(self):
        tokens = ["k", "æ", "t", "s"]
        m1 = DefaultCoarticulationModel(jitter=0.8, seed=42)
        m2 = DefaultCoarticulationModel(jitter=0.8, seed=42)
        assert m1.perturb_sequence(tokens) == m2.perturb_sequence(tokens)

    def test_empty_input(self):
        assert DefaultCoarticulationModel().perturb_sequence([]) == []


class TestFeatureVectors:
    def test_output_length(self):
        result = DefaultCoarticulationModel().perturb_sequence(["p", "a", "t"])
        assert len(result) == 3
        for vec in result:
            assert len(vec) == _NUM_FEATURES

    def test_isolated_phoneme_no_context(self):
        result = DefaultCoarticulationModel().perturb_sequence(["a"])
        base = _base_vec("a")
        for i in range(_NUM_FEATURES):
            assert result[0][i] == float(base[i])

    def test_values_clamped(self):
        model = DefaultCoarticulationModel(jitter=1.0, seed=99)
        for vec in model.perturb_sequence(["m", "b", "i", "n", "k", "u", "p", "a", "ŋ", "g"]):
            for val in vec:
                assert -1.0 <= val <= 1.0


class TestAnticipatory:
    def test_palatalization_before_high_vowel(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["t", "i"])
        base_t = _base_vec("t")
        hi_base = float(base_t[_FEAT_IDX["hi"]])
        hi_perturbed = _feat_val(result[0], "hi")
        if hi_base < 1.0:
            assert hi_perturbed > hi_base

    def test_lip_rounding_before_round_vowel(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["t", "u"])
        base_t = _base_vec("t")
        round_base = float(base_t[_FEAT_IDX["round"]])
        round_perturbed = _feat_val(result[0], "round")
        if round_base < 1.0:
            assert round_perturbed > round_base

    def test_no_anticipation_between_vowels(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["a", "i"])
        base_a = _base_vec("a")
        for i in range(_NUM_FEATURES):
            assert result[0][i] == float(base_a[i])

    def test_multiple_effects_stack(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["t", "u"])
        base_t = _base_vec("t")
        hi_shifted = _feat_val(result[0], "hi") != float(base_t[_FEAT_IDX["hi"]])
        round_shifted = _feat_val(result[0], "round") != float(base_t[_FEAT_IDX["round"]])
        back_shifted = _feat_val(result[0], "back") != float(base_t[_FEAT_IDX["back"]])
        assert sum([hi_shifted, round_shifted, back_shifted]) >= 2


class TestCarryover:
    def test_nasalization_after_nasal(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["m", "a"])
        base_a = _base_vec("a")
        assert _feat_val(result[1], "nas") > float(base_a[_FEAT_IDX["nas"]])

    def test_backing_after_velar(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["k", "ɪ"])
        base_i = _base_vec("ɪ")
        back_base = float(base_i[_FEAT_IDX["back"]])
        if back_base < 1.0:
            assert _feat_val(result[1], "back") > back_base

    def test_no_carryover_between_consonants(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["m", "p"])
        base_p = _base_vec("p")
        assert _feat_val(result[1], "nas") == float(base_p[_FEAT_IDX["nas"]])


class TestAssimilation:
    def test_voicing_assimilation(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["s", "b"])
        base_s = _base_vec("s")
        assert _feat_val(result[0], "voi") > float(base_s[_FEAT_IDX["voi"]])

    def test_devoicing_assimilation(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["z", "t"])
        base_z = _base_vec("z")
        assert _feat_val(result[0], "voi") < float(base_z[_FEAT_IDX["voi"]])

    def test_nasal_place_assimilation(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["n", "k"])
        base_n = _base_vec("n")
        base_k = _base_vec("k")
        shifted = False
        for feat, k_val in [
            ("lab", base_k[_FEAT_IDX["lab"]]),
            ("ant", base_k[_FEAT_IDX["ant"]]),
            ("cor", base_k[_FEAT_IDX["cor"]]),
        ]:
            if k_val != 0:
                base_val = float(base_n[_FEAT_IDX[feat]])
                pert_val = _feat_val(result[0], feat)
                if (k_val == 1 and pert_val > base_val) or (k_val == -1 and pert_val < base_val):
                    shifted = True
        assert shifted


class TestSyllableBoundary:
    def test_within_syllable_stronger(self):
        model = DefaultCoarticulationModel(jitter=0.0, cross_syllable_decay=0.4)
        tokens = ["m", "a"]
        nas_same = _feat_val(model.perturb_sequence(tokens, syllable_boundaries=[0, 0])[1], "nas")
        nas_cross = _feat_val(model.perturb_sequence(tokens, syllable_boundaries=[0, 1])[1], "nas")
        nas_base = float(_base_vec("a")[_FEAT_IDX["nas"]])
        assert nas_same > nas_base
        assert nas_cross > nas_base
        assert nas_same > nas_cross

    def test_zero_decay_blocks_cross_syllable(self):
        model = DefaultCoarticulationModel(jitter=0.0, cross_syllable_decay=0.0)
        result = model.perturb_sequence(["m", "a"], syllable_boundaries=[0, 1])
        nas_base = float(_base_vec("a")[_FEAT_IDX["nas"]])
        assert _feat_val(result[1], "nas") == nas_base

    def test_no_boundaries_assumes_same_syllable(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        tokens = ["m", "a"]
        assert model.perturb_sequence(tokens, syllable_boundaries=None) == model.perturb_sequence(
            tokens, syllable_boundaries=[0, 0]
        )


class TestJitter:
    def test_zero_jitter_deterministic(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        tokens = ["k", "æ", "t", "s"]
        assert model.perturb_sequence(tokens) == model.perturb_sequence(tokens)

    def test_nonzero_jitter_varies(self):
        model = DefaultCoarticulationModel(jitter=0.8, seed=None)
        tokens = ["m", "a", "n", "i", "k"]
        results = {model.perturb_sequence(tokens)[1] for _ in range(20)}
        assert len(results) > 1

    def test_seed_produces_same_jitter(self):
        tokens = ["b", "ɪ", "n", "d", "z"]
        for seed in [0, 42, 12345]:
            m1 = DefaultCoarticulationModel(jitter=0.5, seed=seed)
            m2 = DefaultCoarticulationModel(jitter=0.5, seed=seed)
            assert m1.perturb_sequence(tokens) == m2.perturb_sequence(tokens)


class TestCoarticulatedDistance:
    def test_identical_sequences_zero_distance(self):
        assert coarticulated_feature_edit_distance(
            ["k", "æ", "t"], ["k", "æ", "t"]
        ) == pytest.approx(0.0)

    def test_symmetry(self):
        d1 = coarticulated_feature_edit_distance(["k", "æ", "t"], ["k", "æ", "b"])
        d2 = coarticulated_feature_edit_distance(["k", "æ", "b"], ["k", "æ", "t"])
        assert d1 == pytest.approx(d2)

    def test_normalised_range(self):
        dist = normalised_coarticulated_feature_edit_distance(["k", "æ", "t"], ["m", "u", "s"])
        assert 0.0 <= dist <= 1.0

    def test_normalised_empty(self):
        assert normalised_coarticulated_feature_edit_distance([], []) == 0.0

    def test_coarticulated_distance_differs_from_plain(self):
        from phone_similarity.primitives import normalised_feature_edit_distance

        tokens_a = ["m", "i", "n"]
        tokens_b = ["b", "i", "n"]
        feats = UniversalFeatureEncoder.merge_inventories(
            {ph: {} for ph in set(tokens_a) | set(tokens_b)}
        )
        plain = normalised_feature_edit_distance(tokens_a, tokens_b, feats)
        coart = normalised_coarticulated_feature_edit_distance(tokens_a, tokens_b)
        assert plain != pytest.approx(coart, abs=1e-6)

    def test_phoneme_distance_skips_zero(self):
        a = (0.0, 1.0, -1.0) + (0.0,) * (_NUM_FEATURES - 3)
        b = (0.0, -1.0, 1.0) + (0.0,) * (_NUM_FEATURES - 3)
        assert coarticulated_phoneme_distance(a, b) == pytest.approx(1.0)


class TestSyllableIntegration:
    def test_syllabify_then_coarticulate(self):
        from phone_similarity.syllable import syllabify

        tokens = ["b", "æ", "n", "æ", "n", "ə"]
        syllables = syllabify(tokens, frozenset({"æ", "ə"}))
        boundaries = DefaultCoarticulationModel.syllable_boundary_map(syllables)
        result = DefaultCoarticulationModel(jitter=0.0).perturb_sequence(
            tokens, syllable_boundaries=boundaries
        )
        assert len(result) == 6
        for vec in result:
            assert len(vec) == _NUM_FEATURES


class TestCoarticulationRule:
    def test_rule_fields(self):
        rule = CoarticulationRule("test", 0, 1.0, 0.5)
        assert rule.base_probability == 0.8
        assert rule.within_syllable_only is False
        rule2 = CoarticulationRule(
            "custom", 3, -1.0, 0.3, base_probability=0.5, within_syllable_only=True
        )
        assert rule2.base_probability == 0.5
        assert rule2.within_syllable_only is True


class TestRealWorldScenarios:
    def test_french_nasalization(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["m", "a"])
        nas_base = float(_base_vec("a")[_FEAT_IDX["nas"]])
        assert _feat_val(result[1], "nas") > nas_base

    def test_german_auslautverhaertung_context(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["d", "s"])
        voi_base = float(_base_vec("d")[_FEAT_IDX["voi"]])
        assert _feat_val(result[0], "voi") < voi_base

    def test_pun_distance_closer_with_coarticulation(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        d = coarticulated_feature_edit_distance(["k", "æ", "t"], ["ɡ", "æ", "t"], model=model)
        assert d < 0.5


class TestFricativeConfig:
    def test_defaults_and_custom(self):
        fc = FricativeConfig()
        assert fc.fricative_weight == 1.0
        assert fc.sibilant_weight == 1.0
        assert fc.frication_spread is False
        assert fc.spread_magnitude == 0.30
        fc2 = FricativeConfig(
            fricative_weight=2.0, sibilant_weight=1.5, frication_spread=True, spread_magnitude=0.50
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


class TestFricativeConfigModel:
    def test_config_lifecycle(self):
        model = DefaultCoarticulationModel()
        assert model.fricative_config.fricative_weight == 1.0
        assert model.fricative_config.frication_spread is False
        fc = FricativeConfig(fricative_weight=2.0)
        assert (
            DefaultCoarticulationModel(fricative_config=fc).fricative_config.fricative_weight
            == 2.0
        )
        model.fricative_config = FricativeConfig(fricative_weight=3.0)
        assert model.fricative_config.fricative_weight == 3.0
        with pytest.raises(TypeError, match="FricativeConfig"):
            model.fricative_config = "not a config"  # type: ignore[assignment]


class TestFricativeWeighting:
    def test_weight_scales_fricative_distance(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        d_zero = coarticulated_feature_edit_distance(
            ["s", "a"],
            ["t", "a"],
            model=model,
            fricative_config=FricativeConfig(fricative_weight=0.0),
        )
        d_normal = coarticulated_feature_edit_distance(
            ["s", "a"],
            ["t", "a"],
            model=model,
            fricative_config=FricativeConfig(fricative_weight=1.0),
        )
        d_heavy = coarticulated_feature_edit_distance(
            ["s", "æ", "t"],
            ["t", "æ", "t"],
            model=model,
            fricative_config=FricativeConfig(fricative_weight=3.0),
        )
        d_normal_sat = coarticulated_feature_edit_distance(
            ["s", "æ", "t"],
            ["t", "æ", "t"],
            model=model,
            fricative_config=FricativeConfig(fricative_weight=1.0),
        )
        assert d_zero < d_normal
        assert d_heavy > d_normal_sat

    def test_sibilant_weight_affects_sibilant_pairs(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        d_normal = coarticulated_feature_edit_distance(
            ["s", "a"],
            ["θ", "a"],
            model=model,
            fricative_config=FricativeConfig(sibilant_weight=1.0),
        )
        d_heavy = coarticulated_feature_edit_distance(
            ["s", "a"],
            ["θ", "a"],
            model=model,
            fricative_config=FricativeConfig(sibilant_weight=3.0),
        )
        assert d_heavy > d_normal

    def test_non_fricative_pair_unaffected(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        d_normal = coarticulated_feature_edit_distance(
            ["p", "a"],
            ["b", "a"],
            model=model,
            fricative_config=FricativeConfig(fricative_weight=1.0),
        )
        d_heavy = coarticulated_feature_edit_distance(
            ["p", "a"],
            ["b", "a"],
            model=model,
            fricative_config=FricativeConfig(fricative_weight=5.0),
        )
        assert d_normal == pytest.approx(d_heavy)


class TestFricationSpread:
    def test_frication_spread_off_by_default(self):
        model = DefaultCoarticulationModel(jitter=0.0)
        result = model.perturb_sequence(["s", "a"])
        voi_base = float(_base_vec("a")[_FEAT_IDX["voi"]])
        assert _feat_val(result[1], "voi") == voi_base
        model2 = DefaultCoarticulationModel(
            jitter=0.0, fricative_config=FricativeConfig(frication_spread=True)
        )
        result2 = model2.perturb_sequence(["t", "a"])
        cont_base = float(_base_vec("a")[_FEAT_IDX["cont"]])
        assert _feat_val(result2[1], "cont") == cont_base

    def test_frication_spread_devoices_vowel(self):
        model = DefaultCoarticulationModel(
            jitter=0.0,
            fricative_config=FricativeConfig(frication_spread=True, spread_magnitude=0.30),
        )
        result = model.perturb_sequence(["s", "a"])
        assert _feat_val(result[1], "voi") < float(_base_vec("a")[_FEAT_IDX["voi"]])

    def test_sibilant_spreads_stridency(self):
        fc = FricativeConfig(frication_spread=True, spread_magnitude=0.30)
        model = DefaultCoarticulationModel(jitter=0.0, fricative_config=fc)
        strid_base = float(_base_vec("a")[_FEAT_IDX["strid"]])
        voi_base = float(_base_vec("a")[_FEAT_IDX["voi"]])
        result_s = model.perturb_sequence(["s", "a"])
        assert _feat_val(result_s[1], "strid") > strid_base
        result_th = model.perturb_sequence(["θ", "a"])
        assert _feat_val(result_th[1], "strid") == strid_base
        assert _feat_val(result_th[1], "voi") < voi_base

    def test_spread_magnitude_scales_effect(self):
        m_low = DefaultCoarticulationModel(
            jitter=0.0,
            fricative_config=FricativeConfig(frication_spread=True, spread_magnitude=0.10),
        )
        m_high = DefaultCoarticulationModel(
            jitter=0.0,
            fricative_config=FricativeConfig(frication_spread=True, spread_magnitude=0.60),
        )
        voi_low = _feat_val(m_low.perturb_sequence(["s", "a"])[1], "voi")
        voi_high = _feat_val(m_high.perturb_sequence(["s", "a"])[1], "voi")
        assert voi_high < voi_low
