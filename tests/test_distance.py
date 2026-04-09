"""
Tests for the phonological distance / similarity API.

Tests cover:
1. Low-level primitives (hamming_distance, hamming_similarity,
   phoneme_feature_distance, feature_edit_distance)
2. High-level Distance class (hamming, edit_distance, pairwise methods)
3. Cython / pure-Python parity (both implementations must agree)
"""

import pytest
from bitarray import bitarray

from phone_similarity.distance import (
    Distance,
    batch_pairwise_hamming,
    feature_edit_distance,
    hamming_distance,
    hamming_similarity,
    normalised_feature_edit_distance,
    phoneme_feature_distance,
)

# ===================================================================
# Fixtures
# ===================================================================

SAMPLE_FEATURES = {
    "p": {"voiced": False, "manner": "plosive", "labial": True},
    "b": {"voiced": True, "manner": "plosive", "labial": True},
    "t": {"voiced": False, "manner": "plosive", "place": "alveolar"},
    "d": {"voiced": True, "manner": "plosive", "place": "alveolar"},
    "k": {"voiced": False, "manner": "plosive", "place": "velar"},
    "ɡ": {"voiced": True, "manner": "plosive", "place": "velar"},
    "s": {"voiced": False, "manner": "fricative", "place": "alveolar"},
    "z": {"voiced": True, "manner": "fricative", "place": "alveolar"},
    "m": {"voiced": True, "manner": "nasal", "labial": True},
    "n": {"voiced": True, "manner": "nasal", "place": "alveolar"},
    "æ": {"low": True, "front": True, "round": False},
    "ɛ": {"mid-low": True, "front": True, "round": False},
    "i": {"high": True, "front": True, "round": False},
    "ɑ": {"low": True, "back": True, "round": False},
    "u": {"high": True, "back": True, "round": True},
}


# ===================================================================
# hamming_distance / hamming_similarity
# ===================================================================


class TestHammingDistance:
    def test_identical(self):
        a = bitarray("10110100")
        assert hamming_distance(a, a) == 0

    def test_completely_different(self):
        a = bitarray("1111")
        b = bitarray("0000")
        assert hamming_distance(a, b) == 4

    def test_one_bit_diff(self):
        a = bitarray("1000")
        b = bitarray("0000")
        assert hamming_distance(a, b) == 1

    def test_empty(self):
        a = bitarray()
        assert hamming_distance(a, a) == 0

    def test_unequal_length_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            hamming_distance(bitarray("10"), bitarray("100"))


class TestHammingSimilarity:
    def test_identical(self):
        a = bitarray("10110100")
        assert hamming_similarity(a, a) == 1.0

    def test_completely_different(self):
        a = bitarray("1111")
        b = bitarray("0000")
        assert hamming_similarity(a, b) == 0.0

    def test_half_different(self):
        a = bitarray("1100")
        b = bitarray("0011")
        assert hamming_similarity(a, b) == 0.0

    def test_one_bit_diff(self):
        a = bitarray("1000")
        b = bitarray("0000")
        assert hamming_similarity(a, b) == 0.75

    def test_empty_returns_one(self):
        assert hamming_similarity(bitarray(), bitarray()) == 1.0


# ===================================================================
# phoneme_feature_distance
# ===================================================================


class TestPhonemeFeatureDistance:
    def test_identical(self):
        assert phoneme_feature_distance(SAMPLE_FEATURES["p"], SAMPLE_FEATURES["p"]) == 0.0

    def test_voiced_pair(self):
        """p and b differ only in voicing -- 1 out of 3 features."""
        d = phoneme_feature_distance(SAMPLE_FEATURES["p"], SAMPLE_FEATURES["b"])
        assert d == pytest.approx(1 / 3)

    def test_maximally_different(self):
        """Two phonemes with no overlapping features."""
        a = {"voiced": True}
        b = {"manner": "plosive"}
        # keys = {voiced, manner}, both mismatch -> 2/2 = 1.0
        assert phoneme_feature_distance(a, b) == 1.0

    def test_empty(self):
        assert phoneme_feature_distance({}, {}) == 0.0


# ===================================================================
# feature_edit_distance
# ===================================================================


class TestFeatureEditDistance:
    def test_identical_sequences(self):
        assert feature_edit_distance(["k", "æ", "t"], ["k", "æ", "t"], SAMPLE_FEATURES) == 0.0

    def test_single_substitution(self):
        """cat -> bat: k->b substitution."""
        ed = feature_edit_distance(["k", "æ", "t"], ["b", "æ", "t"], SAMPLE_FEATURES)
        assert ed > 0
        assert ed < 1.0  # Not a full mismatch since manner is shared

    def test_insertion(self):
        """cat -> cats: one insertion."""
        ed = feature_edit_distance(["k", "æ", "t"], ["k", "æ", "t", "s"], SAMPLE_FEATURES)
        assert ed == pytest.approx(1.0)  # Default insert cost

    def test_deletion(self):
        """cats -> cat: one deletion."""
        ed = feature_edit_distance(["k", "æ", "t", "s"], ["k", "æ", "t"], SAMPLE_FEATURES)
        assert ed == pytest.approx(1.0)

    def test_empty_sequences(self):
        assert feature_edit_distance([], [], SAMPLE_FEATURES) == 0.0

    def test_one_empty(self):
        ed = feature_edit_distance(["k", "æ", "t"], [], SAMPLE_FEATURES)
        assert ed == pytest.approx(3.0)  # 3 deletions

    def test_similar_vs_dissimilar(self):
        """Minimal pair (p->b, voicing only) should cost less than
        different place + voicing (p->d)."""
        ed_pb = feature_edit_distance(["p"], ["b"], SAMPLE_FEATURES)
        ed_pd = feature_edit_distance(["p"], ["d"], SAMPLE_FEATURES)
        assert ed_pb < ed_pd

    def test_symmetry(self):
        ed_ab = feature_edit_distance(["k", "æ", "t"], ["b", "æ", "t"], SAMPLE_FEATURES)
        ed_ba = feature_edit_distance(["b", "æ", "t"], ["k", "æ", "t"], SAMPLE_FEATURES)
        assert ed_ab == pytest.approx(ed_ba)


class TestNormalisedFeatureEditDistance:
    def test_identical(self):
        assert (
            normalised_feature_edit_distance(["k", "æ", "t"], ["k", "æ", "t"], SAMPLE_FEATURES)
            == 0.0
        )

    def test_range_01(self):
        d = normalised_feature_edit_distance(["k", "æ", "t"], ["b", "ɑ", "d"], SAMPLE_FEATURES)
        assert 0.0 <= d <= 1.0

    def test_empty(self):
        assert normalised_feature_edit_distance([], [], SAMPLE_FEATURES) == 0.0


# ===================================================================
# batch_pairwise_hamming
# ===================================================================


class TestBatchPairwiseHamming:
    def test_two_identical(self):
        a = bitarray("1010")
        mat = batch_pairwise_hamming([a, a])
        assert mat[0][0] == 1.0
        assert mat[0][1] == 1.0
        assert mat[1][0] == 1.0

    def test_two_opposite(self):
        a = bitarray("1111")
        b = bitarray("0000")
        mat = batch_pairwise_hamming([a, b])
        assert mat[0][1] == 0.0
        assert mat[1][0] == 0.0

    def test_symmetry(self):
        a = bitarray("1010")
        b = bitarray("1100")
        c = bitarray("0011")
        mat = batch_pairwise_hamming([a, b, c])
        for i in range(3):
            for j in range(3):
                assert mat[i][j] == pytest.approx(mat[j][i])

    def test_diagonal_is_one(self):
        arrays = [bitarray("1010"), bitarray("0101"), bitarray("1100")]
        mat = batch_pairwise_hamming(arrays)
        for i in range(3):
            assert mat[i][i] == 1.0

    def test_empty_list(self):
        assert batch_pairwise_hamming([]) == []


# ===================================================================
# Cython / pure-Python parity
# ===================================================================


class TestCythonParity:
    """Verify that Cython and pure-Python implementations produce identical results."""

    def test_hamming_parity(self):
        try:
            from phone_similarity._core import hamming_similarity as cy_sim
        except ImportError:
            pytest.skip("Cython extension not compiled")
        a = bitarray("10110100110011001010101010101010")
        b = bitarray("10010110100011101010101000101110")
        py_val = hamming_similarity(a, b)
        cy_val = cy_sim(a, b)
        assert py_val == pytest.approx(cy_val)

    def test_edit_distance_parity(self):
        try:
            from phone_similarity._core import feature_edit_distance as cy_ed
        except ImportError:
            pytest.skip("Cython extension not compiled")
        seq_a = ["k", "æ", "t", "s"]
        seq_b = ["b", "ɛ", "d", "z"]
        py_val = feature_edit_distance(seq_a, seq_b, SAMPLE_FEATURES)
        cy_val = cy_ed(seq_a, seq_b, SAMPLE_FEATURES)
        assert py_val == pytest.approx(cy_val)

    def test_batch_pairwise_parity(self):
        try:
            from phone_similarity._core import batch_pairwise_hamming as cy_batch
        except ImportError:
            pytest.skip("Cython extension not compiled")
        arrays = [bitarray("10100110"), bitarray("01011001"), bitarray("11110000")]
        py_mat = batch_pairwise_hamming(arrays)
        cy_mat = cy_batch(arrays)
        for i in range(3):
            for j in range(3):
                assert py_mat[i][j] == pytest.approx(cy_mat[i][j])


# ===================================================================
# High-level Distance class (integration)
# ===================================================================


class TestDistanceClass:
    @pytest.fixture
    def eng_distance(self):
        from phone_similarity.bit_array_specification import BitArraySpecification
        from phone_similarity.language import LANGUAGES

        lang = LANGUAGES["eng_us"]
        spec = BitArraySpecification(
            vowels=lang.VOWELS_SET,
            consonants=set(lang.PHONEME_FEATURES) - lang.VOWELS_SET,
            features_per_phoneme=lang.PHONEME_FEATURES,
            features=lang.FEATURES,
        )
        return Distance(spec)

    def test_hamming_identical(self, eng_distance):
        assert eng_distance.hamming("kæt", "kæt") == 1.0

    def test_hamming_different(self, eng_distance):
        sim = eng_distance.hamming("kæt", "dɔɡ")
        assert 0.0 < sim < 1.0

    def test_edit_distance_identical(self, eng_distance):
        assert eng_distance.edit_distance("kæt", "kæt") == 0.0

    def test_edit_distance_positive(self, eng_distance):
        ed = eng_distance.edit_distance("kæt", "hæt")
        assert ed > 0

    def test_normalised_edit_distance_range(self, eng_distance):
        d = eng_distance.normalised_edit_distance("kæt", "dɔɡ")
        assert 0.0 <= d <= 1.0

    def test_pairwise_hamming_shape(self, eng_distance):
        mat = eng_distance.pairwise_hamming(["kæt", "hæt", "dɔɡ"])
        assert len(mat) == 3
        assert all(len(row) == 3 for row in mat)

    def test_pairwise_edit_distance_shape(self, eng_distance):
        mat = eng_distance.pairwise_edit_distance(["kæt", "hæt", "dɔɡ"])
        assert len(mat) == 3
        assert all(len(row) == 3 for row in mat)

    def test_pairwise_edit_distance_diagonal_zero(self, eng_distance):
        mat = eng_distance.pairwise_edit_distance(["kæt", "hæt"])
        assert mat[0][0] == 0.0
        assert mat[1][1] == 0.0

    def test_similar_closer_than_dissimilar(self, eng_distance):
        """'cat' and 'hat' should be closer than 'cat' and 'dog'."""
        d_cat_hat = eng_distance.normalised_edit_distance("kæt", "hæt")
        d_cat_dog = eng_distance.normalised_edit_distance("kæt", "dɔɡ")
        assert d_cat_hat < d_cat_dog
