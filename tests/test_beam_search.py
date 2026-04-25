"""
Tests for the beam search multi-word segmentation module.

Tests cover:
1. Basic segmentation correctness (single word, multi-word)
2. Scoring and ranking (best segmentation first)
3. Pruning (max_words, max_distance)
4. Edge cases (empty inputs, no matches)
5. Batch interface (beam_search_phrases)
6. BeamResult dataclass
"""

import pytest

from phone_similarity.beam_search import (
    BeamResult,
    BeamSearchResources,
    _Hypothesis,
    beam_search_phrases,
    beam_search_segmentation,
    build_beam_search_resources,
)
from phone_similarity.pretokenize import PreTokenizedDictionary

# Shared fixtures

# Minimal phoneme feature set for testing
FEATURES = {
    "k": {"voiced": False, "manner": "plosive", "place": "velar"},
    "ɡ": {"voiced": True, "manner": "plosive", "place": "velar"},
    "t": {"voiced": False, "manner": "plosive", "place": "alveolar"},
    "d": {"voiced": True, "manner": "plosive", "place": "alveolar"},
    "p": {"voiced": False, "manner": "plosive", "labial": True},
    "b": {"voiced": True, "manner": "plosive", "labial": True},
    "s": {"voiced": False, "manner": "fricative", "place": "alveolar"},
    "z": {"voiced": True, "manner": "fricative", "place": "alveolar"},
    "m": {"voiced": True, "manner": "nasal", "labial": True},
    "n": {"voiced": True, "manner": "nasal", "place": "alveolar"},
    "æ": {"low": True, "front": True, "round": False},
    "ɛ": {"mid-low": True, "front": True, "round": False},
    "i": {"high": True, "front": True, "round": False},
    "ɑ": {"low": True, "back": True, "round": False},
    "u": {"high": True, "back": True, "round": True},
    "a": {"low": True, "front": True, "round": False},
    "o": {"mid": True, "back": True, "round": True},
    "e": {"mid": True, "front": True, "round": False},
}


@pytest.fixture
def mock_spec():
    """Create a minimal BitArraySpecification for testing."""
    from phone_similarity.bit_array_specification import BitArraySpecification

    vowels = {"æ", "ɛ", "i", "ɑ", "u", "a", "o", "e"}
    consonants = {"k", "ɡ", "t", "d", "p", "b", "s", "z", "m", "n"}
    features = {
        "consonant": {
            "voiced",
            "manner",
            "place",
            "labial",
        },
        "vowel": {
            "low",
            "mid-low",
            "mid",
            "high",
            "front",
            "back",
            "round",
        },
    }
    return BitArraySpecification(
        vowels=vowels,
        consonants=consonants,
        features=features,
        features_per_phoneme=FEATURES,
    )


@pytest.fixture
def small_ptd():
    """A small pre-tokenized dictionary for beam search testing."""
    entries = [
        ("cat", "kæt", ["k", "æ", "t"]),
        ("bat", "bæt", ["b", "æ", "t"]),
        ("sat", "sæt", ["s", "æ", "t"]),
        ("kit", "kit", ["k", "i", "t"]),
        ("sit", "sit", ["s", "i", "t"]),
        ("mat", "mæt", ["m", "æ", "t"]),
        ("an", "æn", ["æ", "n"]),
        ("in", "in", ["i", "n"]),
        ("at", "æt", ["æ", "t"]),
        ("us", "us", ["u", "s"]),
        ("dot", "dɑt", ["d", "ɑ", "t"]),
    ]
    return PreTokenizedDictionary.from_entries(entries)


# _Hypothesis ordering


class TestHypothesis:
    def test_ordering_by_score(self):
        h1 = _Hypothesis(score=0.1, consumed=3, words=("a",), ipas=("x",), raw_cost=0.3)
        h2 = _Hypothesis(score=0.5, consumed=3, words=("b",), ipas=("y",), raw_cost=1.5)
        assert h1 < h2

    def test_equal_score(self):
        h1 = _Hypothesis(score=0.3, consumed=3, words=("a",), ipas=("x",), raw_cost=0.9)
        h2 = _Hypothesis(score=0.3, consumed=3, words=("b",), ipas=("y",), raw_cost=0.9)
        assert not (h1 < h2)
        assert not (h2 < h1)


# BeamResult


# beam_search_segmentation — basic


class TestBeamSearchSegmentation:
    def test_exact_single_word_match(self, mock_spec, small_ptd):
        """Source 'kæt' should find 'cat' with distance ~0."""
        results = beam_search_segmentation(
            ["k", "æ", "t"],
            FEATURES,
            small_ptd,
            mock_spec,
            FEATURES,
            beam_width=5,
            top_k=3,
            max_distance=0.5,
        )
        assert len(results) > 0
        best = results[0]
        assert best.words == ("cat",)
        assert best.distance == pytest.approx(0.0)

    def test_close_single_word_match(self, mock_spec, small_ptd):
        """Source 'bæt' should find 'bat' exactly, 'cat' close."""
        results = beam_search_segmentation(
            ["b", "æ", "t"],
            FEATURES,
            small_ptd,
            mock_spec,
            FEATURES,
            beam_width=10,
            top_k=5,
            max_distance=0.5,
        )
        assert len(results) >= 1
        words_found = [r.words for r in results]
        assert ("bat",) in words_found
        # 'bat' should be best (exact match)
        assert results[0].words == ("bat",)
        assert results[0].distance == pytest.approx(0.0)

    def test_multi_word_segmentation(self, mock_spec, small_ptd):
        """Source 'kætsæt' (cat+sat) should find a 2-word segmentation."""
        results = beam_search_segmentation(
            ["k", "æ", "t", "s", "æ", "t"],
            FEATURES,
            small_ptd,
            mock_spec,
            FEATURES,
            beam_width=20,
            top_k=5,
            max_words=4,
            max_distance=0.5,
        )
        assert len(results) > 0
        # Should find the exact 2-word split
        two_word = [r for r in results if len(r.words) == 2]
        assert len(two_word) > 0
        best_two = two_word[0]
        assert best_two.distance == pytest.approx(0.0, abs=0.05)

    def test_returns_sorted_by_distance(self, mock_spec, small_ptd):
        results = beam_search_segmentation(
            ["k", "æ", "t"],
            FEATURES,
            small_ptd,
            mock_spec,
            FEATURES,
            beam_width=10,
            top_k=5,
            max_distance=1.0,
        )
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].distance <= results[i + 1].distance

    def test_empty_source_returns_empty(self, mock_spec, small_ptd):
        results = beam_search_segmentation(
            [],
            FEATURES,
            small_ptd,
            mock_spec,
            FEATURES,
        )
        assert results == []

    def test_empty_dictionary(self, mock_spec):
        empty_ptd = PreTokenizedDictionary.from_entries([])
        results = beam_search_segmentation(
            ["k", "æ", "t"],
            FEATURES,
            empty_ptd,
            mock_spec,
            FEATURES,
        )
        assert results == []


# beam_search_segmentation — pruning


class TestBeamSearchPruning:
    def test_max_words_respected(self, mock_spec, small_ptd):
        """With max_words=1, only single-word segmentations should appear."""
        results = beam_search_segmentation(
            ["k", "æ", "t", "s", "æ", "t"],
            FEATURES,
            small_ptd,
            mock_spec,
            FEATURES,
            beam_width=20,
            top_k=10,
            max_words=1,
            max_distance=1.0,
        )
        for r in results:
            assert len(r.words) <= 1

    def test_max_distance_filters(self, mock_spec, small_ptd):
        """Very tight distance threshold should reduce results."""
        tight = beam_search_segmentation(
            ["k", "æ", "t"],
            FEATURES,
            small_ptd,
            mock_spec,
            FEATURES,
            beam_width=10,
            top_k=10,
            max_distance=0.001,
        )
        loose = beam_search_segmentation(
            ["k", "æ", "t"],
            FEATURES,
            small_ptd,
            mock_spec,
            FEATURES,
            beam_width=10,
            top_k=10,
            max_distance=1.0,
        )
        assert len(tight) <= len(loose)

    def test_beam_width_one(self, mock_spec, small_ptd):
        """Beam width 1 should still find something (greedy)."""
        results = beam_search_segmentation(
            ["k", "æ", "t"],
            FEATURES,
            small_ptd,
            mock_spec,
            FEATURES,
            beam_width=1,
            top_k=1,
            max_distance=1.0,
        )
        assert len(results) >= 1


# beam_search_segmentation — quality checks


class TestBeamSearchQuality:
    def test_exact_match_has_zero_distance(self, mock_spec, small_ptd):
        """An exact match in the dictionary should have distance ~0."""
        results = beam_search_segmentation(
            ["s", "i", "t"],
            FEATURES,
            small_ptd,
            mock_spec,
            FEATURES,
            beam_width=10,
            top_k=1,
            max_distance=0.5,
        )
        assert len(results) > 0
        assert results[0].words == ("sit",)
        assert results[0].distance == pytest.approx(0.0)

    def test_voiced_pair_has_low_distance(self, mock_spec, small_ptd):
        """'kæt' vs 'ɡæt' (not in dict) — 'cat' should still match closely."""
        # ɡæt isn't in the dict, but kæt→cat should match well
        results = beam_search_segmentation(
            ["ɡ", "æ", "t"],  # voiced version of 'cat'
            FEATURES,
            small_ptd,
            mock_spec,
            FEATURES,
            beam_width=10,
            top_k=3,
            max_distance=0.5,
        )
        assert len(results) > 0
        # cat should be among the top results
        cat_results = [r for r in results if "cat" in r.words]
        assert len(cat_results) > 0
        # Distance should be low (only voicing differs)
        assert cat_results[0].distance < 0.4

    def test_deduplication(self, mock_spec, small_ptd):
        """Same word-tuple shouldn't appear multiple times in results."""
        results = beam_search_segmentation(
            ["k", "æ", "t"],
            FEATURES,
            small_ptd,
            mock_spec,
            FEATURES,
            beam_width=20,
            top_k=10,
            max_distance=1.0,
        )
        word_tuples = [r.words for r in results]
        assert len(word_tuples) == len(set(word_tuples))


# beam_search_phrases — batch interface


class TestBeamSearchPhrases:
    def test_single_phrase_single_language(self, mock_spec, small_ptd):
        phrases = [("test_key", "kæt")]
        targets = {"test_lang": (mock_spec, FEATURES, small_ptd)}

        results = beam_search_phrases(
            phrases,
            mock_spec,
            FEATURES,
            targets,
            beam_width=5,
            top_k=1,
            max_distance=0.5,
        )
        assert len(results) > 0
        key, lang, beam_result = results[0]
        assert key == "test_key"
        assert lang == "test_lang"
        assert isinstance(beam_result, BeamResult)

    def test_multiple_phrases(self, mock_spec, small_ptd):
        phrases = [
            ("p1", "kæt"),
            ("p2", "bæt"),
        ]
        targets = {"test_lang": (mock_spec, FEATURES, small_ptd)}

        results = beam_search_phrases(
            phrases,
            mock_spec,
            FEATURES,
            targets,
            beam_width=5,
            top_k=1,
            max_distance=0.5,
        )
        keys = {r[0] for r in results}
        assert "p1" in keys
        assert "p2" in keys

    def test_results_sorted_by_distance(self, mock_spec, small_ptd):
        phrases = [
            ("exact", "kæt"),  # exact match -> dist ~0
            ("approx", "ɡæt"),  # close match -> dist > 0
        ]
        targets = {"test_lang": (mock_spec, FEATURES, small_ptd)}

        results = beam_search_phrases(
            phrases,
            mock_spec,
            FEATURES,
            targets,
            beam_width=10,
            top_k=1,
            max_distance=0.5,
        )
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i][2].distance <= results[i + 1][2].distance

    def test_empty_phrases(self, mock_spec, small_ptd):
        results = beam_search_phrases(
            [],
            mock_spec,
            FEATURES,
            {"test_lang": (mock_spec, FEATURES, small_ptd)},
        )
        assert results == []

    def test_empty_targets(self, mock_spec):
        results = beam_search_phrases(
            [("p1", "kæt")],
            mock_spec,
            FEATURES,
            {},
        )
        assert results == []


# Integration test with real language data


class TestBeamSearchIntegration:
    """Integration tests using real eng_us language data (if available)."""

    @pytest.fixture
    def eng_spec(self):
        try:
            from phone_similarity.bit_array_specification import BitArraySpecification
            from phone_similarity.language import LANGUAGES

            lang = LANGUAGES["eng_us"]
            spec = BitArraySpecification(
                vowels=lang.VOWELS_SET,
                consonants=set(lang.PHONEME_FEATURES) - lang.VOWELS_SET,
                features_per_phoneme=lang.PHONEME_FEATURES,
                features=lang.FEATURES,
            )
            return spec, lang.PHONEME_FEATURES
        except Exception:
            pytest.skip("eng_us language module not available")

    def test_real_language_single_word(self, eng_spec):
        spec, features = eng_spec
        entries = [
            ("cat", "kæt", ["k", "æ", "t"]),
            ("hat", "hæt", ["h", "æ", "t"]),
            ("bat", "bæt", ["b", "æ", "t"]),
        ]
        ptd = PreTokenizedDictionary.from_entries(entries)

        source_tokens = spec.ipa_tokenizer("kæt")
        results = beam_search_segmentation(
            source_tokens,
            features,
            ptd,
            spec,
            features,
            beam_width=5,
            top_k=3,
            max_distance=0.5,
        )
        assert len(results) > 0
        assert results[0].words == ("cat",)
        assert results[0].distance == pytest.approx(0.0)


class TestBeamSearchResources:
    def test_precomputed_resources_match_default_path(self, mock_spec, small_ptd):
        source = ["k", "æ", "t"]
        baseline = beam_search_segmentation(
            source,
            FEATURES,
            small_ptd,
            mock_spec,
            FEATURES,
            beam_width=10,
            top_k=5,
            max_distance=1.0,
            min_target_tokens=1,
        )

        resources = build_beam_search_resources(
            FEATURES,
            small_ptd,
            FEATURES,
            min_target_tokens=1,
        )
        assert isinstance(resources, BeamSearchResources)

        precomputed = beam_search_segmentation(
            source,
            FEATURES,
            small_ptd,
            mock_spec,
            FEATURES,
            beam_width=10,
            top_k=5,
            max_distance=1.0,
            min_target_tokens=1,
            resources=resources,
        )

        assert [r.words for r in precomputed] == [r.words for r in baseline]
        assert [r.distance for r in precomputed] == pytest.approx([r.distance for r in baseline])

    def test_min_target_tokens_mismatch_raises(self, mock_spec, small_ptd):
        resources = build_beam_search_resources(
            FEATURES,
            small_ptd,
            FEATURES,
            min_target_tokens=2,
        )
        with pytest.raises(ValueError):
            beam_search_segmentation(
                ["k", "æ", "t"],
                FEATURES,
                small_ptd,
                mock_spec,
                FEATURES,
                min_target_tokens=1,
                resources=resources,
            )

    def test_cython_state_toggle_parity(self, monkeypatch, mock_spec, small_ptd):
        source = ["k", "æ", "t", "s", "æ", "t"]
        resources = build_beam_search_resources(
            FEATURES,
            small_ptd,
            FEATURES,
            min_target_tokens=1,
        )

        import phone_similarity.beam_search as beam_mod

        original_state = beam_mod.HAS_CYTHON_BEAM_STATE

        try:
            monkeypatch.setattr(beam_mod, "HAS_CYTHON_BEAM_STATE", False)
            py_results = beam_search_segmentation(
                source,
                FEATURES,
                small_ptd,
                mock_spec,
                FEATURES,
                beam_width=20,
                top_k=5,
                max_words=4,
                max_distance=0.5,
                min_target_tokens=1,
                resources=resources,
            )

            monkeypatch.setattr(beam_mod, "HAS_CYTHON_BEAM_STATE", original_state)
            cy_results = beam_search_segmentation(
                source,
                FEATURES,
                small_ptd,
                mock_spec,
                FEATURES,
                beam_width=20,
                top_k=5,
                max_words=4,
                max_distance=0.5,
                min_target_tokens=1,
                resources=resources,
            )
        finally:
            monkeypatch.setattr(beam_mod, "HAS_CYTHON_BEAM_STATE", original_state)

        assert [r.words for r in cy_results] == [r.words for r in py_results]
        assert [r.distance for r in cy_results] == pytest.approx([r.distance for r in py_results])

    def test_cpp_state_toggle_parity(self, monkeypatch, mock_spec, small_ptd):
        source = ["k", "æ", "t", "s", "æ", "t"]
        resources = build_beam_search_resources(
            FEATURES,
            small_ptd,
            FEATURES,
            min_target_tokens=1,
        )

        import phone_similarity.beam_search as beam_mod

        original_cpp = beam_mod.HAS_CPP_BEAM_STATE

        try:
            monkeypatch.setattr(beam_mod, "HAS_CPP_BEAM_STATE", False)
            py_results = beam_search_segmentation(
                source,
                FEATURES,
                small_ptd,
                mock_spec,
                FEATURES,
                beam_width=20,
                top_k=5,
                max_words=4,
                max_distance=0.5,
                min_target_tokens=1,
                resources=resources,
            )

            monkeypatch.setattr(beam_mod, "HAS_CPP_BEAM_STATE", original_cpp)
            cpp_results = beam_search_segmentation(
                source,
                FEATURES,
                small_ptd,
                mock_spec,
                FEATURES,
                beam_width=20,
                top_k=5,
                max_words=4,
                max_distance=0.5,
                min_target_tokens=1,
                resources=resources,
            )
        finally:
            monkeypatch.setattr(beam_mod, "HAS_CPP_BEAM_STATE", original_cpp)

        assert [r.words for r in cpp_results] == [r.words for r in py_results]
        assert [r.distance for r in cpp_results] == pytest.approx([r.distance for r in py_results])
