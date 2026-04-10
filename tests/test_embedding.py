"""Tests for the phonetic embedding module."""

from __future__ import annotations

import numpy as np
import pytest

from phone_similarity.embedding import (
    BruteForceIndex,
    PhoneticEmbedder,
    ann_dictionary_scan,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_features():
    """Minimal phoneme feature dict for testing."""
    return {
        "p": {"voiced": False, "labial": True, "manner": "plosive"},
        "b": {"voiced": True, "labial": True, "manner": "plosive"},
        "t": {"voiced": False, "labial": False, "manner": "plosive"},
        "d": {"voiced": True, "labial": False, "manner": "plosive"},
        "k": {"voiced": False, "labial": False, "manner": "plosive"},
        "s": {"voiced": False, "labial": False, "manner": "fricative"},
        "a": {"low": True, "back": True, "round": False},
        "i": {"low": False, "back": False, "round": False},
        "u": {"low": False, "back": True, "round": True},
    }


@pytest.fixture()
def embedder(sample_features):
    return PhoneticEmbedder.from_features(sample_features)


# ---------------------------------------------------------------------------
# PhoneticEmbedder tests
# ---------------------------------------------------------------------------


class TestPhoneticEmbedder:
    def test_dim_positive(self, embedder):
        assert embedder.dim > 0

    def test_embed_phoneme_known(self, embedder):
        vec = embedder.embed_phoneme("p")
        assert vec.shape == (embedder.dim,)
        assert vec.dtype == np.float32
        # "p" has features, so at least one dimension should be 1.0
        assert vec.max() == 1.0

    def test_embed_phoneme_unknown(self, embedder):
        vec = embedder.embed_phoneme("ʒ")  # not in our minimal set
        assert vec.shape == (embedder.dim,)
        # Unknown phonemes get 0.5 everywhere
        np.testing.assert_allclose(vec, 0.5, atol=1e-6)

    def test_embed_sequence_shape(self, embedder):
        vec = embedder.embed_sequence(["p", "a", "t"])
        assert vec.shape == (embedder.dim,)
        assert vec.dtype == np.float32

    def test_embed_empty_sequence(self, embedder):
        vec = embedder.embed_sequence([])
        assert vec.shape == (embedder.dim,)
        # Empty should be unknown (0.5)
        np.testing.assert_allclose(vec, 0.5, atol=1e-6)

    def test_embed_sequence_weighted(self, embedder):
        vec = embedder.embed_sequence_weighted(["p", "a", "t"])
        assert vec.shape == (embedder.dim,)

    def test_embed_batch(self, embedder):
        batch = embedder.embed_batch([["p", "a", "t"], ["k", "a", "t"]])
        assert batch.shape == (2, embedder.dim)
        assert batch.dtype == np.float32

    def test_similar_sequences_closer(self, embedder):
        """'pat' should be closer to 'bat' than to 'siku'."""
        emb_pat = embedder.embed_sequence(["p", "a", "t"])
        emb_bat = embedder.embed_sequence(["b", "a", "t"])
        emb_siku = embedder.embed_sequence(["s", "i", "k", "u"])

        dist_pat_bat = np.linalg.norm(emb_pat - emb_bat)
        dist_pat_siku = np.linalg.norm(emb_pat - emb_siku)
        assert dist_pat_bat < dist_pat_siku

    def test_identical_sequences_zero_distance(self, embedder):
        emb1 = embedder.embed_sequence(["p", "a", "t"])
        emb2 = embedder.embed_sequence(["p", "a", "t"])
        np.testing.assert_array_equal(emb1, emb2)


# ---------------------------------------------------------------------------
# BruteForceIndex tests
# ---------------------------------------------------------------------------


class TestBruteForceIndex:
    def test_from_embeddings(self, embedder):
        data = embedder.embed_batch([["p", "a"], ["t", "a"], ["k", "a"]])
        index = BruteForceIndex.from_embeddings(data)
        assert index.n_entries == 3
        assert index.dim == embedder.dim

    def test_query_returns_correct_shape(self, embedder):
        data = embedder.embed_batch([["p", "a"], ["t", "a"], ["k", "a"]])
        index = BruteForceIndex.from_embeddings(data)
        query = embedder.embed_sequence(["p", "a"])
        indices, distances = index.query(query, top_k=2)
        assert len(indices) == 2
        assert len(distances) == 2

    def test_query_nearest_is_self(self, embedder):
        seqs = [["p", "a"], ["t", "a"], ["s", "i", "k", "u"]]
        data = embedder.embed_batch(seqs)
        index = BruteForceIndex.from_embeddings(data)
        query = embedder.embed_sequence(["p", "a"])
        indices, _distances = index.query(query, top_k=1)
        assert indices[0] == 0  # "pa" is entry 0

    def test_top_k_larger_than_entries(self, embedder):
        data = embedder.embed_batch([["p", "a"], ["t", "a"]])
        index = BruteForceIndex.from_embeddings(data)
        query = embedder.embed_sequence(["p", "a"])
        indices, _distances = index.query(query, top_k=100)
        assert len(indices) == 2  # capped at n_entries

    def test_save_and_load(self, embedder, tmp_path):
        data = embedder.embed_batch([["p", "a"], ["t", "a"], ["k", "a"]])
        index = BruteForceIndex.from_embeddings(data)

        path = tmp_path / "test_index.npy"
        index.save(path)
        assert path.exists()

        loaded = BruteForceIndex.load(path)
        assert loaded.n_entries == 3
        assert loaded.dim == embedder.dim

        # Query results should be the same
        query = embedder.embed_sequence(["p", "a"])
        idx1, d1 = index.query(query, top_k=3)
        idx2, d2 = loaded.query(query, top_k=3)
        np.testing.assert_array_equal(idx1, idx2)
        np.testing.assert_allclose(d1, d2, atol=1e-5)


# ---------------------------------------------------------------------------
# ann_dictionary_scan tests (requires mock PTD)
# ---------------------------------------------------------------------------


class _MockPTD:
    """Minimal mock of PreTokenizedDictionary for testing."""

    def __init__(self, entries):
        """entries: list of (word, ipa, tokens)"""
        inventory = sorted({p for _, _, tokens in entries for p in tokens})
        inv_map = {p: i for i, p in enumerate(inventory)}
        self.words = [w for w, _, _ in entries]
        self.ipas = [ipa for _, ipa, _ in entries]
        self.inventory = inventory
        flat = []
        offsets = [0]
        for _, _, tokens in entries:
            for t in tokens:
                flat.append(inv_map[t])
            offsets.append(len(flat))
        self.token_indices = np.array(flat, dtype=np.int16)
        self.offsets = np.array(offsets, dtype=np.int32)


class TestAnnDictionaryScan:
    def test_basic_scan(self, sample_features):
        entries = [
            ("pat", "pat", ["p", "a", "t"]),
            ("bat", "bat", ["b", "a", "t"]),
            ("kit", "kit", ["k", "i", "t"]),
            ("sit", "sit", ["s", "i", "t"]),
            ("dusk", "dask", ["d", "a", "s", "k"]),
        ]
        ptd = _MockPTD(entries)
        embedder = PhoneticEmbedder.from_features(sample_features)
        index = BruteForceIndex.from_ptd(ptd, embedder)

        results = ann_dictionary_scan(
            source_tokens=["p", "a", "t"],
            ptd=ptd,
            embedder=embedder,
            index=index,
            merged_feats=sample_features,
            ann_candidates=10,
            top_n=3,
            max_distance=0.50,
        )

        assert len(results) > 0
        # "pat" should be the closest match (distance 0.0)
        assert results[0][0] == "pat"
        assert results[0][2] == pytest.approx(0.0)

    def test_empty_source(self, sample_features):
        ptd = _MockPTD([("pat", "pat", ["p", "a", "t"])])
        embedder = PhoneticEmbedder.from_features(sample_features)
        index = BruteForceIndex.from_ptd(ptd, embedder)

        results = ann_dictionary_scan(
            source_tokens=[],
            ptd=ptd,
            embedder=embedder,
            index=index,
            merged_feats=sample_features,
        )
        assert results == []
