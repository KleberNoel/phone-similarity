"""
Phonetic embedding extraction for approximate nearest-neighbor search.

Encodes IPA phoneme sequences as fixed-length dense vectors suitable for
fast candidate retrieval.  The embedding is constructed by one-hot encoding
all ``(feature_key, feature_value)`` pairs observed in the phoneme feature
dictionaries, producing a binary vector per phoneme, then averaging across
the sequence to produce a single float vector per word.

Two index implementations are provided:

* :class:`BruteForceIndex` -- pure numpy, no external dependencies
* :class:`KDTreeIndex` -- scipy ``cKDTree`` for logarithmic query time

Typical usage::

    embedder = PhoneticEmbedder.from_features(merged_feats)
    index = BruteForceIndex.from_ptd(ptd, embedder)
    candidates = index.query(embedder.embed_sequence(source_tokens), top_k=100)
    # re-rank candidates with exact feature_edit_distance
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

FeatureDict = dict[str, Union[bool, str]]
PhonemeFeatures = dict[str, FeatureDict]


# ===================================================================
# Phonetic embedder
# ===================================================================


class PhoneticEmbedder:
    """Encode phonemes and phoneme sequences as fixed-length float vectors.

    The encoding scheme is built from the union of all ``(key, value)``
    pairs across all phonemes in the provided feature dictionary.  Each
    phoneme maps to a binary vector with a 1 in each dimension that
    matches its features.

    Attributes
    ----------
    dim : int
        Dimensionality of the embedding space.
    pair_to_idx : dict
        Mapping of ``(key, value)`` → column index.
    phoneme_vectors : dict
        Pre-computed ``{phoneme: np.ndarray}`` vectors.
    """

    __slots__ = ("_unknown_vec", "dim", "pair_to_idx", "phoneme_vectors")

    def __init__(
        self,
        pair_to_idx: dict[tuple[str, Union[bool, str]], int],
        phoneme_vectors: dict[str, np.ndarray],
    ):
        self.pair_to_idx = pair_to_idx
        self.dim = len(pair_to_idx)
        self.phoneme_vectors = phoneme_vectors
        # Unknown phoneme: 0.5 in every dimension (maximally uncertain)
        self._unknown_vec = np.full(self.dim, 0.5, dtype=np.float32)

    @classmethod
    def from_features(cls, phoneme_features: PhonemeFeatures) -> PhoneticEmbedder:
        """Build an embedder from a merged phoneme-features dictionary.

        Parameters
        ----------
        phoneme_features : dict
            ``{phoneme: {feature_key: feature_value}}`` covering all
            phonemes that may be encountered (typically the merged
            source + target features).
        """
        pairs: set[tuple[str, Union[bool, str]]] = set()
        for feats in phoneme_features.values():
            for k, v in feats.items():
                pairs.add((k, v))

        pair_to_idx = {pair: i for i, pair in enumerate(sorted(pairs, key=repr))}
        dim = len(pair_to_idx)

        phoneme_vectors: dict[str, np.ndarray] = {}
        for phoneme, feats in phoneme_features.items():
            vec = np.zeros(dim, dtype=np.float32)
            for k, v in feats.items():
                idx = pair_to_idx.get((k, v))
                if idx is not None:
                    vec[idx] = 1.0
            phoneme_vectors[phoneme] = vec

        return cls(pair_to_idx, phoneme_vectors)

    def embed_phoneme(self, phoneme: str) -> np.ndarray:
        """Encode a single phoneme as a binary float32 vector."""
        return self.phoneme_vectors.get(phoneme, self._unknown_vec)

    def embed_sequence(self, tokens: list[str]) -> np.ndarray:
        """Encode a phoneme sequence as an averaged float32 vector.

        Parameters
        ----------
        tokens : list of str
            Tokenized IPA sequence.

        Returns
        -------
        np.ndarray
            Shape ``(dim,)``, dtype ``float32``.
        """
        if not tokens:
            return self._unknown_vec.copy()
        vecs = np.stack([self.embed_phoneme(t) for t in tokens])
        return vecs.mean(axis=0)

    def embed_sequence_weighted(
        self,
        tokens: list[str],
        decay: float = 0.85,
    ) -> np.ndarray:
        """Embed a sequence with exponential positional weighting.

        Earlier (onset) phonemes receive higher weight, reflecting their
        greater perceptual salience.

        Parameters
        ----------
        tokens : list of str
            Tokenized IPA sequence.
        decay : float
            Per-position decay factor (default 0.85).  Position *i* gets
            weight ``decay ** i``.

        Returns
        -------
        np.ndarray
            Shape ``(dim,)``, dtype ``float32``.
        """
        if not tokens:
            return self._unknown_vec.copy()
        n = len(tokens)
        weights = np.array([decay**i for i in range(n)], dtype=np.float32)
        weights /= weights.sum()
        vecs = np.stack([self.embed_phoneme(t) for t in tokens])
        return (vecs * weights[:, np.newaxis]).sum(axis=0)

    def embed_batch(self, token_lists: list[list[str]]) -> np.ndarray:
        """Embed multiple sequences at once.

        Parameters
        ----------
        token_lists : list of list of str

        Returns
        -------
        np.ndarray
            Shape ``(N, dim)``, dtype ``float32``.
        """
        return np.stack([self.embed_sequence(tokens) for tokens in token_lists])


# ===================================================================
# Index implementations
# ===================================================================


def _embed_ptd(ptd, embedder: PhoneticEmbedder) -> np.ndarray:
    """Embed all entries from a PreTokenizedDictionary into a matrix.

    Shared helper used by both :class:`BruteForceIndex` and
    :class:`KDTreeIndex` to avoid duplicating the PTD iteration logic.

    Returns
    -------
    np.ndarray
        Shape ``(N, embedder.dim)``, dtype ``float32``.
    """
    inv = ptd.inventory
    ti = ptd.token_indices
    off = ptd.offsets
    n = len(ptd.words)
    embs = np.empty((n, embedder.dim), dtype=np.float32)

    for i in range(n):
        start, end = int(off[i]), int(off[i + 1])
        tokens = [inv[ti[j]] for j in range(start, end)]
        embs[i] = embedder.embed_sequence(tokens)

    return embs


class BruteForceIndex:
    """Exact nearest-neighbor search over phonetic embeddings using numpy.

    Suitable for dictionaries up to ~500K entries (search time ~5-10ms
    per query on modern hardware with BLAS-accelerated numpy).

    Attributes
    ----------
    embeddings : np.ndarray
        Shape ``(N, dim)``, row-normalised L2 vectors.
    norms : np.ndarray
        Original L2 norms before normalisation.
    """

    __slots__ = ("dim", "embeddings", "n_entries", "norms")

    def __init__(self, embeddings: np.ndarray):
        self.n_entries = embeddings.shape[0]
        self.dim = embeddings.shape[1]
        self.norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
        self.embeddings = embeddings / self.norms

    @classmethod
    def from_ptd(
        cls,
        ptd,
        embedder: PhoneticEmbedder,
    ) -> BruteForceIndex:
        """Build an index from a :class:`PreTokenizedDictionary`.

        Iterates the PTD once and embeds each entry.
        """
        embs = _embed_ptd(ptd, embedder)
        logger.info(
            "Built BruteForceIndex: %d entries, %d dimensions", len(ptd.words), embedder.dim
        )
        return cls(embs)

    @classmethod
    def from_embeddings(cls, embeddings: np.ndarray) -> BruteForceIndex:
        """Wrap a pre-computed embedding matrix."""
        return cls(embeddings)

    def save(self, path) -> None:
        """Save the raw (unnormalised) embeddings to a ``.npy`` file."""
        from pathlib import Path

        path = Path(path)
        raw = self.embeddings * self.norms
        np.save(path, raw)
        logger.info("Saved BruteForceIndex (%d entries) to %s", self.n_entries, path)

    @classmethod
    def load(cls, path) -> BruteForceIndex:
        """Load from a ``.npy`` file."""
        from pathlib import Path

        path = Path(path)
        raw = np.load(path)
        logger.info("Loaded BruteForceIndex (%d entries) from %s", raw.shape[0], path)
        return cls(raw)

    def query(
        self,
        query_vec: np.ndarray,
        top_k: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the *top_k* nearest neighbors to *query_vec*.

        Uses cosine similarity (equivalent to L2 on normalised vectors).

        Parameters
        ----------
        query_vec : np.ndarray
            Shape ``(dim,)`` query embedding.
        top_k : int
            Number of neighbors to return.

        Returns
        -------
        indices : np.ndarray of int
            Indices into the original entry list.
        distances : np.ndarray of float
            Cosine distances (1 - similarity), lower = closer.
        """
        norm = np.linalg.norm(query_vec)
        if norm < 1e-8:
            # degenerate query -- return first top_k
            return np.arange(min(top_k, self.n_entries)), np.ones(min(top_k, self.n_entries))
        q = query_vec / norm

        # Cosine similarity = dot product of normalised vectors
        sims = self.embeddings @ q  # (N,)

        k = min(top_k, self.n_entries)
        if k < self.n_entries:
            part_idx = np.argpartition(sims, -k)[-k:]
        else:
            part_idx = np.arange(self.n_entries)
        top_idx = part_idx[np.argsort(-sims[part_idx])]
        distances = 1.0 - sims[top_idx]
        return top_idx, distances


class KDTreeIndex:
    """Logarithmic-time nearest-neighbor search using ``scipy.spatial.cKDTree``.

    Falls back to :class:`BruteForceIndex` if scipy is unavailable.
    """

    __slots__ = ("_fallback", "_tree", "dim", "n_entries")

    def __init__(self, embeddings: np.ndarray):
        self.n_entries = embeddings.shape[0]
        self.dim = embeddings.shape[1]
        self._fallback = None
        try:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(embeddings)
        except ImportError:
            logger.warning("scipy not available; falling back to brute-force search")
            self._tree = None
            self._fallback = BruteForceIndex(embeddings)

    @classmethod
    def from_ptd(cls, ptd, embedder: PhoneticEmbedder) -> KDTreeIndex:
        """Build from a :class:`PreTokenizedDictionary`."""
        embs = _embed_ptd(ptd, embedder)
        logger.info("Built KDTreeIndex: %d entries, %d dimensions", len(ptd.words), embedder.dim)
        return cls(embs)

    def query(
        self,
        query_vec: np.ndarray,
        top_k: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the *top_k* nearest neighbors (L2 distance).

        Parameters
        ----------
        query_vec : np.ndarray
            Shape ``(dim,)`` query embedding.
        top_k : int
            Number of neighbors to return.

        Returns
        -------
        indices : np.ndarray of int
        distances : np.ndarray of float
            L2 distances, lower = closer.
        """
        if self._fallback is not None:
            return self._fallback.query(query_vec, top_k)
        k = min(top_k, self.n_entries)
        distances, indices = self._tree.query(query_vec, k=k)
        if k == 1:
            distances = np.array([distances])
            indices = np.array([indices])
        return indices, distances


# ===================================================================
# Integrated scan with ANN pre-filter
# ===================================================================


def ann_dictionary_scan(
    source_tokens: list[str],
    ptd,
    embedder: PhoneticEmbedder,
    index: BruteForceIndex | KDTreeIndex,
    merged_feats: PhonemeFeatures,
    *,
    ann_candidates: int = 200,
    top_n: int = 10,
    max_distance: float = 0.50,
) -> list[tuple[str, str, float]]:
    """Scan a dictionary using ANN pre-filtering + exact re-ranking.

    1. Embed the source token sequence.
    2. Retrieve *ann_candidates* approximate neighbors from the index.
    3. Re-rank with exact normalised feature edit distance.

    This reduces per-query cost from O(dict_size) to O(ann_candidates).

    Parameters
    ----------
    source_tokens : list of str
        Tokenized IPA of the source phrase.
    ptd : PreTokenizedDictionary
        The target dictionary.
    embedder : PhoneticEmbedder
        The embedder used to build the index.
    index : BruteForceIndex or KDTreeIndex
        Pre-built ANN index over the PTD embeddings.
    merged_feats : dict
        Merged source + target phoneme features.
    ann_candidates : int
        Number of ANN candidates to retrieve (default 200).
    top_n : int
        Return at most this many final results.
    max_distance : float
        Distance threshold for the exact re-ranking stage.

    Returns
    -------
    list of (word, ipa, distance)
        Sorted by ascending normalised feature edit distance.
    """
    from phone_similarity.primitives import normalised_feature_edit_distance

    source_len = len(source_tokens)
    if source_len == 0:
        return []

    # 1. ANN retrieval
    query_vec = embedder.embed_sequence(source_tokens)
    candidate_idx, _ = index.query(query_vec, top_k=ann_candidates)

    # 2. Exact re-ranking
    inv = ptd.inventory
    ti = ptd.token_indices
    off = ptd.offsets

    candidates: list[tuple[str, str, float]] = []
    for idx in candidate_idx:
        idx = int(idx)
        start, end = int(off[idx]), int(off[idx + 1])
        target_len = end - start
        if target_len == 0:
            continue

        ratio = max(source_len, target_len) / min(source_len, target_len)
        if ratio > 2.0:
            continue

        target_tokens = [inv[ti[j]] for j in range(start, end)]
        d = normalised_feature_edit_distance(source_tokens, target_tokens, merged_feats)

        if d <= max_distance:
            candidates.append((ptd.words[idx], ptd.ipas[idx], d))

    candidates.sort(key=lambda t: t[2])
    return candidates[:top_n]
