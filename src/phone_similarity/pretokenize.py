"""
Pre-tokenized dictionary storage and disk caching.

Provides :class:`PreTokenizedDictionary` for compact numpy-backed storage
of tokenized phonological dictionaries, plus :func:`cached_pretokenize_dictionary`
for transparent on-disk caching with automatic invalidation.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from phone_similarity._dispatch import HAS_CYTHON_TOKENIZER as _HAS_CYTHON_TOKENIZER
from phone_similarity._dispatch import cy_batch_ipa_tokenize as _cy_batch_ipa_tokenize
from phone_similarity.base_bit_array_specification import BaseBitArraySpecification

logger = logging.getLogger(__name__)


def pretokenize_dictionary(
    dictionary: dict[str, str],
    spec: BaseBitArraySpecification,
    min_tokens: int = 2,
) -> list[tuple[str, str, list[str]]]:
    """Tokenize a G2P dictionary into (word, cleaned_ipa, tokens) tuples."""
    from phone_similarity.clean_phones import clean_phones as _clean

    words: list[str] = []
    ipas: list[str] = []
    for word, raw_ipa in dictionary.items():
        ipa = _clean(raw_ipa.split(",")[0].strip())
        if ipa:
            words.append(word)
            ipas.append(ipa)

    if _HAS_CYTHON_TOKENIZER and hasattr(spec, "_phones_sorted"):
        phone_set = frozenset(spec._phones_sorted)
        max_phoneme_size = max((len(p) for p in phone_set), default=1)
        all_tokens = _cy_batch_ipa_tokenize(ipas, phone_set, max_phoneme_size, min_tokens)
        result: list[tuple[str, str, list[str]]] = []
        for i in range(len(words)):
            toks = all_tokens[i]
            if toks:
                result.append((words[i], ipas[i], toks))
        return result

    result = []
    for i in range(len(words)):
        tokens = spec.ipa_tokenizer(ipas[i])
        if len(tokens) >= min_tokens:
            result.append((words[i], ipas[i], tokens))
    return result


class PreTokenizedDictionary:
    """Compact storage for a pre-tokenized phonological dictionary.

    Stores phoneme tokens as ``int16`` indices into a phoneme inventory,
    backed by numpy arrays.  This format loads from disk in ~0.1 s (vs
    ~0.8 s for pickle of equivalent Python objects) because numpy arrays
    deserialise as raw memory blocks rather than millions of individual
    Python objects.

    Supports ``len()``, ``[]`` indexing, and ``for ... in`` iteration
    returning ``(word, ipa, tokens)`` tuples for backward compatibility
    with :func:`reverse_dictionary_lookup` and ``batch_dictionary_scan``.
    """

    __slots__ = (
        "_inv_map",
        "inventory",
        "ipas",
        "offsets",
        "token_indices",
        "words",
    )

    def __init__(
        self,
        words: list[str],
        ipas: list[str],
        inventory: list[str],
        token_indices: np.ndarray,
        offsets: np.ndarray,
    ):
        self.words = words
        self.ipas = ipas
        self.inventory = inventory
        self._inv_map: dict[str, int] = {p: i for i, p in enumerate(inventory)}
        self.token_indices = token_indices  # int16, flat concatenated
        self.offsets = offsets  # int32, len = n_entries + 1

    # -- sequence protocol (backward compat) ---------------------------

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, i: int) -> tuple[str, str, list[str]]:
        start, end = int(self.offsets[i]), int(self.offsets[i + 1])
        tokens = [self.inventory[j] for j in self.token_indices[start:end]]
        return (self.words[i], self.ipas[i], tokens)

    def __iter__(self):
        inv = self.inventory
        ti = self.token_indices
        off = self.offsets
        for i in range(len(self.words)):
            start, end = int(off[i]), int(off[i + 1])
            tokens = [inv[j] for j in ti[start:end]]
            yield (self.words[i], self.ipas[i], tokens)

    # -- construction --------------------------------------------------

    @classmethod
    def from_entries(
        cls,
        entries: list[tuple[str, str, list[str]]],
    ) -> PreTokenizedDictionary:
        """Build from a list of ``(word, ipa, tokens)`` tuples."""
        inventory = sorted({p for _, _, tokens in entries for p in tokens})
        inv_map = {p: i for i, p in enumerate(inventory)}
        words = [w for w, _, _ in entries]
        ipas = [ipa for _, ipa, _ in entries]
        flat: list[int] = []
        offsets = [0]
        for _, _, tokens in entries:
            for t in tokens:
                flat.append(inv_map[t])
            offsets.append(len(flat))
        return cls(
            words,
            ipas,
            inventory,
            np.array(flat, dtype=np.int16),
            np.array(offsets, dtype=np.int32),
        )

    # -- serialisation -------------------------------------------------

    def save(self, path: Path) -> None:
        """Save to a single pickle file with numpy byte buffers."""
        data = {
            "version": 2,
            "words": self.words,
            "ipas": self.ipas,
            "inventory": self.inventory,
            "ti": self.token_indices.tobytes(),
            "ti_shape": self.token_indices.shape,
            "to": self.offsets.tobytes(),
            "to_shape": self.offsets.shape,
        }
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> PreTokenizedDictionary:
        """Load from disk (~0.1 s for 278 K entries)."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != 2:
            raise ValueError(f"Unsupported cache version in {path}")
        ti = np.frombuffer(data["ti"], dtype=np.int16).reshape(data["ti_shape"])
        to = np.frombuffer(data["to"], dtype=np.int32).reshape(data["to_shape"])
        return cls(data["words"], data["ipas"], data["inventory"], ti, to)

    # -- utility -------------------------------------------------------

    def to_cleaned_dict(self) -> dict[str, str]:
        """Derive a ``{word: cleaned_ipa}`` dict without loading the raw G2P."""
        return dict(zip(self.words, self.ipas, strict=False))


# Disk cache helpers

_DEFAULT_CACHE_DIR = Path(os.path.expanduser("~/.cache/phone_similarity"))

_G2P_CACHE_DIR = Path(os.path.expanduser("~/.cache/phono-sim"))


def _pretokenize_cache_fingerprint(
    spec: BaseBitArraySpecification,
    min_tokens: int,
    g2p_stat: os.stat_result | None,
) -> str:
    """Compute cache fingerprint from phoneme inventory + G2P file stat.

    Uses the G2P pickle file's mtime and size as a proxy for dictionary
    content, so we never need to *load* the raw dict to check
    invalidation.
    """
    inventory = tuple(sorted(spec._phones_sorted))
    stat_key = (g2p_stat.st_mtime_ns, g2p_stat.st_size) if g2p_stat else (0, 0)
    payload = repr((inventory, min_tokens, stat_key)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def cached_pretokenize_dictionary(
    dict_or_factory,
    spec: BaseBitArraySpecification,
    lang: str = "unknown",
    min_tokens: int = 2,
    *,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> PreTokenizedDictionary:
    """Pre-tokenize a G2P dictionary with transparent on-disk caching."""
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR

    if use_cache:
        # Stat the G2P raw-dict pickle for invalidation (no loading)
        py_ver = f"py{sys.version_info.major}.{sys.version_info.minor}"
        g2p_pkl = _G2P_CACHE_DIR / f"{lang}_{py_ver}.pkl"
        g2p_stat = g2p_pkl.stat() if g2p_pkl.exists() else None

        fingerprint = _pretokenize_cache_fingerprint(spec, min_tokens, g2p_stat)
        cache_file = cache_dir / f"pretok_{lang}_{fingerprint}.v2.pkl"

        if cache_file.exists():
            t0 = time.time()
            ptd = PreTokenizedDictionary.load(cache_file)
            elapsed = time.time() - t0
            logger.info(
                "Loaded cached pre-tokenized %s dictionary (%d entries) in %.3fs from %s",
                lang,
                len(ptd),
                elapsed,
                cache_file,
            )
            return ptd

    # Cache miss -- materialise the dictionary
    if callable(dict_or_factory):
        dictionary = dict_or_factory()
    else:
        dictionary = dict_or_factory

    t0 = time.time()
    entries = pretokenize_dictionary(dictionary, spec, min_tokens=min_tokens)
    ptd = PreTokenizedDictionary.from_entries(entries)
    elapsed = time.time() - t0
    logger.info(
        "Pre-tokenized %s dictionary: %d entries in %.2fs",
        lang,
        len(ptd),
        elapsed,
    )

    if use_cache:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            ptd.save(cache_file)
            logger.info("Cached pre-tokenized %s dictionary to %s", lang, cache_file)
        except OSError as exc:
            logger.warning("Failed to write pretokenize cache: %s", exc)

    return ptd
