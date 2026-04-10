"""
Universal phonological feature system backed by Panphon.

Provides a language-independent feature representation for IPA phonemes,
enabling consistent cross-language phonological distance computation.
Feature vectors are retrieved from the `panphon <https://github.com/dmort27/panphon>`_
library at runtime.

Feature values are ternary: ``+1`` (present), ``-1`` (absent), ``0``
(unspecified/not applicable).  The 24 features are:

    syl, son, cons, cont, delrel, lat, nas, strid, voi, sg, cg,
    ant, cor, distr, lab, hi, lo, back, round, velaric,
    tense, long, hitone, hireg

Usage::

    from phone_similarity.universal_features import UniversalFeatureEncoder

    enc = UniversalFeatureEncoder()
    enc.encode("p")          # (−1, −1, 1, −1, …)  24-int tuple
    enc.feature_dict("p")    # {"syl": -1, "son": -1, "cons": 1, …}

    # Cross-language feature merging (replaces naive {**a, **b})
    merged = enc.merge_inventories(eng_features, fra_features)
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from typing import Union

import panphon

# ---------------------------------------------------------------------------
# Singleton FeatureTable (loaded once, reused everywhere)
# ---------------------------------------------------------------------------
_FT = panphon.FeatureTable()

# ---------------------------------------------------------------------------
# 24-feature names in canonical Panphon order
# ---------------------------------------------------------------------------
PANPHON_FEATURE_NAMES: tuple[str, ...] = tuple(_FT.names)

_NUM_FEATURES = len(PANPHON_FEATURE_NAMES)
_ZERO_VECTOR: tuple[int, ...] = (0,) * _NUM_FEATURES


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class UniversalFeatureEncoder:
    """Encode IPA phonemes as 24-dimensional ternary feature vectors.

    All methods are **static** so callers can use them without instantiation.
    A module-level Panphon ``FeatureTable`` is shared across all calls.

    Resolution chain for unknown segments:

    1. Panphon direct lookup (exact IPA string)
    2. NFD normalisation
    3. Strip combining diacritics → base character(s)
    4. Zero vector ``(0, 0, …, 0)``
    """

    @staticmethod
    @lru_cache(maxsize=2048)
    def encode(phoneme: str) -> tuple[int, ...]:
        """Return the 24-feature ternary vector for *phoneme*.

        Parameters
        ----------
        phoneme : str
            A single IPA segment (e.g. ``"p"``, ``"aɪ"``, ``"tʃ"``).

        Returns
        -------
        tuple[int, ...]
            24-element tuple of ``+1``, ``-1``, or ``0``.
        """
        vec = _resolve(phoneme)
        if vec is not None:
            return vec

        # NFD normalisation
        nfd = unicodedata.normalize("NFD", phoneme)
        vec = _resolve(nfd)
        if vec is not None:
            return vec

        # Strip combining diacritics → base character(s)
        base = "".join(ch for ch in nfd if not unicodedata.combining(ch))
        if base and base != phoneme:
            vec = _resolve(base)
            if vec is not None:
                return vec

        return _ZERO_VECTOR

    @staticmethod
    def feature_dict(phoneme: str) -> dict[str, int]:
        """Return a ``{feature_name: value}`` dict for *phoneme*.

        Compatible with :func:`phoneme_feature_distance` in
        :mod:`phone_similarity.primitives`.
        """
        vec = UniversalFeatureEncoder.encode(phoneme)
        return dict(zip(PANPHON_FEATURE_NAMES, vec))

    # ----- bulk conversion ------------------------------------------------

    @staticmethod
    def convert_inventory(
        phoneme_features: dict[str, dict[str, Union[bool, str]]],
    ) -> dict[str, dict[str, int]]:
        """Re-encode a per-language ``PHONEME_FEATURES`` dict into universal features.

        Parameters
        ----------
        phoneme_features : dict
            Original language-specific ``{phoneme: {feature: value}}`` dict.

        Returns
        -------
        dict
            ``{phoneme: {panphon_feature: int}}`` using the 24-feature set.
        """
        return {ph: UniversalFeatureEncoder.feature_dict(ph) for ph in phoneme_features}

    # ----- distance -------------------------------------------------------

    @staticmethod
    def universal_phoneme_distance(ph_a: str, ph_b: str) -> float:
        """Ternary Hamming distance between two phonemes, normalised to [0, 1].

        ``0`` features (unspecified) in either phoneme are excluded from
        the comparison.  If no comparable features remain, returns 0.0.
        """
        va = UniversalFeatureEncoder.encode(ph_a)
        vb = UniversalFeatureEncoder.encode(ph_b)
        comparable = 0
        mismatches = 0
        for a_val, b_val in zip(va, vb):
            if a_val == 0 or b_val == 0:
                continue
            comparable += 1
            if a_val != b_val:
                mismatches += 1
        if comparable == 0:
            return 0.0
        return mismatches / comparable

    # ----- cross-language merging -----------------------------------------

    @staticmethod
    def merge_inventories(
        *inventories: dict[str, dict[str, Union[bool, str]]],
    ) -> dict[str, dict[str, int]]:
        """Merge multiple language inventories into a single universal feature dict.

        Unlike the naive ``{**a, **b}`` approach, this maps *every*
        phoneme through the Panphon universal feature set so that shared
        phonemes (e.g. ``/e/``) receive a single consistent
        representation regardless of which language defined them.

        Parameters
        ----------
        *inventories : dict
            One or more ``PHONEME_FEATURES`` dicts from language modules.

        Returns
        -------
        dict
            ``{phoneme: {panphon_feature: int}}`` with universal features.
        """
        merged: dict[str, dict[str, int]] = {}
        for inv in inventories:
            for ph in inv:
                if ph not in merged:
                    merged[ph] = UniversalFeatureEncoder.feature_dict(ph)
        return merged


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _resolve(phoneme: str) -> tuple[int, ...] | None:
    """Try to look up *phoneme* in Panphon's FeatureTable.

    Returns a 24-int tuple on success, ``None`` on failure.

    Note: ``panphon.FeatureTable.fts()`` returns a ``Segment`` on success
    and an empty ``dict`` ``{}`` on failure — **not** ``None``.
    """
    seg = _FT.fts(phoneme)
    if not isinstance(seg, panphon.segment.Segment):
        return None
    return tuple(seg[name] for name in PANPHON_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Module-level convenience aliases
# ---------------------------------------------------------------------------

encode_phoneme = UniversalFeatureEncoder.encode
phoneme_feature_dict = UniversalFeatureEncoder.feature_dict
universal_phoneme_distance = UniversalFeatureEncoder.universal_phoneme_distance
merge_inventories = UniversalFeatureEncoder.merge_inventories
