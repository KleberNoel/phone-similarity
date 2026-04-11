"""
Lazy loader for compressed language phoneme data.

Reads ``_data.json`` once on first access and materialises lightweight
namespace objects that expose the same three public attributes that the
old per-language ``.py`` modules provided:

* ``VOWELS_SET``  -- ``set[str]``
* ``PHONEME_FEATURES`` -- ``dict[str, dict[str, bool | str]]``
* ``FEATURES`` -- ``dict[str, set[str]]``  (consonant / vowel / modifier
  column sets used by :class:`BitArraySpecification`)

Feature-set reduction
---------------------
By default the loader uses the **stored** column sets from the JSON
(identical to the original hand-curated ``.py`` files).  Pass
``reduce_features=True`` to :func:`get_language` to instead *derive*
columns from the actual phoneme data, which drops columns that are
present in the stored schema but never activated by any phoneme
(``features_to_bitarray`` always yields 0 for those bits).

This is useful when comparing across languages whose feature inventories
have little overlap -- the reduced set eliminates dead bits and produces
tighter bitarrays.
"""

from __future__ import annotations

import json
import os
from typing import Any

# ---------------------------------------------------------------------------
# Singleton raw-data cache (loaded lazily from _data.json)
# ---------------------------------------------------------------------------
_DATA: dict[str, Any] | None = None
_DATA_PATH = os.path.join(os.path.dirname(__file__), "_data.json")


def _ensure_loaded() -> dict[str, Any]:
    global _DATA
    if _DATA is None:
        with open(_DATA_PATH, encoding="utf-8") as fh:
            _DATA = json.load(fh)
    return _DATA


def available_languages() -> frozenset[str]:
    """Return all language keys without materialising any namespace."""
    return frozenset(_ensure_loaded())


# ---------------------------------------------------------------------------
# Namespace object returned by get_language()
# ---------------------------------------------------------------------------
class _LanguageNamespace:
    """Lightweight stand-in for the old per-language module.

    Exposes ``.VOWELS_SET``, ``.PHONEME_FEATURES``, and ``.FEATURES``
    as read-only properties.
    """

    __slots__ = ("_features", "_key", "_phoneme_features", "_vowels_set")

    def __init__(
        self,
        key: str,
        vowels_set: set[str],
        phoneme_features: dict[str, dict[str, bool | str]],
        features: dict[str, set[str]],
    ) -> None:
        self._key = key
        self._vowels_set = vowels_set
        self._phoneme_features = phoneme_features
        self._features = features

    # --- public attributes (match old module interface) --------------------

    @property
    def VOWELS_SET(self) -> set[str]:
        return self._vowels_set

    @property
    def PHONEME_FEATURES(self) -> dict[str, dict[str, bool | str]]:
        return self._phoneme_features

    @property
    def FEATURES(self) -> dict[str, set[str]]:
        return self._features

    def __repr__(self) -> str:
        n = len(self._phoneme_features)
        return f"<Language {self._key!r}: {n} phonemes>"


# ---------------------------------------------------------------------------
# Feature derivation (reduced mode)
# ---------------------------------------------------------------------------
def derive_features(
    vowels_set: set[str],
    phoneme_features: dict[str, dict[str, bool | str]],
) -> dict[str, set[str]]:
    """Derive column sets from *actual* phoneme feature data.

    This produces a **reduced** feature set: only columns that appear
    as keys in at least one phoneme's feature dict are included.
    Columns present in the stored schema but unused by any phoneme
    are dropped.

    Classification rules
    --------------------
    * A phoneme is a **vowel** if it appears in *vowels_set*.
    * A phoneme is a **modifier** if its feature dict contains
      ``"marker": True`` or ``"modifier": True`` and it is not a vowel.
    * Everything else is a **consonant**.

    For each category the column set is the union of all feature-dict
    keys across the phonemes in that category.
    """
    vowel_cols: set[str] = set()
    consonant_cols: set[str] = set()
    modifier_cols: set[str] = set()

    for phoneme, feats in phoneme_features.items():
        keys = set(feats)
        if phoneme in vowels_set:
            vowel_cols |= keys
        elif feats.get("marker") or feats.get("modifier"):
            modifier_cols |= keys
        else:
            consonant_cols |= keys

    result: dict[str, set[str]] = {
        "consonant": consonant_cols,
        "vowel": vowel_cols,
    }
    if modifier_cols:
        result["modifier"] = modifier_cols
    return result


# ---------------------------------------------------------------------------
# Per-language cache (keyed by (lang, reduce_features))
# ---------------------------------------------------------------------------
_NS_CACHE: dict[tuple[str, bool], _LanguageNamespace] = {}


def get_language(
    key: str,
    *,
    reduce_features: bool = False,
) -> _LanguageNamespace:
    """Return a cached :class:`_LanguageNamespace` for *key*.

    Parameters
    ----------
    key:
        Language identifier (e.g. ``"eng_us"``, ``"fra"``).
    reduce_features:
        If ``True``, derive ``FEATURES`` from the phoneme data (smaller,
        no dead columns).  If ``False`` (default), use the stored column
        sets for backward-compatible bit-widths.
    """
    cache_key = (key, reduce_features)
    if cache_key in _NS_CACHE:
        return _NS_CACHE[cache_key]

    data = _ensure_loaded()
    if key not in data:
        raise KeyError(key)

    entry = data[key]
    vowels_set = set(entry["vowels"])
    phoneme_features: dict[str, dict[str, bool | str]] = entry["phonemes"]

    if reduce_features:
        features = derive_features(vowels_set, phoneme_features)
    else:
        # Reconstruct stored column sets
        features: dict[str, set[str]] = {
            "consonant": set(entry["consonant_columns"]),
            "vowel": set(entry["vowel_columns"]),
        }
        if "modifier_columns" in entry:
            features["modifier"] = set(entry["modifier_columns"])

    ns = _LanguageNamespace(key, vowels_set, phoneme_features, features)
    _NS_CACHE[cache_key] = ns
    return ns


def clear_cache() -> None:
    """Drop all cached language namespaces (useful for testing)."""
    _NS_CACHE.clear()
