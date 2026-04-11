"""
Language module registry with lazy loading.

Language data is stored in a single compressed JSON file
(``_data.json``) and loaded lazily via :mod:`._loader`.  Each
language entry contains phoneme feature tables originally curated
from `CharsiuG2P <https://github.com/lingjzhu/CharsiuG2P>`_
dictionaries and cross-referenced against Panphon and Wikipedia.

Design Patterns
---------------
* **Registry** -- ``LanguageRegistry`` exposes dict-like access,
  ``build_spec()`` (Builder), and ``build_distance()`` for
  constructing a ``BitArraySpecification`` or ``Distance`` object
  from a language key in a single call.
* **Lazy Loading** -- The JSON file is read only on first access
  and each language namespace is cached in ``_loader``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from phone_similarity.language._loader import (
    available_languages,
    get_language,
)

if TYPE_CHECKING:
    from phone_similarity.bit_array_specification import BitArraySpecification
    from phone_similarity.distance_class import Distance
    from phone_similarity.language._loader import _LanguageNamespace

# ---------------------------------------------------------------------------
# Available languages (from JSON keys, no data loaded yet)
# ---------------------------------------------------------------------------
LANGUAGE_FILES: list[str] = sorted(available_languages())
_LANGUAGE_KEY_SET: frozenset[str] = frozenset(LANGUAGE_FILES)


# ---------------------------------------------------------------------------
# Registry (Singleton + Lazy Loading + Builder)
# ---------------------------------------------------------------------------
class LanguageRegistry:
    """Registry of language data with lazy loading and spec building.

    Supports dict-like access::

        from phone_similarity.language import LANGUAGES

        lang = LANGUAGES["eng_us"]            # lazy load from JSON
        spec = LANGUAGES.build_spec("eng_us") # Builder pattern
        dist = LANGUAGES.build_distance("eng_us")  # full Distance object

    Feature reduction
    -----------------
    Pass ``reduce_features=True`` to :meth:`build_spec` or
    :meth:`build_distance` to derive the feature column sets from
    the actual phoneme data instead of using the stored (original)
    columns.  This drops columns that never produce a 1-bit in the
    bitarray encoding, yielding tighter representations for languages
    whose curated column sets include dead entries.
    """

    # -- dict-like access ---------------------------------------------------

    def __getitem__(self, key: str) -> _LanguageNamespace:
        if key not in _LANGUAGE_KEY_SET:
            raise KeyError(key)
        return get_language(key)

    def __contains__(self, key: object) -> bool:
        return key in _LANGUAGE_KEY_SET

    def get(self, key: str, default: object = None) -> _LanguageNamespace | object:
        """Return the language namespace for *key*, or *default*."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> frozenset[str]:
        """All available language keys (no data loaded)."""
        return _LANGUAGE_KEY_SET

    def __len__(self) -> int:
        return len(_LANGUAGE_KEY_SET)

    def __iter__(self):
        return iter(_LANGUAGE_KEY_SET)

    def items(self):
        """Iterate ``(key, namespace)`` pairs -- loads each language."""
        for k in _LANGUAGE_KEY_SET:
            yield k, self[k]

    def values(self):
        """Iterate language namespaces -- loads each language."""
        for k in _LANGUAGE_KEY_SET:
            yield self[k]

    def __repr__(self) -> str:
        total = len(_LANGUAGE_KEY_SET)
        return f"<LanguageRegistry {total} languages>"

    # -- Builder pattern ----------------------------------------------------

    @lru_cache(maxsize=256)  # noqa: B019
    def build_spec(
        self,
        key: str,
        *,
        reduce_features: bool = False,
    ) -> BitArraySpecification:
        """Build a ``BitArraySpecification`` for *key* in one call.

        Replaces the 5-line boilerplate::

            lang = LANGUAGES["eng_us"]
            spec = BitArraySpecification(
                vowels=lang.VOWELS_SET,
                consonants=set(lang.PHONEME_FEATURES) - lang.VOWELS_SET,
                features=lang.FEATURES,
                features_per_phoneme=lang.PHONEME_FEATURES,
            )

        With::

            spec = LANGUAGES.build_spec("eng_us")

        Parameters
        ----------
        key : str
            Language key (e.g. ``"eng_us"``, ``"fra"``).
        reduce_features : bool
            If ``True``, derive columns from phoneme data (smaller
            bitarrays, no dead columns).  Default ``False`` preserves
            the original curated column sets.

        Returns
        -------
        BitArraySpecification
        """
        from phone_similarity.bit_array_specification import BitArraySpecification

        lang = get_language(key, reduce_features=reduce_features)
        return BitArraySpecification(
            vowels=lang.VOWELS_SET,
            consonants=set(lang.PHONEME_FEATURES) - lang.VOWELS_SET,
            features=lang.FEATURES,
            features_per_phoneme=lang.PHONEME_FEATURES,
        )

    @lru_cache(maxsize=256)  # noqa: B019
    def build_distance(
        self,
        key: str,
        *,
        reduce_features: bool = False,
    ) -> Distance:
        """Build a ``Distance`` object for *key* in one call.

        Replaces::

            spec = LANGUAGES.build_spec("eng_us")
            dist = Distance(spec)

        With::

            dist = LANGUAGES.build_distance("eng_us")

        Parameters
        ----------
        key : str
            Language key.
        reduce_features : bool
            Passed through to :meth:`build_spec`.

        Returns
        -------
        Distance
        """
        from phone_similarity.distance_class import Distance

        return Distance(self.build_spec(key, reduce_features=reduce_features))


# ---------------------------------------------------------------------------
# Singleton instance -- the public API
# ---------------------------------------------------------------------------
LANGUAGES = LanguageRegistry()
