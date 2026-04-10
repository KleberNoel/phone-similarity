"""
Language module registry with lazy loading.

Language files have the structure of BCP-47 language tags.  These language
tags predominantly combine ISO-639-1, ISO-639-2, and ISO-639-3.

Language modules are loaded *lazily* on first access to avoid importing all
100+ modules at package import time.

Design Patterns
---------------
* **Registry** — ``LanguageRegistry`` auto-discovers language modules and
  provides ``get()``, ``contains()``, iteration, and ``build_spec()``
  (Builder pattern) for constructing a ``BitArraySpecification`` from a
  language key in a single call.
* **Lazy Loading** — Modules are imported only on first access via
  ``importlib.import_module``.
"""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phone_similarity.bit_array_specification import BitArraySpecification
    from phone_similarity.distance_class import Distance

# ---------------------------------------------------------------------------
# Auto-discover language modules
# ---------------------------------------------------------------------------
_LANG_DIR = os.path.dirname(__file__)
LANGUAGE_FILES = [
    f[:-3] for f in os.listdir(_LANG_DIR) if f.endswith(".py") and f != "__init__.py"
]
_LANGUAGE_FILE_SET = frozenset(LANGUAGE_FILES)


# ---------------------------------------------------------------------------
# Registry (Singleton + Lazy Loading + Builder)
# ---------------------------------------------------------------------------
class LanguageRegistry:
    """Registry of language modules with lazy import and spec building.

    Supports dict-like access::

        from phone_similarity.language import LANGUAGES

        lang = LANGUAGES["eng_us"]            # lazy import
        spec = LANGUAGES.build_spec("eng_us") # Builder pattern
        dist = LANGUAGES.build_distance("eng_us")  # full Distance object

    Parameters
    ----------
    None.  The registry auto-discovers modules from the ``language/``
    directory at construction time.
    """

    def __init__(self) -> None:
        self._cache: dict[str, object] = {}

    # -- dict-like access ---------------------------------------------------

    def __getitem__(self, key: str) -> object:
        if key in self._cache:
            return self._cache[key]
        if key not in _LANGUAGE_FILE_SET:
            raise KeyError(key)
        try:
            mod = importlib.import_module(f"phone_similarity.language.{key}")
        except (AttributeError, ImportError, ModuleNotFoundError):
            raise KeyError(key) from None
        self._cache[key] = mod
        return mod

    def __contains__(self, key: object) -> bool:
        return key in _LANGUAGE_FILE_SET or key in self._cache

    def get(self, key: str, default: object = None) -> object:
        """Return the language module for *key*, or *default*."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> frozenset[str]:
        """All available language keys (without importing them)."""
        return _LANGUAGE_FILE_SET

    def __len__(self) -> int:
        return len(_LANGUAGE_FILE_SET)

    def __iter__(self):
        return iter(_LANGUAGE_FILE_SET)

    def items(self):
        """Iterate ``(key, module)`` pairs — imports all modules."""
        for k in _LANGUAGE_FILE_SET:
            yield k, self[k]

    def values(self):
        """Iterate language modules — imports all modules."""
        for k in _LANGUAGE_FILE_SET:
            yield self[k]

    def __repr__(self) -> str:
        loaded = len(self._cache)
        total = len(_LANGUAGE_FILE_SET)
        return f"<LanguageRegistry {loaded}/{total} loaded>"

    # -- Builder pattern ----------------------------------------------------

    def build_spec(self, key: str) -> BitArraySpecification:
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
            Language module key (e.g. ``"eng_us"``, ``"fra"``).

        Returns
        -------
        BitArraySpecification
        """
        from phone_similarity.bit_array_specification import BitArraySpecification

        lang = self[key]
        return BitArraySpecification(
            vowels=lang.VOWELS_SET,
            consonants=set(lang.PHONEME_FEATURES) - lang.VOWELS_SET,
            features=lang.FEATURES,
            features_per_phoneme=lang.PHONEME_FEATURES,
        )

    def build_distance(self, key: str) -> Distance:
        """Build a ``Distance`` object for *key* in one call.

        Replaces::

            spec = LANGUAGES.build_spec("eng_us")
            dist = Distance(spec)

        With::

            dist = LANGUAGES.build_distance("eng_us")

        Parameters
        ----------
        key : str
            Language module key.

        Returns
        -------
        Distance
        """
        from phone_similarity.distance_class import Distance

        return Distance(self.build_spec(key))


# ---------------------------------------------------------------------------
# Singleton instance — the public API
# ---------------------------------------------------------------------------
LANGUAGES = LanguageRegistry()
