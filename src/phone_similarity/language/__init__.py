"""
Language files have the structure of BCP-47 language tags
These language tags predominantly combine ISO-639-1, ISO-639-2 and ISO-639-3

Language modules are loaded *lazily* on first access to avoid importing all
70+ modules at package import time (~3% of startup CPU).
"""

import importlib
import os

LANGUAGE_FILES = [
    f[:-3]
    for f in os.listdir(os.path.dirname(__file__))
    if f.endswith(".py") and f != "__init__.py"
]

_LANGUAGE_FILE_SET = frozenset(LANGUAGE_FILES)


class _LazyLanguageDict(dict):
    """Dict that imports language modules on first access.

    Supports ``LANGUAGES["eng_us"]``, ``LANGUAGES.get("eng_us")``,
    ``"eng_us" in LANGUAGES``, iteration, and ``len()`` — all without
    eagerly importing every module.
    """

    def __missing__(self, key):
        if key not in _LANGUAGE_FILE_SET:
            raise KeyError(key)
        try:
            mod = importlib.import_module(f"phone_similarity.language.{key}")
        except (AttributeError, ImportError, ModuleNotFoundError):
            raise KeyError(key) from None
        self[key] = mod
        return mod

    def __contains__(self, key):
        return key in _LANGUAGE_FILE_SET or dict.__contains__(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return _LANGUAGE_FILE_SET

    def __len__(self):
        return len(_LANGUAGE_FILE_SET)

    def __iter__(self):
        return iter(_LANGUAGE_FILE_SET)

    def items(self):
        for k in _LANGUAGE_FILE_SET:
            yield k, self[k]

    def values(self):
        for k in _LANGUAGE_FILE_SET:
            yield self[k]


LANGUAGES = _LazyLanguageDict()
