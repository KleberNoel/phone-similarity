"""
Language files have the structure of BCP-47 language tags
These language tags predominantly combine ISO-639-1, ISO-639-2 and ISO-639-3
"""

import importlib
import os

LANGUAGES = {}
LANGUAGE_FILES = [
    f[:-3]
    for f in os.listdir(os.path.dirname(__file__))
    if f.endswith(".py") and f != "__init__.py"
]

for lang_code in LANGUAGE_FILES:
    try:
        LANGUAGES[lang_code] = importlib.import_module(
            f"phone_similarity.language.{lang_code}"
        )
    except (AttributeError, ImportError, ModuleNotFoundError):
        # Handle cases where the module is empty or has import errors
        pass
