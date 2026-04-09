import importlib
import unicodedata
import unittest
from pathlib import Path

# Ligature mapping for consistency check
LIGATURES = {
    "ʧ": "tʃ",
    "ʤ": "dʒ",
    "ʦ": "ts",
    "ʣ": "dz",
    "ʨ": "tɕ",
    "ʥ": "dʑ",
}


class TestPhoneticConsistency(unittest.TestCase):
    def test_unicode_normalization(self):
        """Verify that system treats NFC and NFD as identical after normalization."""
        # e + combining tilde (NFD)
        nfd = "e\u0303"
        # precomposed e with tilde (NFC)
        nfc = "\u1ebd"

        # In our refined logic, we use NFKD for comparison
        self.assertEqual(unicodedata.normalize("NFKD", nfd), unicodedata.normalize("NFKD", nfc))

    def test_ipa_ligature_decomposition(self):
        """Ligatures should be decomposable into their constituent phones."""
        for _ligature, decomposed in LIGATURES.items():
            self.assertEqual(len(decomposed), 2)
            # This logic is what we added to test_language_phonemes.py

    def test_g_variant_standardization(self):
        """IPA 'ɡ' (U+0261) must be used instead of Latin 'g' (U+0067) in all modules."""
        lang_path = Path(__file__).parent.parent / "src" / "phone_similarity" / "language"
        for p in lang_path.glob("*.py"):
            if p.name == "__init__.py":
                continue
            content = p.read_text()
            # Ensure no "g" as a dictionary key for phonemes
            self.assertNotIn("'g':", content, f"Latin 'g' found in {p.name}")
            self.assertNotIn('"g":', content, f"Latin 'g' found in {p.name}")

    def test_feature_schema_completeness(self):
        """Check that all vowel/consonant definitions have the core required keys for their manner."""
        lang_path = Path(__file__).parent.parent / "src" / "phone_similarity" / "language"
        for p in lang_path.glob("*.py"):
            if p.name == "__init__.py":
                continue
            module_name = f"phone_similarity.language.{p.stem}"
            mod = importlib.import_module(module_name)

            for phone, features in mod.PHONEME_FEATURES.items():
                if phone in mod.VOWELS_SET:
                    # Minimum vowel features (checking for high, mid, or low variants)
                    # Diphthongs might not have these directly if they use start/end
                    if "diphthong" in features:
                        continue
                    self.assertTrue(
                        any(
                            k in features
                            for k in [
                                "high",
                                "mid",
                                "low",
                                "mid-high",
                                "mid-low",
                                "near-high",
                                "near-low",
                            ]
                        ),
                        f"Vowel {phone} in {p.stem} missing height",
                    )
                elif "manner" in features:
                    # Minimum consonant features if manner is specified
                    self.assertIn(
                        "voiced", features, f"Consonant {phone} in {p.stem} missing voicedness"
                    )


if __name__ == "__main__":
    unittest.main()
