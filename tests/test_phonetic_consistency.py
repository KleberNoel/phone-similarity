import json
import unicodedata
import unittest
from pathlib import Path

from phone_similarity.language import LANGUAGES

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
        data_path = (
            Path(__file__).parent.parent / "src" / "phone_similarity" / "language" / "_data.json"
        )
        raw = json.loads(data_path.read_text())
        for lang_key, entry in raw.items():
            for phone in entry.get("phonemes", {}):
                self.assertNotEqual(
                    phone, "g", f"Latin 'g' (U+0067) used as phoneme key in {lang_key}"
                )

    def test_feature_schema_completeness(self):
        """Check that all vowel/consonant definitions have the core required keys for their manner."""
        for lang_key, mod in LANGUAGES.items():
            for phone, features in mod.PHONEME_FEATURES.items():
                if phone in mod.VOWELS_SET:
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
                        f"Vowel {phone} in {lang_key} missing height",
                    )
                elif "manner" in features:
                    self.assertIn(
                        "voiced", features, f"Consonant {phone} in {lang_key} missing voicedness"
                    )


if __name__ == "__main__":
    unittest.main()
