import json
import unittest
from pathlib import Path


class TestPhoneticConsistency(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
