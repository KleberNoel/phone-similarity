import unittest

from bitarray import bitarray

from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.language.eng_uk import FEATURES
from phone_similarity.model_factory import ModelFactory

model_factory = ModelFactory()


class TestModelFactory(unittest.TestCase):
    def test_get_unregistered_model(self):
        with self.assertRaises(ValueError):
            model_factory.get_model(
                "fr",
                vowels=set(),
                consonants=set(),
                features_per_phoneme={},
                features=FEATURES,
            )

    def test_custom_registration(self):
        class FrenchBitArraySpecification(BitArraySpecification):
            def generate(self, text):
                return bitarray()

        model_factory.register_model("fr", FrenchBitArraySpecification)
        model = model_factory.get_model(
            "fr",
            vowels=set(),
            consonants=set(),
            features_per_phoneme={"a": {}},
            features=FEATURES,
        )
        self.assertIsInstance(model, FrenchBitArraySpecification)
