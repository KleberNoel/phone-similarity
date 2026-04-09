from phone_similarity.base_bit_array_specification import BaseBitArraySpecification


class ModelFactory:
    """
    Factory for creating phonetic similarity models.
    """

    def __init__(self):
        self._models = {}

    def register_model(self, language: str, model_class: type[BaseBitArraySpecification]):
        """
        Register a model class for a given language.
        """
        self._models[language] = model_class

    def get_model(self, language: str, *args, **kwargs) -> BaseBitArraySpecification:
        """
        Get an instance of a model for a given language.
        """
        model_class = self._models.get(language)
        if not model_class:
            raise ValueError(f"Model for language '{language}' not found.")
        return model_class(*args, **kwargs)
