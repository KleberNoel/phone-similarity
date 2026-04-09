"""
Concrete bitarray specification for phonological feature encoding.

Extends :class:`BaseBitArraySpecification` to convert IPA strings into
syllable-structured bitarrays via naive syllabification (onset-nucleus-coda
decomposition) and feature-to-bit mapping.
"""

import logging

from bitarray import bitarray

from phone_similarity.base_bit_array_specification import BaseBitArraySpecification


class BitArraySpecification(BaseBitArraySpecification):
    """Syllable-structured bitarray encoder for IPA strings.

    Converts IPA strings into fixed-width bitarrays by decomposing them
    into onset-nucleus-coda syllable triples and encoding each component's
    phonological features as binary vectors.
    """

    def __init__(self, *args, features: dict[str, set[str]], **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )
        self._features: dict[str, tuple[str]] = self.sort_features(features=features)

    @property
    def max_syllable_length(self) -> int:
        """
        max_syllable_length
        """
        if hasattr(self, "empty_vector"):
            return len(self._features["consonant"]) * 2 + len(self._features["vowel"])

        raise ValueError("Max Syllable Length cannot be calculated using BitArrayGenerator")

    @property
    def features(self):
        """
        features
        """
        return self._features

    def ipa_to_bitarray(self, ipa: str, max_syllables: int) -> "bitarray":
        """Converts an IPA string into a bitarray representation.

        This method processes an IPA string, breaks it down into syllables,
        and then converts each syllable into a bitarray based on its onset, nucleus,
        and coda. The resulting bitarrays are concatenated and padded to a fixed length.

        Parameters
        ----------
        ipa : str
            The Input IPA string to convert.
        max_syllables : int
            The maximum number of syllables to process from the end of the string.

        Returns
        -------
        bitarray
            The final bitarray representing the IPA string.

        Raises
        ------
        Exception
            Propagates exceptions from padding, indicating a potential issue in length calculation.

        """
        arr = bitarray()
        for idx, _arr_parts in enumerate(self.ipa_to_syllable(ipa)[::-1], start=1):
            for key in sorted(_arr_parts.keys(), reverse=True):
                arr += _arr_parts[key]
            if idx == max_syllables:
                break

        try:
            arr += bitarray([0] * (self.max_syllable_length * max_syllables - len(arr)))
        except ValueError:
            logging.error("Error filling remainder of array %s", arr)
            raise

        return arr

    def ipa_to_syllable(self, ipa_str: str) -> list[dict[str, "bitarray"]]:
        """Decomposes an IPA string into a list of syllables.

        This method follows a naive approach to syllabification:
        Scan for:
        - Vowels to identify nucleus of a syllable, and group features in a bitarray (0s and 1s).
        - Consonants preceding a vowel are grouped in a bitarray (0s and 1s) as the onset.
        - Consonants following a vowel are grouped in a bitarray (0s and 1s) as the coda.

        This implementation treats consonant clusters as single units and may merge
        a previous syllable's coda with the current syllable's onset if they are adjacent.

        Steps:
        1. Gather onset (initial consonants until a vowel)
        2. Check if current token is final. If so, merge onset with previous coda.
           else if prev, merge prev coda with current onset and delete prev coda
        3. tokens[i] is vowel / nucleus; extract until vowels have been exhausted
        4. Gather coda (all consonants after the vowel until next vowel or end)
        5. Combine onset + coda and store result for syllable

        Parameters
        ----------
        ipa_str : str
            The IPA string to be syllabified.

        Returns
        -------
        List[Dict[str, "bitarray"]]
            A list of dictionaries, where each dictionary represents a syllable
            and contains bitarray representations for its 'onset', 'nucleus', and 'coda'.

        """
        tokens = self.ipa_tokenizer(ipa_str)
        n: int = len(tokens)
        i: int = 0
        results: list[dict] = []

        while i < n:
            # 1. Gather onset (initial consonants until a vowel)
            feature_type = "consonant"
            onset: bitarray = self.empty_vector(feature_type)
            while i < n and tokens[i] not in self._vowels:
                onset = self.update_array_segment(tokens[i], onset, feature_type="consonant")
                i += 1

            # 2. Check if current token is final. If so, merge onset with previous coda.
            # Else if prev, merge prev coda with current onset and delete prev coda
            if i == n:
                if results:
                    results[-1] = {
                        "onset": results[-1]["onset"],
                        "nucleus": results[-1]["nucleus"],
                        "coda": results[-1]["coda"] | onset,
                    }
                break

            if len(results) > 0 and results[-1].get("coda") is not None:
                # Compress previous coda into current onset
                onset = results[-1]["coda"] | onset
                if tokens[i] in self._vowels:
                    del results[-1]["coda"]

            # 3. tokens[i] is vowel / nucleus.
            feature_type = "vowel"
            nucleus: bitarray = self.empty_vector(feature_type)
            _vowel_start: int = i
            if tokens[i] not in self._vowels:
                raise ValueError(f"Expected vowel at position {i} but got {tokens[i]!r}")
            while i < n and tokens[i] in self._vowels:
                if i != _vowel_start:
                    logging.warning(
                        (
                            "Found multiple vowels in the nucleus (tokens: %s)."
                            " Perhaps better parsing can be applied or the vowel"
                            " set can be expanded into dipthongs."
                        ),
                        " ".join(tokens[_vowel_start:i]),
                    )
                nucleus = self.update_array_segment(tokens[i], nucleus, feature_type)
                i += 1  # Consume token

            # 4. Gather coda (all consonants after the vowel until next vowel or end)
            feature_type = "consonant"
            coda: bitarray = self.empty_vector(feature_type)
            while i < n and tokens[i] not in self._vowels:
                coda = self.update_array_segment(tokens[i], coda, feature_type="consonant")
                i += 1

            # 5. Combine onset + coda and store the result for this syllable
            result = {"nucleus": nucleus}
            if sum(coda) > 0:
                result.update(
                    {
                        "coda": coda,
                    }
                )

            if sum(onset) > 0:
                result.update(
                    {
                        "onset": onset,
                    }
                )

            results.append(result)

            if i == n:
                break

            logging.debug("Incremental result: %s", results)

        return results

    def generate(self, text: str) -> "bitarray":
        """Generates a bitarray for a given text.

        This is the main entry point for converting a text string into its bitarray representation.
        It orchestrates the process of syllabification and bitarray conversion based on the
        configured parameters for syllable count and length.

        Parameters
        ----------
        text : str
            The input string to be converted, expected to be in IPA format.

        Returns
        -------
        bitarray
            The resulting bitarray representation of the input text.

        """
        return self.ipa_to_bitarray(
            ipa=text,
            max_syllables=self._max_syllables_per_text_chunk,
        )

    def update_array_segment(
        self, phoneme: str, current_segment: "bitarray", feature_type: str
    ) -> "bitarray":
        """Updates a bitarray segment by adding the features of a new phoneme.

        This method takes an existing bitarray segment and performs an element-wise
        addition with the bitarray of a new phoneme. This is used to cumulate
        features within a syllable component (e.g., onset, coda).

        Parameters
        ----------
        phoneme : str
            The phoneme to add to the segment.
        current_segment : bitarray
            The current bitarray segment to be updated.
        feature_type : str
            The type of features to consider ('consonant' or 'vowel'), which
            determines the columns for bitarray conversion.

        Returns
        -------
        bitarray
            The updated bitarray segment.

        """
        features = self.get_phoneme_features(phoneme=phoneme)
        return current_segment | self.features_to_bitarray(
            feature_dict=features,
            columns=self._features[feature_type],
        )

    def empty_vector(self, feature_type: str) -> "bitarray":
        """Creates a zeroed bitarray for a given feature type.

        This method generates a bitarray of a specific length corresponding to the
        number of features for a given type ('consonant' or 'vowel').  A fresh
        bitarray is returned on every call so callers can safely mutate it.

        Parameters
        ----------
        feature_type : str
            The type of features ('consonant' or 'vowel') for which to create
            the empty vector.

        Returns
        -------
        bitarray
            A bitarray of the correct length, initialized with all zeros.

        """
        return bitarray(len(self._features[feature_type]))

    def fold(self, arr: "bitarray"):
        """
        Fold bitarray syllables into the 'rich-format' space of a single syllable
        """
        new_arr = bitarray(self.max_syllable_length)
        for syllable_bit_idx in range(
            0,
            self._max_syllables_per_text_chunk * self.max_syllable_length,
            self.max_syllable_length,
        ):
            new_arr ^= arr[syllable_bit_idx : syllable_bit_idx + self.max_syllable_length]

        return new_arr
