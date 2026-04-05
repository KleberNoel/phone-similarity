from collections import defaultdict
from typing import Dict, Tuple

from bitarray import bitarray

from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator


class Distance:
    def __init__(self, a: BitArraySpecification, b: BitArraySpecification):
        self.a = a
        self.g2p_a = CharsiuGraphemeToPhonemeGenerator(language="eng-us")

    def pdict_bitarray(self) -> Dict[str, Tuple[bitarray]]:
        newpdict = defaultdict(list)
        assert self.g2p_a.pdict is not None
        for w, p in self.g2p_a.pdict.items():
            arr = bitarray()
            for phones in p.split(","):
                phones = phones.replace("ˈ", "").replace(
                    "", ""
                )  # Remove stress markers
                arr = self.a.ipa_to_syllable(phones)

            newpdict[w].append(arr)
            (
                print("Encoded", len(newpdict))
                if len(newpdict.keys()) % 10000 == 0
                else None
            )
            # Overwrite as newpdict entries as tuples
        return {k: tuple(v) for k, v in newpdict.items()}
