from typing import Any

import jiwer
import torch
from bitarray import bitarray

from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.clean_phones import clean_phones
from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator
from phone_similarity.language import de, en_gb
from phone_similarity.phones_product import phones_product
from phone_similarity.raw_phone_features import RAW_FEATURES

WORDS_A = ["euthanasia"]
WORDS_B = ["Youth", "in", "asia"]
BITARRAY_SPECIFICATION = {}

for language, module in {"eng-us": en_gb, "ger": de}.items():
    vowels = module.VOWELS_SET
    feats_per_phoneme = module.PHONEME_FEATURES
    feats = module.FEATURES

    BITARRAY_SPECIFICATION.update(
        {
            language: BitArraySpecification(
                vowels=vowels,
                consonants=set(
                    filter(
                        lambda phone: phone not in vowels,
                        feats_per_phoneme,
                    )
                ),
                features_per_phoneme=feats_per_phoneme,
                features=feats,
            )
        }
    )


def test_charsiu_g2p_many_hypotheses():
    language = "eng-us"
    g2p = CharsiuGraphemeToPhonemeGenerator(language)

    _syllables = []
    generation_args = dict(  # pylint: disable=use-dict-literal
        num_beams=5,
        num_return_sequences=5,
        min_p=0.5,
        max_length=50,
        do_sample=True,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
    )

    phones_for_words, _ = g2p.generate(words=tuple(WORDS_A), **generation_args)

    assert len(phones_for_words[0]) == 5
    assert BITARRAY_SPECIFICATION[language].ipa_to_bitarray(
        phones_for_words[0][0], 6
    ) == bitarray(
        "0000100000010101000001000100000010000100011101000101100"
        "0000000010000010000100000010001100010001100000000000000"
        "0000000000000000000000000000000000000000000000000000000"
        "0000000000000000000000000000000000000000000000000000000"
        "00000000"
    )

    if isinstance(phones_for_words[0], str):
        phones_for_words = [clean_phones(p) for p in phones_for_words]

    else:
        phones_for_words = list(
            set([clean_phones(p) for word in phones_for_words for p in word])
        )

    phones_for_words_product = phones_product(
        phones_for_words, tokenizer=BITARRAY_SPECIFICATION[language].ipa_tokenizer
    )

    for phones in phones_for_words_product:
        assert isinstance(phones, str)
        _syllables.append(BITARRAY_SPECIFICATION[language].ipa_to_syllable(phones))

    if torch.cuda.is_available():
        assert len(_syllables) == 12
    else:
        assert len(_syllables) in {12, 18}


def test_charsiu_g2p_one_hypothesis():
    language = "eng-us"
    g2p = CharsiuGraphemeToPhonemeGenerator(language)

    _syllables = []
    generation_args = dict(  # pylint: disable=use-dict-literal
        num_beams=1,
        max_length=50,
    )

    phones_for_words, _ = g2p.generate(words=tuple(WORDS_A), **generation_args)

    if isinstance(phones_for_words[0], str):
        phones_for_words = [clean_phones(p) for p in phones_for_words]
        for phones in phones_for_words:
            assert isinstance(phones, str)
            _syllables.append(BITARRAY_SPECIFICATION[language].ipa_to_syllable(phones))

    assert _syllables[0] == [
        {"nucleus": bitarray("1000100011"), "onset": bitarray("00100000010001")},
        {"nucleus": bitarray("0100000100"), "onset": bitarray("01011000000000")},
        {"nucleus": bitarray("0001110100"), "onset": bitarray("01000000100001")},
        {"nucleus": bitarray("0100000100"), "onset": bitarray("00001000000101")},
    ]


def test_charsiu_g2p_similarity():
    language = "eng-us"
    g2p = CharsiuGraphemeToPhonemeGenerator(language)

    _syllables = []
    generation_args = dict(  # pylint: disable=use-dict-literal
        num_beams=5,
        num_return_sequences=5,
        min_p=0.5,
        max_length=50,
        do_sample=True,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
    )

    phones_for_words, _ = g2p.generate(words=tuple(WORDS_A), **generation_args)

    assert len(phones_for_words[0]) == 5
    assert BITARRAY_SPECIFICATION["eng-us"].ipa_to_bitarray(
        phones_for_words[0][0], 6
    ) == bitarray(
        "0000100000010101000001000100000010000100011101000101100"
        "0000000010000010000100000010001100010001100000000000000"
        "0000000000000000000000000000000000000000000000000000000"
        "0000000000000000000000000000000000000000000000000000000"
        "00000000"
    )

    if isinstance(phones_for_words[0], str):
        phones_for_words = [clean_phones(p) for p in phones_for_words]

    else:
        phones_for_words = list(
            set([clean_phones(p) for word in phones_for_words for p in word])
        )

    phones_for_words_product = phones_product(
        phones_for_words, tokenizer=BITARRAY_SPECIFICATION["eng-us"].ipa_tokenizer
    )

    for phones in phones_for_words_product:
        assert isinstance(phones, str)
        _syllables.append(BITARRAY_SPECIFICATION["eng-us"].ipa_to_syllable(phones))

    assert len(_syllables) == 12


def validate_german_ipa(
    word: str, bitarray_specification: Any, g2p: Any, reference_ipa
):
    generation_args = dict(  # pylint: disable=use-dict-literal
        num_beams=50,
        num_return_sequences=50,
        min_p=0.5,
        max_length=50,
        do_sample=True,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
    )

    phones_for_words, _ = g2p.generate(words=tuple([word]), **generation_args)

    if isinstance(phones_for_words[0], str):
        phones_for_words = [p for p in phones_for_words]

    else:
        phones_for_words = list([p for word in phones_for_words for p in word])

    print(phones_for_words)

    generations = {}
    for generated_ipa in phones_for_words:
        ref_tokens = bitarray_specification.ipa_tokenizer(clean_phones(reference_ipa))
        gen_tokens = bitarray_specification.ipa_tokenizer(clean_phones(generated_ipa))
        wer_output = jiwer.wer(" ".join(ref_tokens), " ".join(gen_tokens))

        if wer_output == 0:
            # Exact match to reference (given tokens)
            return
        generations.update(
            {
                (
                    tuple(ref_tokens),
                    tuple(gen_tokens),
                ): wer_output
            }
        )

    raise ValueError()


def test_charsiu_g2p_many_hypotheses_german():
    language = "ger"
    g2p = CharsiuGraphemeToPhonemeGenerator(language)

    word = "betriebshinweis"
    reference_ipa = "bəˈtʁiːpsˌhɪnvaɪ̯s"

    validate_german_ipa(word, BITARRAY_SPECIFICATION["ger"], g2p, reference_ipa)

    word = "programmoptimierung"
    reference_ipa = "pʁoˈɡʁamʔɔptiˌmiːʁʊŋ"

    validate_german_ipa(word, BITARRAY_SPECIFICATION["ger"], g2p, reference_ipa)
