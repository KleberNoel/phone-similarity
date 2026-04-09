"""
Phoneme alignment product generation.

Computes pairwise word-error-rate alignments across IPA transcription
variants (via ``jiwer``) and expands all combinatorial alternatives into
flat IPA strings for downstream distance comparison.
"""

from collections import defaultdict
from itertools import permutations, product

from jiwer import process_words


def phones_product(phones_for_words_input, tokenizer):
    jiwer_alignments = []
    for a, b in permutations(phones_for_words_input, 2):
        tokens_a = tokenizer(a)
        tokens_b = tokenizer(b)
        if len(tokens_a) == len(tokens_b):
            jiwer_alignments.append(process_words(tokens_a, tokens_b))

    _phonemes = defaultdict(set)
    for word_substitution_alignment_pair in jiwer_alignments:
        reference_tokenized_ipa = tokenizer(
            "".join([r[0] for r in word_substitution_alignment_pair.references])
        )
        hyp_tokenized_ipa = tokenizer(
            "".join([r[0] for r in word_substitution_alignment_pair.hypotheses])
        )

        idx2alignment = {
            idx: alignmentchunk[0]
            for idx, alignmentchunk in enumerate(
                word_substitution_alignment_pair.alignments, start=0
            )
        }

        previous_label = "NONE"
        start = 0
        idx = 0
        symbols_r = []
        symbols_h = []

        for idx, alignment in idx2alignment.items():
            if alignment.type != previous_label and idx != 0:
                _phonemes[(start, idx)].add(tuple(symbols_r))
                _phonemes[(start, idx + 1)].add(tuple(symbols_h))
                symbols_r = []
                symbols_h = []
                start = idx

            symbols_r = reference_tokenized_ipa[start : idx + alignment.ref_end_idx]
            symbols_h = hyp_tokenized_ipa[start : idx + alignment.hyp_end_idx]

            _phonemes[(start, idx)].add(tuple(symbols_r))
            _phonemes[(start, idx + 1)].add(tuple(symbols_h))

    # Collect the phoneme-string alternatives for each span, ordered by
    # span start position.
    ordered_alternatives = [
        phone_set for _span, phone_set in sorted(_phonemes.items(), key=lambda x: x[0][0])
    ]

    # Build every combination across spans and join into flat strings.
    result_phones = []
    for combo in product(*ordered_alternatives):
        result_phones.append("".join("".join(p) for p in combo))

    return result_phones
