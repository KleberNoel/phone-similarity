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
    
        result_phones = [
            "".join(p)
            for p in product(
                sorted(_phonemes.items(), key=lambda x: x[0][0]),
                repeat=2
            ):
                item0 = item[0] if isinstance(item, tuple) else item
                item1 = item[1] if isinstance(item, tuple) else item
                p = (item0, item1)
                result_phones.append("".join(p))
        return result_phones
