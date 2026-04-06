---
language:
  - multilingual
  - af
  - am
  - sq
  - ar
  - hy
  - az
  - eu
  - be
  - bs
  - bg
  - my
  - ca
  - zh
  - hr
  - cs
  - da
  - nl
  - en
  - eo
  - et
  - fi
  - fr
  - ka
  - de
  - el
  - hi
  - hu
  - is
  - id
  - it
  - ja
  - kk
  - km
  - ko
  - ku
  - la
  - lt
  - lb
  - mk
  - ms
  - mt
  - mi
  - nb
  - or
  - fa
  - pl
  - pt
  - ro
  - ru
  - sa
  - sr
  - sk
  - sl
  - es
  - sw
  - sv
  - tl
  - ta
  - tt
  - th
  - tr
  - tk
  - uk
  - ur
  - ug
  - uz
  - vi
  - cy
  - sd
license: cc-by-4.0
library_name: optimum
tags:
  - onnx
  - t5
  - g2p
  - grapheme-to-phoneme
  - phonetics
  - phonology
  - ipa
  - byt5
  - text2text-generation
  - charsiu
  - multilingual
pipeline_tag: text2text-generation
base_model: charsiu/g2p_multilingual_byT5_tiny_16_layers_100
model-index:
  - name: g2p_multilingual_byT5_tiny_onnx
    results: []
---

# Multilingual Grapheme-to-Phoneme (G2P) ByT5 Tiny — ONNX

ONNX export of [charsiu/g2p_multilingual_byT5_tiny_16_layers_100](https://huggingface.co/charsiu/g2p_multilingual_byT5_tiny_16_layers_100), a _massively multilingual grapheme-to-phoneme model covering ~100 languages_.
This model converts written words into IPA (International Phonetic Alphabet) transcriptions.

## Model Details

The model was trained on a large set of data including a mix of licenses.

T5ForConditionalGeneration
ByT5-tiny - 16 layers
Phoneme Error Rate: 9.5%
Word Error Rate: 27.7%
20.6M Parameters

See the model repo [here](https://github.com/lingjzhu/CharsiuG2P) for information about the methodology, literature, training, and data used to train this model.
See the original pytorch weights on huggingface [here](https://huggingface.co/charsiu/g2p_multilingual_byT5_tiny_16_layers)

### Model Description

This is an ONNX-exported version of the CharsiuG2P multilingual ByT5-tiny model.
The original model was trained on byte-level T5 (ByT5) to perform grapheme-to-phoneme conversion across approximately 100 languages.
The ONNX export enables inference without PyTorch, using only ONNX Runtime.

- **Developed by:** [Jian Zhu](https://github.com/lingjzhu), [Cong Zhang](https://congzhang.name/), [David Jurgens](https://jurgens.people.si.umich.edu/) (original model); ONNX export by [klebster](https://huggingface.co/klebster)
- **Model type:** Encoder-decoder (T5ForConditionalGeneration), exported to ONNX
- **Language(s):** ~100 languages (see [Supported Languages](#supported-languages) below)
- **License:** [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) (following the original model)
- **Finetuned from:** [google/byt5-small](https://huggingface.co/google/byt5-small) (via the original CharsiuG2P training)
- **Base model:** [charsiu/g2p_multilingual_byT5_tiny_16_layers_100](https://huggingface.co/charsiu/g2p_multilingual_byT5_tiny_16_layers_100)

### Model Sources

- **Original Repository:** [https://github.com/lingjzhu/CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P)
- **Paper:** [ByT5 model for massively multilingual grapheme-to-phoneme conversion](https://arxiv.org/abs/2204.03067) (Interspeech 2022)
- **Original HuggingFace model:** [charsiu/g2p_multilingual_byT5_tiny_16_layers_100](https://huggingface.co/charsiu/g2p_multilingual_byT5_tiny_16_layers_100)
- **ONNX export consumer:** [phone-similarity](https://github.com/klebster/phone-similarity) library

## Uses

### Direct Use

Convert written words to IPA phonemic transcriptions. Input format is `<language_code>: word`, where `language_code` is a CharsiuG2P language code (e.g., `eng-us`, `fra`, `ger`).

### Downstream Use

- Phonological distance/similarity computation
- Pronunciation dictionaries
- Text-to-speech preprocessing
- Linguistic research on phonological systems
- Language learning applications

### Out-of-Scope Use

- This model is not intended for audio/speech processing (it operates on text only)
- Not suitable for sentence-level or free-form text generation
- Performance on languages not in the training set is not guaranteed (though zero-shot transfer to unseen languages may work to some extent, per the original paper)

## How to Get Started with the Model

### With HuggingFace Optimum (recommended)

```python
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

model = ORTModelForSeq2SeqLM.from_pretrained("klebster/g2p_multilingual_byT5_tiny_onnx")
tokenizer = AutoTokenizer.from_pretrained("klebster/g2p_multilingual_byT5_tiny_onnx")

# English (US)
# Please note, the prefixes '<language>:' are required; without themoutput may not work as expected
inputs = tokenizer("<eng-us>: hello", padding=True, add_special_tokens=False, return_tensors="pt")
preds = model.generate(**inputs, num_beams=1, max_length=50)
print(tokenizer.decode(preds[0], skip_special_tokens=True))
# Output: ˈhɛɫoʊ

# French
inputs = tokenizer("<fra>: bonjour", padding=True, add_special_tokens=False, return_tensors="pt")
preds = model.generate(**inputs, num_beams=1, max_length=50)
print(tokenizer.decode(preds[0], skip_special_tokens=True))
# Output: bɔ̃ʒuʁ

# German
inputs = tokenizer("<ger>: Straße", padding=True, add_special_tokens=False, return_tensors="pt")
preds = model.generate(**inputs, num_beams=1, max_length=50)
print(tokenizer.decode(preds[0], skip_special_tokens=True))
# Output: ˈʃtʁaːsə
```

### With the phone-similarity library

```python
from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator

g2p = CharsiuGraphemeToPhonemeGenerator("eng-us")

# Dictionary lookup (no model needed)
phones = g2p.pdict.get("hello")
print(phones)  # ˈhɛɫoʊ

# Model-based generation (model loads lazily on first call)
phones, scores = g2p.generate(("hello",), num_beams=1, max_length=50)
print(phones)  # ['ˈhɛɫoʊ']
```

## Technical Specifications

### Model Architecture and Objective

The model is a T5 (Text-to-Text Transfer Transformer) encoder-decoder, specifically the ByT5 variant that operates on raw bytes rather than subword tokens.

| Parameter            | Value                                   |
| -------------------- | --------------------------------------- |
| Architecture         | `T5ForConditionalGeneration`            |
| Base                 | ByT5 (byte-level)                       |
| `d_model`            | 256                                     |
| `d_ff`               | 1024                                    |
| `d_kv`               | 64                                      |
| `num_heads`          | 6                                       |
| `num_encoder_layers` | 12                                      |
| `num_decoder_layers` | 4                                       |
| `vocab_size`         | 384                                     |
| `feed_forward_proj`  | `gated-gelu`                            |
| Total parameters     | ~20.6M                                  |
| Tokenizer            | `ByT5Tokenizer` (byte-level, 384 vocab) |

### ONNX Export Details

The model was exported using [HuggingFace Optimum](https://huggingface.co/docs/optimum/) (`ORTModelForSeq2SeqLM.from_pretrained(..., export=True)`), producing three ONNX graphs:

| File                           | Size        | Purpose                                                                        |
| ------------------------------ | ----------- | ------------------------------------------------------------------------------ |
| `encoder_model.onnx`           | 57.2 MB     | Encodes input bytes (runs once per input)                                      |
| `decoder_model.onnx`           | 26.1 MB     | First decoding step (no KV cache)                                              |
| `decoder_with_past_model.onnx` | 23.0 MB     | Subsequent decoding steps (with KV cache for faster autoregressive generation) |
| **Total**                      | **~106 MB** |                                                                                |

The ONNX output has been validated to be **identical** to the PyTorch output across all supported languages for both greedy decoding and beam search (including `sequences_scores`).

### Compute Infrastructure

#### Export Hardware

- ONNX export performed on CPU
- Exported using `optimum>=1.17` with `onnxruntime>=1.17`

#### Software

- `transformers>=4.34`
- `optimum[onnxruntime]>=1.17`
- Python 3.10+

#### Inference

- CPU-only inference via ONNX Runtime (no GPU required)
- No PyTorch dependency needed for inference

## Training Details

### Training Data

The original model was trained on G2P data curated from various sources covering ~100 languages. See the [CharsiuG2P repository](https://github.com/lingjzhu/CharsiuG2P) and the [original paper](https://arxiv.org/abs/2204.03067) for full training data details.

### Training Procedure

This ONNX model was **not retrained** — it is a format conversion of the original PyTorch weights. The original training procedure is documented in [Zhu et al. (2022)](https://arxiv.org/abs/2204.03067).

#### Training Hyperparameters

Refer to the original paper and [CharsiuG2P repository](https://github.com/lingjzhu/CharsiuG2P) for training hyperparameters.

## Evaluation

### Results

The ONNX export produces **bit-identical** output to the original PyTorch model for:

- Greedy decoding across 14 tested languages (dan, dut, eng-uk, eng-us, fra, ger, gre, ita, lat-clas, por-bz, por-po, rus, spa, sqi)
- Beam search with `num_return_sequences > 1`
- `sequences_scores` values

For phoneme error rates and cross-lingual evaluation results, refer to the [original paper](https://arxiv.org/abs/2204.03067).

## Bias, Risks, and Limitations

- **Training data bias:** The model may reflect biases in the pronunciation dictionaries it was trained on. Some languages have more training data than others.
- **Dialect coverage:** Language codes like `eng-us` and `eng-uk` represent standard dialects; regional or non-standard pronunciations may not be well represented.
- **Out-of-vocabulary:** While ByT5 can process any UTF-8 input, accuracy on scripts or languages not seen during training may be poor.
- **IPA consistency:** IPA transcription conventions vary across sources; the model may produce transcriptions that follow one convention over another.

### Recommendations

- For best results, use the pronunciation dictionary (`.pdict`) first and fall back to model generation only for out-of-vocabulary words.
- The original authors note: "We do not find beam search helpful. Greedy decoding is enough." However, beam search is available for exploring alternative pronunciations.

## Supported Languages

The model supports approximately 100 languages via CharsiuG2P language codes. A selection:

| Code       | Language             | Code        | Language                |
| ---------- | -------------------- | ----------- | ----------------------- |
| `eng-us`   | English (US)         | `eng-uk`    | English (UK)            |
| `fra`      | French               | `fra-qu`    | French (Quebec)         |
| `ger`      | German               | `ita`       | Italian                 |
| `spa`      | Spanish              | `spa-latin` | Spanish (Latin America) |
| `por-bz`   | Portuguese (Brazil)  | `por-po`    | Portuguese (Portugal)   |
| `rus`      | Russian              | `dan`       | Danish                  |
| `dut`      | Dutch                | `swe`       | Swedish                 |
| `gre`      | Greek                | `pol`       | Polish                  |
| `jpn`      | Japanese             | `kor`       | Korean                  |
| `zho-s`    | Chinese (Simplified) | `zho-t`     | Chinese (Traditional)   |
| `ara`      | Arabic               | `hin`       | Hindi                   |
| `tur`      | Turkish              | `vie-n`     | Vietnamese (Northern)   |
| `lat-clas` | Latin (Classical)    | `sqi`       | Albanian                |

For the full list of 101 language codes, see the [CharsiuG2P dictionaries](https://github.com/lingjzhu/CharsiuG2P/tree/main/dicts).

## ONNX Export Reproduction

To reproduce the ONNX export from the original PyTorch model:

```python
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

# Export from PyTorch to ONNX
model = ORTModelForSeq2SeqLM.from_pretrained(
    "charsiu/g2p_multilingual_byT5_tiny_16_layers_100",
    export=True
)
model.save_pretrained("onnx_export/")

# Save tokenizer alongside
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
tokenizer.save_pretrained("onnx_export/")

# Validate
model = ORTModelForSeq2SeqLM.from_pretrained("onnx_export/")
inputs = tokenizer("<eng-us>: hello", padding=True, add_special_tokens=False, return_tensors="pt")
preds = model.generate(**inputs, num_beams=1, max_length=50)
print(tokenizer.decode(preds[0], skip_special_tokens=True))
# Expected: ˈhɛɫoʊ
```

Or use the provided script:

```bash
pip install torch transformers optimum[onnxruntime]
python scripts/export_onnx.py --push-to-hub klebster/g2p_multilingual_byT5_tiny_onnx
```

## Citation

If you use this model, please cite the original CharsiuG2P paper:

**BibTeX:**

```bibtex
@misc{zhu2022byt5modelmassivelymultilingual,
      title={ByT5 model for massively multilingual grapheme-to-phoneme conversion},
      author={Jian Zhu and Cong Zhang and David Jurgens},
      year={2022},
      eprint={2204.03067},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2204.03067},
}
```

**APA:**

Zhu, J., Zhang, C., & Jurgens, D. (2022). ByT5 model for massively multilingual grapheme-to-phoneme conversion. _Proceedings of Interspeech 2022_. https://arxiv.org/abs/2204.03067

## Glossary

- **G2P (Grapheme-to-Phoneme):** The task of converting written text (graphemes) to phonemic/phonetic transcriptions (phonemes), typically in IPA.
- **IPA (International Phonetic Alphabet):** A standardized system for transcribing the sounds of spoken language.
- **ByT5:** A byte-level variant of the T5 model that operates directly on UTF-8 bytes rather than subword tokens, enabling processing of any language without a language-specific tokenizer.
- **KV Cache:** Key-value cache used in autoregressive decoding to avoid recomputing attention over previous tokens at each generation step.
- **ONNX (Open Neural Network Exchange):** An open format for representing machine learning models, enabling interoperability between frameworks.

## Model Card Authors

- [klebster](https://huggingface.co/klebster) (ONNX export and model card)

## Model Card Contact

- HuggingFace: [klebster](https://huggingface.co/klebster)
- For issues with the original model: [CharsiuG2P GitHub Issues](https://github.com/lingjzhu/CharsiuG2P/issues)
