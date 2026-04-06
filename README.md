# phone-bitarray

An efficient library for phonological (feature) representations of language.

## Multilingual Support

The library supports 15 languages with BCP-47 language codes:

- `da-DK` (Danish)
- `nl-NL` (Dutch)
- `en-GB` (English - UK)
- `en-US` (English - US)
- `fr-FR` (French)
- `de-DE` (German)
- `el-GR` (Greek)
- `it-IT` (Italian)
- `la-Latn` (Latin - Classical)
- `pt-BR` (Portuguese - Brazil)
- `pt-PT` (Portuguese - Portugal)
- `ru-RU` (Russian)
- `es-ES` (Spanish)
- `sq-AL` (Albanian)

Each language module provides `FEATURES`, `PHONEME_FEATURES`, and `VOWELS_SET` for phonetic analysis.

## Phonetic Processing Pipeline

1. **IPA String**: Input text in International Phonetic Alphabet notation
2. **NFKD Normalization**: Unicode normalization using `unicodedata.normalize('NFKD', ...)`
3. **Tokenization**: IPA string is tokenized into phoneme units
4. **Bitarray Generation**: Each phoneme is converted to a bitarray based on its features

## Ignored Phonetic Symbols

The following symbols are intentionally ignored during similarity calculations:
- Stress markers: `ˈ` (primary), `ˌ` (secondary)
- Length marks: `ː` (long), `ˑ` (half-long)
- Linking markers: `‿`
- Other modifiers that do not contribute to core phonetic similarity

## Installation

```bash
python -m pip install -e . pytest
```

> **Note**: This library uses ONNX Runtime (via HuggingFace Optimum) instead of PyTorch for G2P inference, so no PyTorch installation or `--extra-index-url` is needed.

## Example Usage

```python
from phone_similarity.bit_array_specification import BitArraySpecification
from phone_similarity.language.eng_us import FEATURES, PHONEME_FEATURES, VOWELS_SET

spec = BitArraySpecification(
    vowels=VOWELS_SET,
    consonants=set(PHONEME_FEATURES.keys()) - VOWELS_SET,
    features_per_phoneme=PHONEME_FEATURES,
    features=FEATURES
)

bitarray = spec.generate("strɪŋz")
```

## G2P Model

The grapheme-to-phoneme (G2P) backend uses an ONNX-exported version of the multilingual ByT5-tiny model from [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P), hosted at [`klebster/g2p_multilingual_byT5_tiny_onnx`](https://huggingface.co/klebster/g2p_multilingual_byT5_tiny_onnx) on HuggingFace Hub.

The model is loaded **lazily** — it is only downloaded and initialized on the first call to `.generate()`. Dictionary-only usage (`.pdict`) does not require the ML runtime.

## Attribution

This project uses the [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P) project for:
- Underlying G2P (grapheme-to-phoneme) models
- Language-specific phonetic dictionaries

> Zhu, J., Zhang, C., & Jurgens, D. (2022). ByT5 model for massively multilingual grapheme-to-phoneme conversion. *Proceedings of INTERSPEECH 2022*. [GitHub](https://github.com/lingjzhu/CharsiuG2P)

## Resources

### Models

The library uses bit arrays to represent phonetic features with a fixed schema:
- **onset**: consonant features
- **nucleus**: vowel features  
- **coda**: consonant features

### Requirements

See `pyproject.toml` for dependencies.

## Table of contents

TBD
