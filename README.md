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

Use `phone_similarity.language.combined` for multi-language projects.

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
python -m pip install -e . pytest --extra-index-url "https://download.pytorch.org/whl/cpu"
```

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

## Attribution

This project uses the [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P) project for:
- Underlying G2P (grapheme-to-phoneme) models
- Language-specific phonetic dictionaries

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
