# Language phoneme data

This directory contains the phonological feature tables for 100
languages supported by phone-similarity.

## Data format

All language data lives in a single compressed JSON file
(`_data.json`).  Each entry is keyed by a language identifier
(e.g. `eng_us`, `fra`, `jpn`) and contains:

| Field                | Type                            | Description |
|----------------------|---------------------------------|-------------|
| `vowels`             | `list[str]`                     | Sorted IPA vowel inventory |
| `phonemes`           | `dict[str, dict[str, bool\|str]]` | Per-phoneme articulatory features |
| `consonant_columns`  | `list[str]`                     | Feature column names for consonants |
| `vowel_columns`      | `list[str]`                     | Feature column names for vowels |
| `modifier_columns`   | `list[str]` *(optional)*        | Feature column names for modifiers (stress markers, length, etc.) |

The `_loader.py` module reads `_data.json` lazily and exposes each
language as a namespace with `VOWELS_SET`, `PHONEME_FEATURES`, and
`FEATURES` attributes -- the same interface the old per-language
`.py` modules provided.

## Feature set reduction

By default the stored column sets are used (backward-compatible with
the original hand-curated files).  Pass `reduce_features=True` to
`LANGUAGES.build_spec()` to instead derive columns from the actual
phoneme data, dropping columns that never produce a 1-bit in the
bitarray encoding.

## Provenance

The phoneme inventories are aligned with **CharsiuG2P** pronunciation
dictionaries:

> Zhu, J., Zhang, C., & Jurgens, D. (2022).  ByT5 model for massively
> multilingual grapheme-to-phoneme conversion.  *Proceedings of
> INTERSPEECH 2022*.
> <https://github.com/lingjzhu/CharsiuG2P>

Charsiu provides word-to-IPA dictionaries (TSV files hosted on
GitHub) for 100+ languages.  These dictionaries define the **phoneme
inventory** -- the set of IPA symbols that appear in a language's
pronunciation entries.

The **articulatory feature decomposition** (voicing, place of
articulation, manner, height, rounding, etc.) was curated by cross-
referencing each phoneme against:

* **Panphon** -- Mortensen, D. R. (2017).  PanPhon: A resource for
  mapping IPA segments to articulatory feature vectors.
  <https://github.com/dmort27/panphon>
* **Wikipedia** phonology articles for each language (consonant and
  vowel tables).

The test suite (`tests/test_language_phonemes.py`,
`tests/test_phoneme_features.py`) enforces that every phoneme
in the Charsiu dictionaries has a corresponding feature entry,
and that feature assignments are internally consistent (vowels
have height, consonants have manner and voicing, etc.).

## Language code mapping

Charsiu uses hyphenated codes (e.g. `eng-us`, `fra-qu`).  The
language data keys use underscores (e.g. `eng_us`, `fra_qu`).  The
conversion is:

```python
charsiu_code = module_key.replace("_", "-")
module_key   = charsiu_code.replace("-", "_")
```

The full list of Charsiu codes and their ISO-639-3 mappings is in
`phone_similarity.g2p.charsiu.LANGUAGE_CODES_CHARSIU`.
