# phone-similarity

Phonological distance and similarity metrics for cross-lingual analysis.

`phone-similarity` computes feature-weighted edit distances between IPA
transcriptions, scans foreign-language dictionaries for phonological
near-matches, and discovers multi-word interlingual puns via beam search
segmentation -- all accelerated by a Cython backend.

## Features

- **Feature-weighted edit distance** -- substitution cost is proportional to
  articulatory difference, not a flat 1.0
- **Cython-accelerated hot paths** -- Hamming distance, edit distance,
  dictionary scanning, IPA tokenization, and feature inversion all dispatch
  to compiled C when available (4--10x faster)
- **Pre-tokenized dictionary caching** -- numpy-backed compact storage with
  automatic disk caching and invalidation
- **Parallel multi-language scanning** -- fan out across target languages
  with `ProcessPoolExecutor`
- **Beam search segmentation** -- find optimal multi-word foreign
  segmentations for a source phrase
- **Phonetic embeddings & ANN** -- approximate nearest-neighbor pre-filtering
  with brute-force or KD-tree indices
- **100+ languages** via the CharsiuG2P grapheme-to-phoneme backend
  (dictionary lookup + ONNX neural inference)

## Installation

```bash
pip install phone-similarity
```

With Cython acceleration (recommended):

```bash
pip install phone-similarity[dev]   # includes Cython, builds _core.so
```

For G2P (grapheme-to-phoneme) support:

```bash
pip install phone-similarity[g2p]   # adds transformers + ONNX Runtime
```

Development install from source:

```bash
git clone https://github.com/klebster2/phono-sim.git
cd phono-sim
pip install -e ".[dev,g2p]"
python setup.py build_ext --inplace   # compile Cython extension
```

> **Note**: The G2P backend uses ONNX Runtime (via HuggingFace Optimum)
> instead of PyTorch, so no PyTorch installation is needed.

## Quick start

### Compare two IPA strings

```python
from phone_similarity import Distance
from phone_similarity.language import LANGUAGES
from phone_similarity.bit_array_specification import BitArraySpecification

lang = LANGUAGES["eng_us"]
spec = BitArraySpecification(
    vowels=lang.VOWELS_SET,
    consonants=set(lang.PHONEME_FEATURES) - lang.VOWELS_SET,
    features=lang.FEATURES,
    features_per_phoneme=lang.PHONEME_FEATURES,
)
dist = Distance(spec)

# Hamming similarity (bitarray-level, fixed-width)
print(dist.hamming("kæt", "kæb"))        # ~0.97

# Feature-weighted edit distance (phoneme-sequence-level)
print(dist.edit_distance("kæt", "kæb"))  # ~0.12
print(dist.normalised_edit_distance("kæt", "kæb"))  # ~0.04
```

### Scan a foreign dictionary

```python
from phone_similarity import (
    cached_pretokenize_dictionary,
    reverse_dictionary_lookup,
)
from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator

# Load English source
eng = LANGUAGES["eng_us"]
eng_spec = BitArraySpecification(
    vowels=eng.VOWELS_SET,
    consonants=set(eng.PHONEME_FEATURES) - eng.VOWELS_SET,
    features=eng.FEATURES,
    features_per_phoneme=eng.PHONEME_FEATURES,
)

# Load French target
fra = LANGUAGES["fra"]
fra_spec = BitArraySpecification(
    vowels=fra.VOWELS_SET,
    consonants=set(fra.PHONEME_FEATURES) - fra.VOWELS_SET,
    features=fra.FEATURES,
    features_per_phoneme=fra.PHONEME_FEATURES,
)
fra_g2p = CharsiuGraphemeToPhonemeGenerator("fra")

# Pre-tokenize and cache the French dictionary
ptd = cached_pretokenize_dictionary(
    lambda: fra_g2p.pdict, fra_spec, lang="fra",
)

# Find French words closest to English "hello" (/hɛloʊ/)
matches = reverse_dictionary_lookup(
    source_ipa="hɛloʊ",
    source_lang_code="eng-us",
    source_spec=eng_spec,
    source_phoneme_features=eng.PHONEME_FEATURES,
    target_lang_code="fra",
    target_spec=fra_spec,
    target_phoneme_features=fra.PHONEME_FEATURES,
    target_dictionary={},  # ignored when pre_tokenized is given
    pre_tokenized=ptd,
    top_n=5,
    max_distance=0.30,
)
for word, ipa, dist in matches:
    print(f"  {word:20s} /{ipa}/  d={dist:.3f}")
```

### Multi-word beam search

```python
from phone_similarity import beam_search_segmentation

source_tokens = eng_spec.ipa_tokenizer("mɛɹikɹɪsməs")

results = beam_search_segmentation(
    source_tokens,
    source_spec=eng_spec,
    source_features=eng.PHONEME_FEATURES,
    target_ptd=ptd,
    target_spec=fra_spec,
    target_features=fra.PHONEME_FEATURES,
    beam_width=10,
    top_k=3,
    max_distance=0.40,
)
for r in results:
    print(f"  {' + '.join(r.words):30s}  /{r.glued_ipa}/  d={r.distance:.3f}")
```

### Feature-vector inversion

```python
from phone_similarity import invert_features

# What French phonemes are closest to English /r/?
candidates = invert_features(
    eng.PHONEME_FEATURES["ɹ"],
    fra.PHONEME_FEATURES,
    top_n=3,
)
for phoneme, dist in candidates:
    print(f"  /{phoneme}/  d={dist:.3f}")
```

## Multilingual support

The library ships phoneme feature tables for 15 languages:

| Code | Language | Code | Language |
|------|----------|------|----------|
| `eng_us` | English (US) | `ita` | Italian |
| `eng_uk` | English (UK) | `spa` | Spanish |
| `fra` | French | `por_br` | Portuguese (BR) |
| `deu` | German | `por_pt` | Portuguese (PT) |
| `dut` | Dutch | `rus` | Russian |
| `dan` | Danish | `alb` | Albanian |
| `gre` | Greek | `lat` | Latin |

G2P dictionaries and neural inference cover 100+ languages via
[CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P).

## Architecture

```
IPA string
  |
  v
clean_phones()          Strip stress markers, NFKD normalise
  |
  v
ipa_tokenizer()         Greedy longest-match against phoneme inventory
  |
  v
feature_edit_distance()  Levenshtein DP with gradient substitution cost
  |
  v
reverse_dictionary_lookup() / parallel_dictionary_scan()
  |                            |
  v                            v
beam_search_segmentation()   Multi-word segmentation via beam search
```

### Module map

| Module | Purpose |
|--------|---------|
| `primitives` | Hamming distance/similarity, phoneme feature distance, feature edit distance, batch pairwise Hamming |
| `distance_class` | `Distance` high-level API wrapping a `BitArraySpecification` |
| `pretokenize` | `PreTokenizedDictionary`, disk caching with `cached_pretokenize_dictionary` |
| `dictionary_scan` | `reverse_dictionary_lookup`, `parallel_dictionary_scan` |
| `inversion` | `invert_features`, `invert_ipa` -- reverse feature-to-phoneme lookup |
| `cross_language` | `compare_cross_language` -- pairwise comparison across languages |
| `beam_search` | `beam_search_segmentation`, `beam_search_phrases` -- multi-word segmentation |
| `embedding` | `PhoneticEmbedder`, `BruteForceIndex`, `KDTreeIndex`, `ann_dictionary_scan` |
| `bit_array_specification` | `BitArraySpecification` -- phoneme-to-bitarray encoding |
| `clean_phones` | IPA cleaning and NFKD normalisation |
| `language` | Per-language phoneme feature tables (lazy-loaded) |
| `g2p` | CharsiuG2P grapheme-to-phoneme backend |

## Performance

With Cython compiled (`_HAS_CYTHON = True`):

- Dictionary scan (French, 245K entries): ~31 ms
- Dictionary scan (Dutch, 148K entries): ~9 ms
- Feature edit distance: ~4x faster than pure Python
- No memory leaks: +0.001 MB delta over repeated scan cycles

## G2P model

The grapheme-to-phoneme backend uses an ONNX-exported ByT5-tiny model from
[CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P), hosted at
[`klebster/g2p_multilingual_byT5_tiny_onnx`](https://huggingface.co/klebster/g2p_multilingual_byT5_tiny_onnx)
on HuggingFace Hub.

The model loads **lazily** -- only downloaded on the first `.generate()` call.
Dictionary-only usage (`.pdict`) does not require the ML runtime.

## Attribution

> Zhu, J., Zhang, C., & Jurgens, D. (2022). ByT5 model for massively
> multilingual grapheme-to-phoneme conversion. *Proceedings of
> INTERSPEECH 2022*.
> [GitHub](https://github.com/lingjzhu/CharsiuG2P)

## License

Apache-2.0
