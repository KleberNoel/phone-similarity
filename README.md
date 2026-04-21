# phone-similarity

Phonological distance and similarity metrics for cross-lingual analysis.

Computes feature-weighted edit distances between IPA transcriptions, scans foreign-language dictionaries for phonological near-matches, and discovers multi-word interlingual puns via beam search segmentation. Accelerated by a Cython backend.

## Key Capabilities

- Feature-weighted edit distance using 24 articulatory features (voicing, place, manner, etc.)
- Cython + OpenMP acceleration for dictionary-scale scanning (4-10x faster)
- Pre-tokenized dictionary caching with automatic disk invalidation
- Parallel multi-language scanning via `ProcessPoolExecutor`
- Beam search segmentation for multi-word foreign phrase matching
- 100+ languages via CharsiuG2P (dictionary lookup + ONNX neural inference)

## Installation

```bash
pip install phone-similarity
```

### Optional extras

| Extra | Command | Adds |
|-------|---------|------|
| Cython acceleration | `pip install phone-similarity[dev]` then `python setup.py build_ext --inplace` | Compiled C hot paths |
| G2P support | `pip install phone-similarity[g2p]` | transformers + ONNX Runtime |

### Charsiu G2P usage policy

The neural CharsiuG2P generator is intended for research usage only.
Training/evaluation data quality varies significantly by language.

- Dictionary lookup (`.pdict`, `get_phones_from_dict`) is enabled by default.
- Neural generation (`.generate`) requires explicit opt-in:

```python
from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator

g2p = CharsiuGraphemeToPhonemeGenerator("eng-us", allow_research_g2p=True)
phones, probs = g2p.generate(("hello",))
```

You can also opt in via environment variable:

```bash
export PHONE_SIM_ALLOW_RESEARCH_G2P=1
```

### From source

```bash
git clone https://github.com/klebster2/phono-sim.git
cd phono-sim
pip install -e ".[dev,g2p]"
python setup.py build_ext --inplace
```

> The G2P backend uses ONNX Runtime (via HuggingFace Optimum). No PyTorch installation required.

## Quick Start

### Compare two IPA strings

```python
from phone_similarity.language import LANGUAGES

dist = LANGUAGES.build_distance("eng_us")

dist.hamming("kæt", "kæb")                    # ~0.97
dist.edit_distance("kæt", "kæb")              # ~0.12
dist.normalised_edit_distance("kæt", "kæb")   # ~0.04
```

### Scan a foreign dictionary

```python
from phone_similarity import cached_pretokenize_dictionary, reverse_dictionary_lookup
from phone_similarity.language import LANGUAGES
from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator

eng_spec = LANGUAGES.build_spec("eng_us")
eng = LANGUAGES["eng_us"]

fra_spec = LANGUAGES.build_spec("fra")
fra = LANGUAGES["fra"]
fra_g2p = CharsiuGraphemeToPhonemeGenerator("fra")

ptd = cached_pretokenize_dictionary(
    lambda: fra_g2p.pdict, fra_spec, lang="fra",
)

matches = reverse_dictionary_lookup(
    source_ipa="mjuzɪk",
    source_lang_code="eng-us",
    source_spec=eng_spec,
    source_phoneme_features=eng.PHONEME_FEATURES,
    target_lang_code="fra",
    target_spec=fra_spec,
    target_phoneme_features=fra.PHONEME_FEATURES,
    target_dictionary={},
    pre_tokenized=ptd,
    top_n=5,
    max_distance=0.40,
)
for word, ipa, dist in matches:
    print(f"  {word:20s} /{ipa}/  d={dist:.3f}")
```

```
  bionique             /bjɔnik/  d=0.253
  bioniques            /bjɔnik/  d=0.253
  moujik               /muʒik/  d=0.264
  moujiks              /muʒik/  d=0.264
  new-look             /njuluk/  d=0.272
```

### Multi-word beam search

```python
from phone_similarity import beam_search_segmentation
from phone_similarity.language import LANGUAGES

eng_spec = LANGUAGES.build_spec("eng_us")
eng = LANGUAGES["eng_us"]
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

candidates = invert_features(eng.PHONEME_FEATURES["ɹ"], fra.PHONEME_FEATURES, top_n=3)
for phoneme, dist in candidates:
    print(f"  /{phoneme}/  d={dist:.3f}")
```

## Architecture

```
IPA string
  -> clean_phones()                  Strip stress markers, NFKD normalise
  -> ipa_tokenizer()                 Greedy longest-match against phoneme inventory
  -> feature_edit_distance()         Levenshtein DP with gradient substitution cost
  -> reverse_dictionary_lookup()     Single-language scan
     or parallel_dictionary_scan()   Multi-language fan-out
  -> beam_search_segmentation()      Multi-word segmentation
```

## Module Map

| Module | Purpose |
|--------|---------|
| `primitives` | Hamming distance, phoneme feature distance, feature edit distance |
| `distance_class` | High-level `Distance` API wrapping `BitArraySpecification` |
| `pretokenize` | `PreTokenizedDictionary`, disk caching |
| `dictionary_scan` | `reverse_dictionary_lookup`, `parallel_dictionary_scan` |
| `inversion` | Reverse feature-to-phoneme lookup |
| `cross_language` | Pairwise comparison across languages |
| `beam_search` | Multi-word segmentation |
| `coarticulation` | Co-articulation distance modelling |
| `syllable` | Syllabification, sonority scale, max onset segmenter |
| `universal_features` | Panphon-backed feature encoder |
| `bit_array_specification` | Phoneme-to-bitarray encoding |
| `clean_phones` | IPA cleaning, stress extraction, NFKD normalisation |
| `language` | Per-language phoneme feature tables (lazy-loaded registry) |
| `analysis` | Entropy and bit utilisation diagnostics |
| `_dispatch` | Cython extension detection (Chain of Responsibility) |
| `g2p` | CharsiuG2P grapheme-to-phoneme backend |

## Performance

With Cython compiled:

| Benchmark | Time |
|-----------|------|
| Dictionary scan (French, 245K entries) | ~31 ms |
| Dictionary scan (Dutch, 148K entries) | ~9 ms |
| Feature edit distance vs pure Python | ~4x faster |
| Memory delta over repeated scans | +0.001 MB |

## G2P Model

Uses an ONNX-exported ByT5-tiny model from [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P), hosted at [`klebster/g2p_multilingual_byT5_tiny_onnx`](https://huggingface.co/klebster/g2p_multilingual_byT5_tiny_onnx) on HuggingFace Hub. Loads lazily on first `.generate()` call. Dictionary-only usage does not require the ML runtime.

## Attribution

- Mortensen, D. R., Dalmia, S., & Littell, P. (2018). Epitran: Precision G2P for many languages. *LREC 2018*. [GitHub](https://github.com/dmort27/panphon)
- Zhu, J., Zhang, C., & Jurgens, D. (2022). ByT5 model for massively multilingual grapheme-to-phoneme conversion. *INTERSPEECH 2022*. [GitHub](https://github.com/lingjzhu/CharsiuG2P)

## License

Apache-2.0
