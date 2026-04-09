# TODO - phone-similarity

## 1. Language Coverage (101 languages)

Each language module requires: `VOWELS_SET`, `PHONEME_FEATURES`, `CONSONANT_COLUMNS`, `VOWEL_COLUMNS`, `FEATURES` exports.
Per-language work: extract phoneme inventory from CharsiuG2P dictionary, cross-reference against Wikipedia phonology / Panphon, create module in `src/phone_similarity/language/`, add parametrized test coverage.
Dictionaries are downloaded on demand from CharsiuG2P GitHub and cached in `~/.cache/phono-sim/dicts/`.

### Germanic (7 missing, 4 done)

- [x] **dan** (Danish) -- audit against Panphon/Wikipedia for completeness
- [x] **dut** (Dutch) -- audit against Panphon/Wikipedia for completeness
- [x] **eng_uk** (English UK) -- audit against Panphon/Wikipedia for completeness
- [x] **eng_us** (English US) -- `ʌ` and `ɚ` added to PHONEME_FEATURES; audit complete
- [x] **ger** (German) -- audit against Panphon/Wikipedia for completeness
- [ ] **afr** (Afrikaans) -- module + dictionary + tests
- [ ] **ang** (Old English) -- module + dictionary + tests
- [ ] **enm** (Middle English) -- module + dictionary + tests
- [ ] **ice** (Icelandic) -- module + dictionary + tests; note: `isl` also present as separate code
- [ ] **isl** (Icelandic alt code) -- verify if duplicate of `ice` or distinct dictionary; module + dictionary + tests
- [ ] **ltz** (Luxembourgish) -- module + dictionary + tests
- [ ] **nob** (Norwegian Bokmal) -- module + dictionary + tests
- [ ] **swe** (Swedish) -- module + dictionary + tests

### Romance (6 missing, 6 done)

- [x] **fra** (French) -- audit against Panphon/Wikipedia for completeness
- [x] **ita** (Italian) -- audit against Panphon/Wikipedia for completeness
- [x] **lat_clas** (Classical Latin) -- audit against Panphon/Wikipedia for completeness
- [x] **por_bz** (Brazilian Portuguese) -- audit against Panphon/Wikipedia for completeness
- [x] **por_po** (European Portuguese) -- audit against Panphon/Wikipedia for completeness
- [x] **spa** (Spanish) -- audit against Panphon/Wikipedia for completeness
- [ ] **arg** (Aragonese) -- module + dictionary + tests
- [ ] **cat** (Catalan) -- module + dictionary + tests
- [ ] **fra_qu** (Quebec French) -- module + dictionary + tests
- [ ] **glg** (Galician) -- module + dictionary + tests
- [ ] **lat_eccl** (Ecclesiastical Latin) -- module + dictionary + tests
- [ ] **pap** (Papiamento) -- module + dictionary + tests
- [ ] **ron** (Romanian) -- module + dictionary + tests
- [ ] **spa_latin** (Latin American Spanish) -- module + dictionary + tests
- [ ] **spa_me** (Mexican Spanish) -- module + dictionary + tests

### Slavic (13 missing)

- [x] **rus** (Russian) -- audit against Panphon/Wikipedia for completeness
- [ ] **bel** (Belarusian) -- module + dictionary + tests
- [ ] **bos** (Bosnian) -- module + dictionary + tests
- [ ] **bul** (Bulgarian) -- module + dictionary + tests
- [ ] **cze** (Czech) -- module + dictionary + tests
- [ ] **hbs_cyrl** (Serbo-Croatian Cyrillic) -- module + dictionary + tests
- [ ] **hbs_latn** (Serbo-Croatian Latin) -- module + dictionary + tests
- [ ] **mac** (Macedonian) -- module + dictionary + tests
- [ ] **pol** (Polish) -- module + dictionary + tests
- [ ] **slk** (Slovak) -- module + dictionary + tests
- [ ] **slo** (Slovenian alt code) -- module + dictionary + tests
- [ ] **slv** (Slovenian) -- verify if duplicate of `slo`; module + dictionary + tests
- [ ] **srp** (Serbian) -- module + dictionary + tests
- [ ] **ukr** (Ukrainian) -- module + dictionary + tests

### Turkic (7 missing)

- [ ] **aze** (Azerbaijani) -- module + dictionary + tests
- [ ] **bak** (Bashkir) -- module + dictionary + tests
- [ ] **kaz** (Kazakh) -- module + dictionary + tests
- [ ] **tat** (Tatar) -- module + dictionary + tests
- [ ] **tuk** (Turkmen) -- module + dictionary + tests
- [ ] **tur** (Turkish) -- module + dictionary + tests
- [ ] **uzb** (Uzbek) -- module + dictionary + tests

### Uralic (4 missing)

- [ ] **est** (Estonian) -- module + dictionary + tests
- [ ] **fin** (Finnish) -- module + dictionary + tests
- [ ] **hun** (Hungarian) -- module + dictionary + tests
- [ ] **sme** (Northern Sami) -- module + dictionary + tests

### Indo-Iranian (7 missing)

- [ ] **fas** (Persian/Farsi) -- module + dictionary + tests
- [ ] **hin** (Hindi) -- module + dictionary + tests
- [ ] **kur** (Kurdish) -- module + dictionary + tests
- [ ] **ori** (Odia) -- module + dictionary + tests
- [ ] **san** (Sanskrit) -- module + dictionary + tests
- [ ] **snd** (Sindhi) -- module + dictionary + tests
- [ ] **urd** (Urdu) -- module + dictionary + tests

### Celtic (3 missing)

- [ ] **gle** (Irish) -- module + dictionary + tests
- [ ] **wel_nw** (Welsh, North) -- module + dictionary + tests
- [ ] **wel_sw** (Welsh, South) -- module + dictionary + tests

### Armenian (2 missing)

- [ ] **arm_e** (Eastern Armenian) -- module + dictionary + tests
- [ ] **arm_w** (Western Armenian) -- module + dictionary + tests

### Baltic (1 missing)

- [ ] **lit** (Lithuanian) -- module + dictionary + tests

### Hellenic (1 done, 1 missing)

- [x] **gre** (Modern Greek) -- audit against Panphon/Wikipedia for completeness
- [ ] **grc** (Ancient Greek) -- module + dictionary + tests

### Albanian (1 done)

- [x] **sqi** (Albanian) -- audit against Panphon/Wikipedia for completeness

### Semitic & Afroasiatic (5 missing)

- [ ] **amh** (Amharic) -- module + dictionary + tests
- [ ] **ara** (Arabic) -- module + dictionary + tests; note complex coda clusters, profile syllable length
- [ ] **egy** (Egyptian Arabic) -- module + dictionary + tests
- [ ] **mlt** (Maltese) -- module + dictionary + tests
- [ ] **syc** (Syriac) -- module + dictionary + tests

### Caucasian (2 missing)

- [ ] **ady** (Adyghe) -- module + dictionary + tests; large consonant inventory
- [ ] **geo** (Georgian) -- module + dictionary + tests; complex onset/coda clusters, profile syllable length

### East Asian (5 missing)

- [ ] **jpn** (Japanese) -- module + dictionary + tests; mora-timed, CV structure
- [ ] **kor** (Korean) -- module + dictionary + tests
- [ ] **yue** (Cantonese) -- module + dictionary + tests; tonal
- [ ] **zho_s** (Mandarin Simplified) -- module + dictionary + tests; tonal
- [ ] **zho_t** (Mandarin Traditional) -- module + dictionary + tests; tonal

### Southeast Asian (11 missing)

- [ ] **bur** (Burmese) -- module + dictionary + tests; tonal
- [ ] **ind** (Indonesian) -- module + dictionary + tests
- [ ] **khm** (Khmer) -- module + dictionary + tests; complex clusters, no tones
- [ ] **msa** (Malay) -- module + dictionary + tests
- [ ] **nan** (Min Nan / Hokkien) -- module + dictionary + tests; tonal
- [ ] **tha** (Thai) -- module + dictionary + tests; tonal
- [ ] **tgl** (Tagalog) -- module + dictionary + tests
- [ ] **tts** (Isan / Northeastern Thai) -- module + dictionary + tests; tonal
- [ ] **vie_c** (Vietnamese, Central) -- module + dictionary + tests; tonal
- [ ] **vie_n** (Vietnamese, Northern) -- module + dictionary + tests; tonal
- [ ] **vie_s** (Vietnamese, Southern) -- module + dictionary + tests; tonal

### Dravidian (1 missing)

- [ ] **tam** (Tamil) -- module + dictionary + tests; retroflex series

### Austronesian (1 missing)

- [ ] **mri** (Maori) -- module + dictionary + tests; small phoneme inventory

### Niger-Congo (1 missing)

- [ ] **swa** (Swahili) -- module + dictionary + tests

### Uyghur (1 missing)

- [ ] **uig** (Uyghur) -- module + dictionary + tests

### Constructed (3 missing)

- [ ] **epo** (Esperanto) -- module + dictionary + tests
- [ ] **ido** (Ido) -- module + dictionary + tests
- [ ] **ina** (Interlingua) -- module + dictionary + tests

### Language coverage notes

- Profile syllable length distributions across all target languages to set appropriate `max_phonemes` for fixed-width mode (current assumption: 6 slots; may not cover Georgian, Arabic, or Georgian clusters)
- Tonal languages (zho, yue, nan, vie, tha, bur, tts) need a decision on whether tone is encoded as a feature dimension or handled separately
- Verify duplicate codes: `ice` vs `isl`, `slo` vs `slv` -- may share dictionaries or need distinct modules
- Languages with non-Latin scripts (ara, amh, hin, urd, tam, jpn, kor, zho, etc.) may need additional Unicode handling in the tokenizer

---

## 2. Critical: Fix Existing Bugs

- [x] **Fix `phones_product.py` syntax error** -- malformed list comprehension at lines 51-60 causes import crash; rewrite as a proper loop or valid comprehension
- [x] **Fix `distance.py` stub** -- constructor ignores `b` parameter, hardcodes `eng-us`, and exposes no actual distance metric
- [x] **Fix `release.yaml` typo** -- `charisu` corrected to `charsiu` in the dictionary download URL
- [x] **Fix `eng_us.py` phoneme gaps** -- added `ʌ` and `ɚ` to `PHONEME_FEATURES`
- [x] **Fix `utils.py`** -- functions marked `TODO FIXME`; either implement or remove

---

## 3. Core: Implement Phonological Distance API

The library's stated purpose (phonological distance/similarity metrics) is now functional.

- [x] **Define distance/similarity interface** -- `Distance` class with `.hamming()`, `.edit_distance()`, `.normalised_edit_distance()`, `.pairwise_hamming()`, `.pairwise_edit_distance()`, plus `compare_cross_language()` for cross-language comparison
- [x] **Implement Hamming similarity for fixed-width bitarrays** -- normalised bit-match ratio between same-length syllable encodings
- [x] **Implement phoneme-level edit distance** -- DP alignment using per-phoneme feature cost (gradient substitution cost rather than binary match/mismatch)
- [x] **Implement `compare_word_across_languages`** -- `compare_cross_language()` function for pairwise phonological distance across multiple language encodings
- [ ] **Implement cross-language aggregation** -- word-level mean over syllables; corpus-level distance matrix for clustering

---

## 4. Cross-Language Feature Mapping

- [ ] **Adopt universal distinctive feature set** -- standardise on Panphon (~24 features) or PHOIBLE as the shared basis so bitarrays from different languages are directly comparable
- [ ] **Build feature encoder** -- `phoneme -> bitarray` via a unified feature table (replaces per-language ad-hoc feature dicts for comparison purposes)
- [ ] **Implement `IntersectingBitArraySpecification`** -- flesh out the existing 31-line stub to properly combine multiple language specs with aligned feature dimensions
- [ ] **Add syllable segmenter with onset/nucleus/coda boundaries** -- language-aware syllabification (onset-maximisation or rule-based) as a pre-encoding step

---

## 5. Cython Performance Layer

Tight numeric loops (DP edit distance, Hamming over corpus, batch comparison) are strong Cython candidates.

- [x] **Create `_core.pyx`** -- Cython module for hot-path numeric operations
  - Typed DP edit distance inner loop (`boundscheck(False)`, `wraparound(False)`, C-typed loop vars) -- 3.5x speedup measured
  - Batch Hamming similarity over corpus -- 2.3x speedup measured (bitarray XOR already C-accelerated)
- [ ] **Add `prange` parallel corpus comparison** -- OpenMP-parallelised pairwise distance matrix computation with `nogil`
- [x] **Create `_core.pxd`** -- public typed declarations for Cython module
- [x] **Update `pyproject.toml`** -- add Cython build dependency, configure extension modules, optional dependency groups
- [x] **Organise Cython/Python split** -- numeric loops in `_core.pyx`, linguistic logic (syllabification, feature lookup) stays in Python; `distance.py` auto-detects Cython and falls back to pure Python

---

## 6. ONNX Runtime C API Integration

Replace Python ONNX bindings with direct C API calls on the inference hot path.

- [ ] **Evaluate C API for G2P inference** -- benchmark current `optimum[onnxruntime]` Python path vs direct `onnxruntime_c_api.h` calls
- [ ] **Create Cython `ort_api.pxd`** -- `cdef extern from "onnxruntime_c_api.h"` with `nogil` annotations for GIL-free inference
- [ ] **Implement C API session management** -- `OrtSession` creation, input/output tensor buffer management with `CreateTensorWithDataAsOrtValue`
- [ ] **Wire Cython G2P inference** -- bypass Python bindings: Cython -> C API -> C++ runtime, eliminating GIL overhead on `session.run()`
- [ ] **Thread-safe concurrent inference** -- multiple `OrtSession` instances across native threads without GIL contention

---

## 7. Testing

- [x] **Un-comment `test_cognates.py`** -- rewritten against the new cross-language distance API with 6 tests (cognate pair distances, intra- vs cross-family comparison, symmetric distance matrix)
- [x] **Un-skip `test_generator_caching.py`** -- rewritten with 2 tests (pickle caching performance, lazy constructor verification)
- [x] **Add distance/similarity unit tests** -- Hamming similarity, edit distance, cross-language comparison with known cognate pairs (42 tests in `test_distance.py`)
- [x] **Add Cython extension tests** -- verify numeric parity between pure Python and Cython implementations (3 parity tests in `test_distance.py::TestCythonParity`)
- [ ] **Add benchmark tests** -- comparative timing for Python vs Cython vs C API paths

---

## 8. Rhyme Detection & Rhyme Distance

The library already decomposes IPA into onset/nucleus/coda via `ipa_to_syllable()`, but no rhyme-specific API exists. Rhyme is one of the most common and useful phonological relationships.

- [ ] **`rhyme_distance(ipa_a, ipa_b, spec, phoneme_features) -> float`** -- compare nucleus+coda of the final stressed syllable (or last syllable if stress unknown). Returns feature distance over just those segments. Perfect rhyme = 0.0, slant/near rhyme = low distance.
- [ ] **`rhyme_type(ipa_a, ipa_b) -> str`** -- classify as `"perfect"`, `"near"`, `"assonance"` (vowels only), `"consonance"` (codas only), `"eye"` (orthographic only), `"none"`
- [ ] **Cross-lingual rhyme search** -- `find_rhymes(source_word, target_lang, dictionary) -> List[(word, dist)]` — scan a foreign dictionary for words that rhyme with an English word
- [ ] **Multi-syllable rhyme (feminine/dactylic)** -- compare the last N syllables (nucleus+coda+onset of following syllable). "glorious" / "victorious" = feminine rhyme (2-syllable match)
- [ ] **Stress-aware rhyming** -- currently `clean_phones` strips `ˈ` and `ˌ`. Add `preserve_stress=True` option so rhyme detection can anchor on the stressed syllable, not just the final one
- [ ] **Rhyme scheme detection** -- given a list of line-ending words (e.g. from poetry), detect the pattern (ABAB, AABB, ABBA, etc.) using rhyme_distance

---

## 9. Spoonerisms & Phoneme-Cluster Swaps

A spoonerism swaps the **onsets** of two words: "crushing blow" → "blushing crow". Requires syllable-level decomposition that the library already partially supports.

- [ ] **Syllable onset extraction** -- `get_onset(ipa_word, spec) -> List[str]` — return the onset cluster of the first syllable
- [ ] **`spoonerize(word_a_ipa, word_b_ipa, spec) -> (new_a, new_b)`** -- swap onset clusters and return new IPA strings. E.g. `/kɹʌʃɪŋ/, /bloʊ/` → `/blʌʃɪŋ/, /kɹoʊ/`
- [ ] **Spoonerism dictionary search** -- given two words, swap onsets, then look up results in the dictionary. If both are real words → valid spoonerism. `find_spoonerisms(phrase, lang, dictionary) -> List`
- [ ] **Cross-lingual spoonerisms** -- swap onsets between an English and foreign word to see if results form words in either language
- [ ] **Phoneme metathesis (general)** -- beyond onset-swapping, detect/generate phoneme transpositions at any position: "ask" → "aks", "nuclear" → "nucular"

---

## 10. Malapropisms, Eggcorns & Mondegreens

- [ ] **`find_malapropisms(word, lang, dictionary, threshold=0.2) -> List`** -- return dictionary words within low edit distance (phonologically similar but different words). "for all intensive purposes" → "intents and purposes"
- [ ] **Eggcorn detection** -- like malapropisms but the replacement makes some semantic sense: "old-timers' disease" for "Alzheimer's disease". Generation mode + known-eggcorn database
- [ ] **Mondegreen / resegmentation** -- `find_mondegreens(phrase_ipa, lang, dictionary) -> List` — given a flat IPA string (no word boundaries), find alternative word-boundary placements that produce valid dictionary words. "'Scuse me while I kiss the sky" → "kiss this guy". DP over the flat IPA to find all segmentations into dictionary words with total distance below threshold
- [ ] **Cross-lingual mondegreens** -- the pun finder's "glue" step already does this implicitly; formalize as a reusable API

---

## 11. Alliteration, Assonance & Consonance

Fundamental poetic/rhetorical devices that operate on subsets of the phoneme.

- [ ] **`has_alliteration(words_ipa: List[str], spec) -> bool`** -- check if consecutive words share onset phoneme(s)
- [ ] **`alliteration_score(phrase_ipa, spec) -> float`** -- fraction of consecutive word-pairs sharing onset phonemes
- [ ] **`assonance_score(phrase_ipa, spec) -> float`** -- measure repeated vowel nucleus patterns across a phrase
- [ ] **`consonance_score(phrase_ipa, spec) -> float`** -- measure repeated consonant patterns (onset + coda)
- [ ] **Generate alliterative alternatives** -- given a phrase, suggest word replacements that increase alliteration while keeping phonological distance low
- [ ] **Tongue twister detection** -- identify phrases with high alliteration + minimal phoneme variation (hard to say fast)

---

## 12. Portmanteaus (Blended Words)

Portmanteaus merge two words at a phonological overlap: "breakfast" + "lunch" → "brunch" (overlap at /ʌn/).

- [ ] **`find_overlap(ipa_a, ipa_b, spec, min_overlap=2) -> List[Tuple]`** -- find positions where suffix of word A overlaps with prefix of word B (feature-tolerant matching)
- [ ] **`portmanteau(ipa_a, ipa_b, overlap) -> str`** -- generate blended IPA: prefix of A up to overlap, then suffix of B from overlap onward
- [ ] **Cross-lingual portmanteaus** -- find English-foreign word pairs with phonological overlap that blend into a pronounceable hybrid
- [ ] **Phonotactic validation** -- check if generated portmanteaus obey the target language's phonotactic constraints

---

## 13. Syllable & Prosodic Structure Improvements

Many features above require better syllable decomposition than the current naive vowel-nucleus scan.

- [ ] **Proper syllabification** -- implement the Maximum Onset Principle (consonants prefer onsets over codas) with language-specific phonotactic constraints. Current `ipa_to_syllable()` uses a simple heuristic
- [ ] **Stress preservation mode** -- add `preserve_stress=True` to `clean_phones` and distance functions. Critical for rhyme anchoring, spoonerism targeting, and prosodic matching
- [ ] **Syllable-level edit distance** -- compare syllable-structured `(onset, nucleus, coda)` tuples rather than flat phoneme sequences. Weight nucleus mismatches more heavily (more perceptually salient)
- [ ] **Syllable count & meter detection** -- `syllable_count(ipa, spec) -> int` and `stress_pattern(ipa, spec) -> str` (e.g. "01001" for iambic). Useful for poetry analysis and limerick detection
- [ ] **Prosodic tier** -- optional pitch/tone encoding for tonal languages (Mandarin, Vietnamese, Thai, etc.)

---

## 14. Phonemization Pipeline Improvements

The text→phoneme pipeline is: raw text → `clean_phones` (strip stress/length) → `ipa_tokenizer` (greedy longest-match) → phoneme list → feature vectors. Each stage has concrete improvement opportunities.

### 14a. Phonemization code logic

- [ ] **Unified G2P fallback chain** -- currently `CharsiuG2P.generate()` tries dict lookup then ONNX model, but the two paths return slightly different IPA conventions (e.g. dict may use /ɹ/ while model outputs /r/). Normalize outputs at the boundary so downstream distance is consistent
- [ ] **Multi-word G2P sandhi** -- when phonemizing a phrase like "a net", generate IPA for the phrase as a unit (capturing liaison/elision/assimilation) rather than concatenating per-word results. Relevant for French liaison ("les amis" → /le.z‿a.mi/) and English flapping
- [ ] **OOV phoneme handling** -- when `ipa_tokenizer` encounters a character not in the language's phoneme inventory, it currently silently drops it. Add a configurable strategy: `drop`, `pass_through`, or `nearest_phoneme` (map to closest feature-vector match in the inventory)
- [ ] **Language-specific normalization rules** -- `clean_phones` applies the same strip-set (`ˈˌːˑ‿`) to all languages. Some languages need different treatment: length is contrastive in Finnish/Japanese, tone diacritics are contrastive in Mandarin/Vietnamese. Add per-language `CleanConfig` with `strip_stress`, `strip_length`, `strip_tone` booleans

### 14b. Disk caching for pre-tokenized dictionaries  ✅ DONE

Implemented `PreTokenizedDictionary` class with numpy-backed storage (int16 token indices + int32 offsets). Cache files live at `~/.cache/phone_similarity/pretok_{lang}_{fingerprint}.v2.pkl`.

- [x] **Serialize `pretokenize_dictionary()` output** -- `PreTokenizedDictionary.save()` / `.load()` with numpy byte buffers (~0.1s load vs ~0.8s for pickle)
- [x] **Cache invalidation** -- `_pretokenize_cache_fingerprint()` hashes phoneme inventory + min_tokens + G2P pickle file stat (mtime+size). No raw dict loading needed for validation
- [x] **Lazy per-language loading** -- `cached_pretokenize_dictionary()` accepts a `dict_or_factory` callable; raw G2P dict only loaded on cache miss. `wikipedia_np_puns.py` uses `lambda _lc=lc: g2ps[_lc].pdict`
- [ ] **Memory-mapped feature matrices** -- store the pre-computed `np.ndarray` feature matrices as memory-mapped files (`np.memmap`) so multiple processes can share the same physical memory page

### 14c. Parallel dictionary scans  ✅ DONE

Implemented `parallel_dictionary_scan()` and `_scan_one_language()` in `distance.py`. Uses `ProcessPoolExecutor` with `spawn` context for fork safety. Integrated into `wikipedia_np_puns.py` (replaces the sequential triple-nested loop).

- [x] **`ProcessPoolExecutor` per-language fan-out** -- one worker per target language, collects results via `as_completed()`. `--workers N` CLI arg (default 1=sequential). 5 workers gives ~1.7x speedup on scan phase
- [ ] **Cython `prange` over dictionary entries** -- inside `batch_dictionary_scan`, the inner loop over dictionary entries can use OpenMP `prange` with `nogil`. Requires compiling with `-fopenmp` and marking the loop body as `nogil`-safe (already nearly there since feature matrices are C arrays)
- [ ] **Async streaming results** -- yield `(phrase, lang, word, distance)` tuples as they're found below threshold, rather than collecting all results then sorting. Enables real-time output in the pun finder and early termination when enough matches are found
- [ ] **GIL-free pipeline** -- chain tokenization → feature extraction → distance computation entirely in Cython/C without touching Python objects, allowing true multi-threaded parallelism for the full scan

### 14d. Cythonize the tokenizer  ✅ DONE

Implemented `cython_ipa_tokenizer()` and `batch_ipa_tokenize()` in `_core.pyx`. Uses frozenset-based O(1) phoneme lookup instead of O(n) linear scan.

- [x] **Port `ipa_tokenizer` to `_core.pyx`** -- `cython_ipa_tokenizer()` with frozenset lookup. `BaseBitArraySpecification.ipa_tokenizer` auto-dispatches to Cython when available. French 2.9x, German 8.8x, Spanish 2.3x speedup
- [ ] **Precompute a trie from the phoneme inventory** -- instead of linear scan through sorted phonemes at each position, build a trie (prefix tree) once per language. Trie lookup is O(max_phoneme_length) per position vs O(inventory_size) for linear scan. Implement in Cython with `cdef struct TrieNode`
- [ ] **Fused tokenize-and-featurize** -- combine tokenization and feature-vector lookup into a single Cython pass. Currently we tokenize to `List[str]`, then look up each phoneme in `PHONEME_FEATURES`. Fusing avoids the intermediate Python list and dict lookups
- [x] **Batch tokenization** -- `batch_ipa_tokenize()` processes an entire dictionary in one Cython call, minimizing Python<->C boundary crossings

### 14e. Phonetic embedding extraction  ✅ DONE

Brute-force edit distance against 600K dictionary entries per language is O(n·m·k) where n=phrases, m=dict_size, k=avg_phoneme_length. Dense embeddings enable approximate nearest-neighbor (ANN) search in O(n·log(m)).

- [x] **Phoneme-sequence → fixed-dim vector** -- encode each IPA sequence as a fixed-length dense vector by averaging per-phoneme feature vectors (already computed as bitarrays). This gives a 20-30 dim embedding per word for free
- [x] **Weighted positional encoding** -- weight phonemes by position (onset phonemes matter more for perception than coda phonemes). Use exponential decay or learned weights to produce position-aware embeddings
- [x] **Build ANN index per language** -- use `faiss` or `annoy` to index all dictionary embeddings. At query time, retrieve top-100 approximate neighbors, then re-rank with exact `feature_edit_distance`. Reduces per-query cost from 600K distance computations to ~100
- [ ] **Locality-sensitive hashing (LSH) alternative** -- for environments where faiss is too heavy, implement a simpler LSH scheme over the binary feature vectors. Each phoneme is already a bitarray; concatenated/hashed phoneme sequences can be bucketed for fast candidate retrieval
- [ ] **Incremental index updates** -- when a new language is loaded or the phoneme inventory changes, update the ANN index incrementally rather than rebuilding from scratch

### 14f. Beam search  ✅ DONE

Multi-word segmentation (the "glue" step in `wikipedia_np_puns.py`) currently uses greedy left-to-right matching. Beam search explores multiple segmentation hypotheses in parallel.

- [x] **Beam search for multi-word segmentation** -- given a target English phoneme sequence and a foreign dictionary, find the top-K segmentations into foreign words that minimize total distance. Maintain a beam of width B partial hypotheses `(consumed_phonemes, [words_so_far], cumulative_distance)`, expanding each by trying all dictionary entries that match the next unconsumed segment
- [x] **Length-normalized scoring** -- score hypotheses by `cumulative_distance / num_phonemes_consumed` to avoid bias toward shorter segmentations. Add a coverage bonus for hypotheses that consume more of the target sequence
- [x] **Pruning heuristics** -- prune hypotheses where `cumulative_distance / consumed > 2 × best_complete_score` (admissible bound). Also limit maximum words per segmentation (e.g. 4) to avoid degenerate splits into single-phoneme words
- [ ] **G2P beam decoding** -- the ONNX G2P model currently uses greedy argmax decoding. Implement beam search over the model's output logits to produce top-K pronunciation hypotheses per word, then pick the one that yields the best downstream pun match
- [ ] **Phrase-level beam search** -- combine per-word beam results: for an English NP with N words, maintain a beam over the Cartesian product of per-word top-K foreign matches, scored by total phrase distance + a fluency bonus for adjacent foreign words that form natural collocations
