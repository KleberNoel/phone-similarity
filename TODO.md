# TODO - phone-similarity

1. Consider whether the 24 length float array for coarticulation is needed (bitarray may be better unless written using C / Cython)
2. Add the charsiu small model too (its performance is better)
3. Verify all languages, especially harder ones e.g. finnish, georgian, tonal languages.
4. CHECK ALL FIXME/ TODO notes remaining in code, and complete
5. Ensure cythonization is systematic and useful
6. update readme, add espeakng backend, and add install support ... e.g. ':pip install phone-similarity[g2p] # adds transformers + ONNX Runtime'
7. consider adding dictionary index e.g. FAISS, however, also with new words (added by g2p model)

### Language coverage notes

- Profile syllable length distributions across all target languages to set appropriate `max_phonemes` for fixed-width mode (current assumption: 6 slots; may not cover Georgian, Arabic, or Georgian clusters)
- Tonal languages (zho, yue, nan, vie, tha, bur, tts) need a decision on whether tone is encoded as a feature dimension or handled separately
- Verify duplicate codes: `ice` vs `isl`, `slo` vs `slv` -- may share dictionaries or need distinct modules
- Languages with non-Latin scripts (ara, amh, hin, urd, tam, jpn, kor, zho, etc.) may need additional Unicode handling in the tokenizer

- [ ] **Implement cross-language aggregation** -- word-level mean over syllables; corpus-level distance matrix for clustering

Replace Python ONNX bindings with direct C API calls on the inference hot path.

- [ ] **Evaluate C API for G2P inference** -- benchmark current `optimum[onnxruntime]` Python path vs direct `onnxruntime_c_api.h` calls
- [ ] **Create Cython `ort_api.pxd`** -- `cdef extern from "onnxruntime_c_api.h"` with `nogil` annotations for GIL-free inference
- [ ] **Implement C API session management** -- `OrtSession` creation, input/output tensor buffer management with `CreateTensorWithDataAsOrtValue`
- [ ] **Wire Cython G2P inference** -- bypass Python bindings: Cython -> C API -> C++ runtime, eliminating GIL overhead on `session.run()`
- [ ] **Thread-safe concurrent inference** -- multiple `OrtSession` instances across native threads without GIL contention

- [ ] **Add benchmark tests** -- comparative timing for Python vs Cython vs C API paths

- [ ] **`rhyme_distance(ipa_a, ipa_b, spec, phoneme_features) -> float`** -- compare nucleus+coda of the final stressed syllable (or last syllable if stress unknown). Returns feature distance over just those segments. Perfect rhyme = 0.0, slant/near rhyme = low distance.
- [ ] **`rhyme_type(ipa_a, ipa_b) -> str`** -- classify as `"perfect"`, `"near"`, `"assonance"` (vowels only), `"consonance"` (codas only), `"eye"` (orthographic only), `"none"`
- [ ] **Cross-lingual rhyme search** -- `find_rhymes(source_word, target_lang, dictionary) -> List[(word, dist)]` — scan a foreign dictionary for words that rhyme with an English word
- [ ] **Multi-syllable rhyme (feminine/dactylic)** -- compare the last N syllables (nucleus+coda+onset of following syllable). "glorious" / "victorious" = feminine rhyme (2-syllable match)
- [ ] **Stress-aware rhyming** -- currently `clean_phones` strips `ˈ` and `ˌ`. Add `preserve_stress=True` option so rhyme detection can anchor on the stressed syllable, not just the final one
- [ ] **Rhyme scheme detection** -- given a list of line-ending words (e.g. from poetry), detect the pattern (ABAB, AABB, ABBA, etc.) using rhyme_distance

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
- [ ] **Syllable-level edit distance** -- compare syllable-structured `(onset, nucleus, coda)` tuples rather than flat phoneme sequences. Weight nucleus mismatches more heavily (more perceptually salient)
- [ ] **Prosodic tier** -- optional pitch/tone encoding for tonal languages (Mandarin, Vietnamese, Thai, etc.)

---

## 14. Phonemization Pipeline Improvements

The text→phoneme pipeline is: raw text → `clean_phones` (strip stress/length) → `ipa_tokenizer` (greedy longest-match) → phoneme list → feature vectors. Each stage has concrete improvement opportunities.

### 14a. Phonemization code logic

- [ ] **Unified G2P fallback chain** -- currently `CharsiuG2P.generate()` tries dict lookup then ONNX model, but the two paths return slightly different IPA conventions (e.g. dict may use /ɹ/ while model outputs /r/). Normalize outputs at the boundary so downstream distance is consistent
- [ ] **Multi-word G2P sandhi** -- when phonemizing a phrase like "a net", generate IPA for the phrase as a unit (capturing liaison/elision/assimilation) rather than concatenating per-word results. Relevant for French liaison ("les amis" → /le.z‿a.mi/) and English flapping
- [ ] **OOV phoneme handling** -- when `ipa_tokenizer` encounters a character not in the language's phoneme inventory, it currently silently drops it. Add a configurable strategy: `drop`, `pass_through`, or `nearest_phoneme` (map to closest feature-vector match in the inventory)

### 14b. Disk caching for pre-tokenized dictionaries ✅ DONE

Implemented `PreTokenizedDictionary` class with numpy-backed storage (int16 token indices + int32 offsets). Cache files live at `~/.cache/phone_similarity/pretok_{lang}_{fingerprint}.v2.pkl`.

- [ ] **Memory-mapped feature matrices** -- store the pre-computed `np.ndarray` feature matrices as memory-mapped files (`np.memmap`) so multiple processes can share the same physical memory page

### 14c. Parallel dictionary scans ✅ DONE

Implemented `parallel_dictionary_scan()` and `_scan_one_language()` in `distance.py`. Uses `ProcessPoolExecutor` with `spawn` context for fork safety. Integrated into `wikipedia_np_puns.py` (replaces the sequential triple-nested loop).

- [ ] **Cython `prange` over dictionary entries** -- inside `batch_dictionary_scan`, the inner loop over dictionary entries can use OpenMP `prange` with `nogil`. Requires compiling with `-fopenmp` and marking the loop body as `nogil`-safe (already nearly there since feature matrices are C arrays)
- [ ] **Async streaming results** -- yield `(phrase, lang, word, distance)` tuples as they're found below threshold, rather than collecting all results then sorting. Enables real-time output in the pun finder and early termination when enough matches are found
- [ ] **GIL-free pipeline** -- chain tokenization → feature extraction → distance computation entirely in Cython/C without touching Python objects, allowing true multi-threaded parallelism for the full scan

### 14d. Cythonize the tokenizer ✅ DONE

Implemented `cython_ipa_tokenizer()` and `batch_ipa_tokenize()` in `_core.pyx`. Uses frozenset-based O(1) phoneme lookup instead of O(n) linear scan.

- [ ] **Precompute a trie from the phoneme inventory** -- instead of linear scan through sorted phonemes at each position, build a trie (prefix tree) once per language. Trie lookup is O(max_phoneme_length) per position vs O(inventory_size) for linear scan. Implement in Cython with `cdef struct TrieNode`
- [ ] **Fused tokenize-and-featurize** -- combine tokenization and feature-vector lookup into a single Cython pass. Currently we tokenize to `List[str]`, then look up each phoneme in `PHONEME_FEATURES`. Fusing avoids the intermediate Python list and dict lookups

### 14e. Phonetic embedding extraction ✅ DONE

Brute-force edit distance against 600K dictionary entries per language is O(n·m·k) where n=phrases, m=dict_size, k=avg_phoneme_length. Dense embeddings enable approximate nearest-neighbor (ANN) search in O(n·log(m)).

- [ ] **Locality-sensitive hashing (LSH) alternative** -- for environments where faiss is too heavy, implement a simpler LSH scheme over the binary feature vectors. Each phoneme is already a bitarray; concatenated/hashed phoneme sequences can be bucketed for fast candidate retrieval
- [ ] **Incremental index updates** -- when a new language is loaded or the phoneme inventory changes, update the ANN index incrementally rather than rebuilding from scratch

### 14f. Beam search ✅ DONE

Multi-word segmentation (the "glue" step in `wikipedia_np_puns.py`) currently uses greedy left-to-right matching. Beam search explores multiple segmentation hypotheses in parallel.

- [ ] **G2P beam decoding** -- the ONNX G2P model currently uses greedy argmax decoding. Implement beam search over the model's output logits to produce top-K pronunciation hypotheses per word, then pick the one that yields the best downstream pun match
- [ ] **Phrase-level beam search** -- combine per-word beam results: for an English NP with N words, maintain a beam over the Cartesian product of per-word top-K foreign matches, scored by total phrase distance + a fluency bonus for adjacent foreign words that form natural collocations
