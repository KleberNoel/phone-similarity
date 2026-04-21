# TODO: Remove Cognates from Pun Mining

## Manual Check (current extreme list)

- Most ultra-low-distance pairs are not useful puns; they are mostly cognates, inflections, or transliteration/name transfer.
- High-frequency examples that should be removed:
  - `district ~ distrikt`
  - `chocolade ~ schokolade`
  - `batalla ~ batalha`
  - `football ~ futbol`
  - `mafias ~ mafia`
  - `plaza ~ praca`
  - `martin ~ martir` (often name-family noise)
- Cross-script low-distance often includes transliterated names/loanwords, not true pun discovery.

## Deliverables

- [x] Produce `notebooks/pun_candidates_noncognate.jsonl` (high-confidence non-cognate set).
- [x] Produce `notebooks/pun_candidates_removed_cognates.jsonl` (with reason codes).
- [x] Add `scripts/filter_cognates_llm.py` for LLM adjudication.

## Pipeline Plan

### 1) Strong heuristic pre-filter (before LLM)

- [x] Remove short tokens (`len(normalized) < 4`) unless manually allowlisted.
- [x] Remove exact normalized matches (NFKD + casefold).
- [x] Remove near-spelling matches by language family threshold (Levenshtein similarity).
- [x] Remove repeated hub targets per directed pair (artifact suppression).
- [x] Remove obvious named entities with title-case/person/place hints.

### 2) LLM cognate adjudication schema

- [x] Use strict JSON output schema:
  - `label`: one of `cognate`, `loanword`, `named_entity_transfer`, `chance_homophone`, `unknown`
  - `confidence`: float in `[0,1]`
  - `reason`: short rationale
- [x] Add second field `remove_from_pun_set` (`true` for first three labels).
- [x] Include word pair, IPA pair, language pair, source title, and orthographic similarity in prompt.

### 3) HF/vLLM inference options

- [x] Implement Hugging Face Inference route (batched API calls, retry/backoff).
- [x] Implement local vLLM route (same prompt, same JSON schema).
- [x] Keep model choice configurable via CLI:
  - `--backend hf`
  - `--backend vllm`
  - `--model <name>`

### 4) Decision policy

- [x] Auto-remove if `label in {cognate, loanword, named_entity_transfer}` AND `confidence >= 0.70`.
- [x] Keep if `chance_homophone` AND `confidence >= 0.60`.
- [x] Send uncertain rows to manual review queue.
- [x] For high-impact pairs, run two-pass voting (prompt variants) and require agreement.

### 5) Quality checks

- [x] Build a small gold set with manual labels.
- [x] Report precision/recall for "remove cognate" decision.
- [x] Track false positives where real puns were removed.
- [x] Add regression test so core known-cognate pairs are always filtered.

## Integration

- [x] Wire LLM adjudication into `scripts/filter_pun_candidates.py` as optional stage.
- [x] Re-run multilingual 500-page experiment and publish:
  - pre-filter counts
  - post-LLM counts
  - top non-cognate extreme pairs

## Current status

- Stage 1 complete: heuristic pre-filter + required JSONL outputs.
- Stage 2 complete: standalone LLM adjudication script with `hf`/`vllm`/`mock` backends.
- Stage 3 complete: integrated optional LLM stage, unit tests, and quality report.
