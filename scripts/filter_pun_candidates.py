#!/usr/bin/env python3
"""Filter pun-mining matches to remove likely cognates and artifacts.

This script consumes the JSON produced by ``scripts/wikipedia_pun_mining.py``
and applies a conservative, explainable filtering pipeline:

1) drop low-information tokens (very short words)
2) drop hub artifacts (same target repeated many times)
3) drop likely cognates/orthographic variants
4) optionally emit an LLM-review queue for ambiguous survivors

The goal is to keep non-trivial, cross-lexical pun-like candidates while
removing obvious genealogical overlap (cognates) and retrieval artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def _normalize_letters_only(text: str) -> str:
    """Casefold + strip accents + retain alphabetic characters only."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return "".join(ch for ch in stripped if ch.isalpha())


def _script_tag(text: str) -> str:
    """Coarse script detection for filtering heuristics."""
    has_latin = False
    has_cyrillic = False
    has_kana = False
    has_han = False

    for ch in text:
        cp = ord(ch)
        if 0x0041 <= cp <= 0x024F:
            has_latin = True
        elif 0x0400 <= cp <= 0x052F:
            has_cyrillic = True
        elif 0x3040 <= cp <= 0x30FF:
            has_kana = True
        elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            has_han = True

    tags = [
        ("latin", has_latin),
        ("cyrillic", has_cyrillic),
        ("kana", has_kana),
        ("han", has_han),
    ]
    active = [name for name, flag in tags if flag]
    if not active:
        return "other"
    if len(active) == 1:
        return active[0]
    return "+".join(active)


def _looks_named_entity(word: str) -> bool:
    """Approximate named-entity heuristic from orthography.

    We intentionally keep this conservative: title-cased tokens with at least
    four letters are treated as possible names and are filtered by default in
    pun mining, because they often become transliteration/name-transfer artifacts.
    """
    core = word.strip("'’`\"")
    alpha = "".join(ch for ch in core if ch.isalpha())
    if len(alpha) < 4:
        return False
    # Single-token title case, e.g. "Robinson", "Martín".
    if alpha[:1].isupper() and alpha[1:].islower():
        return True
    # Acronym-like mixed-case entities.
    if re.search(r"[A-Z].*[A-Z]", core):
        return True
    return False


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance for short strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def _orthographic_similarity(a_norm: str, b_norm: str) -> float:
    if not a_norm and not b_norm:
        return 1.0
    denom = max(len(a_norm), len(b_norm), 1)
    return 1.0 - (_levenshtein(a_norm, b_norm) / denom)


def _pair_key(row: dict) -> tuple[str, str]:
    return (row["source_lang"], row["target_lang"])


def _is_shared_history_pair(row: dict) -> bool:
    return row.get("family") == "shared_history"


def _drop_reasons(
    row: dict,
    target_hub_count: int,
    min_word_len: int,
    hub_threshold: int,
    max_distance: float,
) -> tuple[list[str], dict]:
    """Return drop reasons and computed diagnostics."""
    source_word = row["source_word"]
    target_word = row["target_word"]
    dist = float(row["distance"])

    src_norm = _normalize_letters_only(source_word)
    tgt_norm = _normalize_letters_only(target_word)
    ortho_sim = _orthographic_similarity(src_norm, tgt_norm)
    src_script = _script_tag(source_word)
    tgt_script = _script_tag(target_word)
    same_script = src_script == tgt_script

    reasons: list[str] = []

    # 1) low-information tokens
    if len(src_norm) < min_word_len or len(tgt_norm) < min_word_len:
        reasons.append("short_token")

    # 2) target hub artifacts
    if target_hub_count >= hub_threshold:
        reasons.append("target_hub")

    # 3) low-quality phonological match
    if dist > max_distance:
        reasons.append("distance_too_high")

    # 4) likely cognates / orthographic variants
    if src_norm and tgt_norm:
        if src_norm == tgt_norm:
            reasons.append("exact_orthographic_match")
        elif same_script and src_script in {"latin", "cyrillic"}:
            # Strong near-spelling overlap with close phonology.
            if ortho_sim >= 0.80 and dist <= 0.30 and min(len(src_norm), len(tgt_norm)) >= 4:
                reasons.append("likely_cognate_or_variant")
            # Shared-history pairs: slightly looser threshold.
            elif (
                _is_shared_history_pair(row)
                and ortho_sim >= 0.65
                and dist <= 0.35
                and min(len(src_norm), len(tgt_norm)) >= 4
            ):
                reasons.append("shared_history_cognate_like")

    # 5) likely named-entity transfer artifacts
    if _looks_named_entity(source_word) or _looks_named_entity(target_word):
        reasons.append("named_entity_like")

    diagnostics = {
        "src_norm": src_norm,
        "tgt_norm": tgt_norm,
        "orthographic_similarity": ortho_sim,
        "src_script": src_script,
        "tgt_script": tgt_script,
        "same_script": same_script,
        "target_hub_count": target_hub_count,
    }
    return reasons, diagnostics


def _needs_llm_review(row: dict, diagnostics: dict) -> bool:
    """Mark ambiguous survivors for optional LLM adjudication.

    These are cases heuristics cannot confidently reject:
    - moderate orthographic overlap with good phonology
    - cross-script very-close matches (possible transliteration loans)
    """
    sim = diagnostics["orthographic_similarity"]
    dist = float(row["distance"])
    same_script = bool(diagnostics["same_script"])
    src_script = diagnostics["src_script"]
    tgt_script = diagnostics["tgt_script"]

    if same_script and src_script in {"latin", "cyrillic"} and 0.45 <= sim < 0.80 and dist <= 0.28:
        return True
    if src_script != tgt_script and dist <= 0.18:
        return True
    return False


def _llm_prompt(row: dict, diagnostics: dict) -> str:
    return (
        "Classify this cross-language phonological pair as one of: "
        "cognate, loanword, named_entity_transfer, chance_pun, unknown. "
        'Return strict JSON: {"label":...,"confidence":0..1,"reason":...}.\n\n'
        f"family={row['family']}\n"
        f"source_lang={row['source_lang']} target_lang={row['target_lang']}\n"
        f"source_word={row['source_word']} source_ipa={row['source_ipa']}\n"
        f"target_word={row['target_word']} target_ipa={row['target_ipa']}\n"
        f"distance={row['distance']:.6f} similarity={row['similarity']:.6f}\n"
        f"orthographic_similarity={diagnostics['orthographic_similarity']:.6f}\n"
        f"scripts={diagnostics['src_script']}->{diagnostics['tgt_script']}\n"
        f"source_title={row['source_title']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter cognates/artefacts from pun-mining output"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("notebooks/wikipedia_pun_mining_1000_results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/wikipedia_pun_mining_1000_filtered.json"),
    )
    parser.add_argument(
        "--llm-queue-output",
        type=Path,
        default=Path("notebooks/wikipedia_pun_mining_1000_llm_queue.jsonl"),
    )
    parser.add_argument(
        "--noncognate-output-jsonl",
        type=Path,
        default=Path("notebooks/pun_candidates_noncognate.jsonl"),
    )
    parser.add_argument(
        "--removed-output-jsonl",
        type=Path,
        default=Path("notebooks/pun_candidates_removed_cognates.jsonl"),
    )
    parser.add_argument("--min-word-len", type=int, default=4)
    parser.add_argument("--hub-threshold", type=int, default=4)
    parser.add_argument("--max-distance", type=float, default=0.50)
    parser.add_argument("--max-llm-queue", type=int, default=500)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data["per_word_matches"]

    target_hubs: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        target_hubs[_pair_key(row)][row["target_word"].casefold()] += 1

    kept: list[dict] = []
    dropped: list[dict] = []
    reason_counts: Counter[str] = Counter()
    llm_queue: list[dict] = []

    for row in rows:
        pair = _pair_key(row)
        hub_count = target_hubs[pair][row["target_word"].casefold()]
        reasons, diagnostics = _drop_reasons(
            row,
            hub_count,
            min_word_len=args.min_word_len,
            hub_threshold=args.hub_threshold,
            max_distance=args.max_distance,
        )

        enriched = {
            **row,
            **diagnostics,
        }

        if reasons:
            enriched["drop_reasons"] = reasons
            dropped.append(enriched)
            for reason in reasons:
                reason_counts[reason] += 1
            continue

        enriched["drop_reasons"] = []
        enriched["needs_llm_review"] = _needs_llm_review(row, diagnostics)
        kept.append(enriched)

        if enriched["needs_llm_review"] and len(llm_queue) < args.max_llm_queue:
            llm_queue.append(
                {
                    "id": f"{row['source_lang']}->{row['target_lang']}::{row['source_page_id']}::{row['source_word']}::{row['target_word']}",
                    "source_lang": row["source_lang"],
                    "target_lang": row["target_lang"],
                    "source_word": row["source_word"],
                    "target_word": row["target_word"],
                    "distance": row["distance"],
                    "similarity": row["similarity"],
                    "prompt": _llm_prompt(row, diagnostics),
                }
            )

    kept_distances = [float(r["distance"]) for r in kept]
    mean_distance = sum(kept_distances) / len(kept_distances) if kept_distances else float("nan")

    kept_by_pair: dict[str, int] = Counter(f"{r['source_lang']}->{r['target_lang']}" for r in kept)

    output = {
        "meta": {
            "input_path": str(args.input),
            "min_word_len": args.min_word_len,
            "hub_threshold": args.hub_threshold,
            "max_distance": args.max_distance,
            "total_rows": len(rows),
            "kept_rows": len(kept),
            "dropped_rows": len(dropped),
            "kept_mean_distance": mean_distance,
            "kept_mean_similarity": (1.0 - mean_distance) if kept_distances else float("nan"),
        },
        "reason_counts": dict(reason_counts),
        "kept_by_pair": dict(kept_by_pair),
        "kept_candidates": kept,
        "dropped_candidates": dropped,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    args.noncognate_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.noncognate_output_jsonl.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    args.removed_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.removed_output_jsonl.open("w", encoding="utf-8") as f:
        for row in dropped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if llm_queue:
        args.llm_queue_output.parent.mkdir(parents=True, exist_ok=True)
        with args.llm_queue_output.open("w", encoding="utf-8") as f:
            for rec in llm_queue:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Input rows: {len(rows)}")
    print(f"Kept rows: {len(kept)}")
    print(f"Dropped rows: {len(dropped)}")
    print(f"Reason counts: {dict(reason_counts)}")
    print(f"Saved filtered output to {args.output}")
    print(f"Saved non-cognate JSONL to {args.noncognate_output_jsonl}")
    print(f"Saved removed-candidates JSONL to {args.removed_output_jsonl}")
    if llm_queue:
        print(f"Saved LLM review queue ({len(llm_queue)}) to {args.llm_queue_output}")


if __name__ == "__main__":
    main()
