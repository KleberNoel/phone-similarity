#!/usr/bin/env python3
"""Evaluate cognate-removal precision/recall against a small gold set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REMOVE_LABELS = {"cognate", "loanword", "named_entity_transfer"}


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _key(row: dict) -> tuple[str, str, str, str]:
    return (
        row["source_lang"],
        row["target_lang"],
        row["source_word"].casefold(),
        row["target_word"].casefold(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cognate-removal quality")
    parser.add_argument(
        "--gold-jsonl",
        type=Path,
        default=Path("notebooks/pun_cognate_gold_labels.jsonl"),
    )
    parser.add_argument(
        "--removed-jsonl",
        type=Path,
        default=Path("notebooks/pun_candidates_removed_cognates_integrated.jsonl"),
    )
    parser.add_argument(
        "--kept-jsonl",
        type=Path,
        default=Path("notebooks/pun_candidates_noncognate_integrated.jsonl"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("notebooks/cognate_filter_eval_report.json"),
    )
    args = parser.parse_args()

    gold = _load_jsonl(args.gold_jsonl)
    removed = {_key(r): r for r in _load_jsonl(args.removed_jsonl)}
    kept = {_key(r): r for r in _load_jsonl(args.kept_jsonl)}

    tp = fp = tn = fn = 0
    false_positive_examples = []
    false_negative_examples = []

    for row in gold:
        k = _key(row)
        gold_remove = row["gold_label"] in REMOVE_LABELS
        pred_remove = k in removed

        if gold_remove and pred_remove:
            tp += 1
        elif not gold_remove and pred_remove:
            fp += 1
            false_positive_examples.append(row)
        elif not gold_remove and not pred_remove:
            tn += 1
        else:
            fn += 1
            false_negative_examples.append(row)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    report = {
        "n_gold": len(gold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision_remove": precision,
        "recall_remove": recall,
        "f1_remove": f1,
        "false_positive_examples": false_positive_examples,
        "false_negative_examples": false_negative_examples,
        "note": "Small seed gold set; expand for robust metrics.",
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Gold rows:", len(gold))
    print("TP/FP/TN/FN:", tp, fp, tn, fn)
    print(f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")
    print("Saved report:", args.report_json)


if __name__ == "__main__":
    main()
