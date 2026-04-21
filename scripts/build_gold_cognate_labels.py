#!/usr/bin/env python3
"""Build a small manual gold set for cognate-removal evaluation.

This script materializes a deterministic subset from the multilingual run with
manual labels for known cognate-like and chance-homophone examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GOLD_LABELS = {
    # Known cognate/variant-like pairs
    ("eng_uk", "ger", "district", "distrikt"): "cognate",
    ("spa", "por_po", "batalla", "batalha"): "cognate",
    ("spa", "por_po", "plaza", "praça"): "cognate",
    ("fra", "spa", "football", "fútbol"): "loanword",
    ("dut", "ger", "chocolade", "schokolade"): "cognate",
    ("fra", "ita", "mafias", "mafia"): "cognate",
    ("eng_us", "dut", "peak", "piek"): "cognate",
    # Named-entity transfer / transliteration-like
    ("ger", "jpn", "robinson", "ロービジョン"): "named_entity_transfer",
    ("ady", "jpn", "апэрэ", "ガバナー"): "named_entity_transfer",
    # Chance-homophone examples (kept)
    ("ger", "eng_us", "hans", "hance"): "chance_homophone",
    ("eng_uk", "ger", "feast", "wiest"): "chance_homophone",
    ("dut", "eng_uk", "shit", "sheet"): "chance_homophone",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gold cognate labels")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("notebooks/wikipedia_pun_mining_multilingual_500each.json"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("notebooks/pun_cognate_gold_labels.jsonl"),
    )
    args = parser.parse_args()

    data = json.loads(args.input_json.read_text(encoding="utf-8"))
    rows = data["per_word_matches"]

    selected = []
    needed = set(GOLD_LABELS.keys())
    for row in rows:
        key = (
            row["source_lang"],
            row["target_lang"],
            row["source_word"].casefold(),
            row["target_word"].casefold(),
        )
        if key in needed:
            selected.append(
                {
                    **row,
                    "gold_label": GOLD_LABELS[key],
                }
            )
            needed.remove(key)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Gold rows written: {len(selected)}")
    if needed:
        print("Missing keys from source data:")
        for k in sorted(needed):
            print("  ", k)


if __name__ == "__main__":
    main()
