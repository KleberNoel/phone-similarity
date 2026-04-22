#!/usr/bin/env python3
"""Wikipedia-scale pun mining experiment.

Collects random Wikipedia pages, extracts one dictionary-backed word per page,
maps words to IPA with Charsiu dictionaries, and computes cross-language
oronym-style nearest-neighbor similarity with reverse dictionary lookup.

The default run uses 1,000 pages total distributed across:
  - eng_us (English)
  - ger (German)
  - spa (Spanish)
  - por_po (Portuguese - Portugal)
  - jpn (Japanese)
  - ady (Adyghe; used as a proxy for Abkhaz-like syllable complexity)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from phone_similarity.clean_phones import clean_phones
from phone_similarity.dictionary_scan import reverse_dictionary_lookup
from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator
from phone_similarity.language import LANGUAGES
from phone_similarity.pretokenize import cached_pretokenize_dictionary


USER_AGENT = "phono-sim-research-bot/0.1 (https://github.com/KleberNoel/phone-similarity)"
WIKI_API_TEMPLATE = "https://{domain}.wikipedia.org/w/api.php"

GENERIC_TOKEN_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
JAPANESE_TOKEN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff々ー]+")


@dataclass(frozen=True)
class LanguageConfig:
    key: str
    charsiu_code: str
    wiki_domain: str
    label: str


LANGUAGES_FOR_EXPERIMENT: list[LanguageConfig] = [
    LanguageConfig("eng_us", "eng-us", "en", "English (US)"),
    LanguageConfig("ger", "ger", "de", "German"),
    LanguageConfig("spa", "spa", "es", "Spanish"),
    LanguageConfig("por_po", "por-po", "pt", "Portuguese (PT)"),
    LanguageConfig("jpn", "jpn", "ja", "Japanese"),
    LanguageConfig("ady", "ady", "ady", "Adyghe (proxy for Abkhaz-type complexity)"),
]


# Undirected pair definitions. We evaluate both directions for each.
PAIR_FAMILIES: list[tuple[str, str, str]] = [
    ("eng_us", "ger", "shared_history"),
    ("spa", "por_po", "shared_history"),
    ("eng_us", "jpn", "distant"),
    ("ger", "jpn", "distant"),
    ("ady", "jpn", "syllable_complexity_asymmetry"),
]


def _quota_split(total: int, n: int) -> list[int]:
    base = total // n
    rem = total % n
    return [base + (1 if i < rem else 0) for i in range(n)]


def _tokenize_text(lang_key: str, text: str) -> list[str]:
    if lang_key == "jpn":
        return JAPANESE_TOKEN_RE.findall(text)
    return GENERIC_TOKEN_RE.findall(text)


def _candidate_forms(token: str) -> list[str]:
    forms = [token, token.lower()]
    stripped = token.strip("'’`\"")
    forms.extend([stripped, stripped.lower()])
    out: list[str] = []
    seen: set[str] = set()
    for f in forms:
        if not f:
            continue
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def _first_dict_word_and_ipa(
    lang_key: str, text: str, lexicon: dict[str, str]
) -> tuple[str, str] | None:
    for tok in _tokenize_text(lang_key, text):
        for cand in _candidate_forms(tok):
            if len(cand) < 2 or len(cand) > 40:
                continue
            raw_ipa = lexicon.get(cand)
            if not raw_ipa:
                continue
            ipa = clean_phones(raw_ipa.split(",")[0].strip())
            if ipa:
                return cand, ipa
    return None


def _fetch_random_pages(session: requests.Session, wiki_domain: str, limit: int) -> list[dict]:
    resp = session.get(
        WIKI_API_TEMPLATE.format(domain=wiki_domain),
        params={
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": limit,
            "prop": "extracts",
            "explaintext": 1,
            "exintro": 1,
            "exchars": 900,
        },
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    return list(payload.get("query", {}).get("pages", {}).values())


def _collect_page_samples(
    session: requests.Session,
    cfg: LanguageConfig,
    lexicon: dict[str, str],
    target_pages: int,
    batch_size: int,
    max_batches: int,
) -> tuple[list[dict], int]:
    samples: list[dict] = []
    seen_page_ids: set[int] = set()
    batches = 0

    while len(samples) < target_pages and batches < max_batches:
        pages = _fetch_random_pages(session, cfg.wiki_domain, batch_size)
        batches += 1
        for p in pages:
            page_id = p.get("pageid")
            if isinstance(page_id, int) and page_id in seen_page_ids:
                continue
            if isinstance(page_id, int):
                seen_page_ids.add(page_id)

            title = p.get("title", "")
            extract = p.get("extract", "")
            text = f"{title}\n{extract}"
            picked = _first_dict_word_and_ipa(cfg.key, text, lexicon)
            if picked is None:
                continue

            word, ipa = picked
            samples.append(
                {
                    "page_id": page_id,
                    "title": title,
                    "word": word,
                    "ipa": ipa,
                }
            )
            if len(samples) >= target_pages:
                break
        time.sleep(0.05)

    return samples, batches


def _summarize_distances(distances: list[float]) -> dict[str, float | int]:
    if not distances:
        return {
            "n_scored": 0,
            "mean_distance": float("nan"),
            "median_distance": float("nan"),
            "mean_similarity": float("nan"),
            "hit_rate_le_0_25": 0.0,
            "hit_rate_le_0_35": 0.0,
            "hit_rate_le_0_50": 0.0,
        }

    sims = [1.0 - d for d in distances]
    n = len(distances)
    return {
        "n_scored": n,
        "mean_distance": statistics.fmean(distances),
        "median_distance": statistics.median(distances),
        "mean_similarity": statistics.fmean(sims),
        "hit_rate_le_0_25": sum(d <= 0.25 for d in distances) / n,
        "hit_rate_le_0_35": sum(d <= 0.35 for d in distances) / n,
        "hit_rate_le_0_50": sum(d <= 0.50 for d in distances) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wikipedia pun mining experiment")
    parser.add_argument("--total-pages", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-batches", type=int, default=120)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/wikipedia_pun_mining_1000_results.json"),
    )
    args = parser.parse_args()

    if args.total_pages < len(LANGUAGES_FOR_EXPERIMENT):
        raise ValueError("--total-pages must be >= number of languages")

    quotas = _quota_split(args.total_pages, len(LANGUAGES_FOR_EXPERIMENT))

    print("Preparing language artifacts...")
    artifacts: dict[str, dict] = {}
    for cfg in LANGUAGES_FOR_EXPERIMENT:
        lang_ns = LANGUAGES[cfg.key]
        g2p = CharsiuGraphemeToPhonemeGenerator(cfg.charsiu_code)
        lexicon = g2p.pdict
        spec = LANGUAGES.build_spec(cfg.key)
        ptd = cached_pretokenize_dictionary(
            lambda lex=lexicon: lex,
            spec,
            lang=cfg.key,
            min_tokens=1,
        )
        artifacts[cfg.key] = {
            "cfg": cfg,
            "lexicon": lexicon,
            "spec": spec,
            "features": lang_ns.PHONEME_FEATURES,
            "ptd": ptd,
        }
        print(f"  {cfg.key:7s} lexicon={len(lexicon):7d} pretokenized={len(ptd):7d}")

    print("\nCollecting random Wikipedia pages...")
    sampled: dict[str, list[dict]] = {}
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        for cfg, quota in zip(LANGUAGES_FOR_EXPERIMENT, quotas, strict=False):
            t0 = time.time()
            rows, batches = _collect_page_samples(
                session,
                cfg,
                artifacts[cfg.key]["lexicon"],
                quota,
                args.batch_size,
                args.max_batches,
            )
            sampled[cfg.key] = rows
            elapsed = time.time() - t0
            print(
                f"  {cfg.key:7s} target={quota:4d} collected={len(rows):4d} "
                f"batches={batches:3d} time={elapsed:6.1f}s"
            )

    print("\nRunning directed cross-language mining...")
    directed_results: list[dict] = []
    per_word_matches: list[dict] = []

    for a_key, b_key, family in PAIR_FAMILIES:
        for src_key, tgt_key in ((a_key, b_key), (b_key, a_key)):
            src_art = artifacts[src_key]
            tgt_art = artifacts[tgt_key]
            src_samples = sampled[src_key]
            distances: list[float] = []
            t0 = time.time()

            for row in src_samples:
                matches = reverse_dictionary_lookup(
                    row["ipa"],
                    src_key,
                    src_art["spec"],
                    src_art["features"],
                    tgt_key,
                    tgt_art["spec"],
                    tgt_art["features"],
                    tgt_art["lexicon"],
                    top_n=1,
                    max_distance=1.0,
                    pre_tokenized=tgt_art["ptd"],
                )
                if not matches:
                    continue
                best_word, best_ipa, dist = matches[0]
                distances.append(dist)
                per_word_matches.append(
                    {
                        "family": family,
                        "source_lang": src_key,
                        "target_lang": tgt_key,
                        "source_page_id": row["page_id"],
                        "source_title": row["title"],
                        "source_word": row["word"],
                        "source_ipa": row["ipa"],
                        "target_word": best_word,
                        "target_ipa": best_ipa,
                        "distance": dist,
                        "similarity": 1.0 - dist,
                    }
                )

            summary = _summarize_distances(distances)
            summary.update(
                {
                    "family": family,
                    "source_lang": src_key,
                    "target_lang": tgt_key,
                    "n_input_words": len(src_samples),
                    "elapsed_seconds": time.time() - t0,
                }
            )
            directed_results.append(summary)
            print(
                f"  {src_key:7s} -> {tgt_key:7s} "
                f"n={summary['n_scored']:4d}/{len(src_samples):4d} "
                f"mean_sim={summary['mean_similarity']:.3f} "
                f"mean_dist={summary['mean_distance']:.3f}"
            )

    # Build asymmetry summaries by family/pair.
    pair_lookup = {(r["family"], r["source_lang"], r["target_lang"]): r for r in directed_results}
    asymmetry: list[dict] = []
    for a_key, b_key, family in PAIR_FAMILIES:
        ab = pair_lookup[(family, a_key, b_key)]
        ba = pair_lookup[(family, b_key, a_key)]
        asymmetry.append(
            {
                "family": family,
                "lang_a": a_key,
                "lang_b": b_key,
                "mean_similarity_a_to_b": ab["mean_similarity"],
                "mean_similarity_b_to_a": ba["mean_similarity"],
                "delta_similarity_a_minus_b": ab["mean_similarity"] - ba["mean_similarity"],
                "mean_distance_a_to_b": ab["mean_distance"],
                "mean_distance_b_to_a": ba["mean_distance"],
                "delta_distance_a_minus_b": ab["mean_distance"] - ba["mean_distance"],
            }
        )

    payload = {
        "meta": {
            "total_pages_requested": args.total_pages,
            "language_keys": [cfg.key for cfg in LANGUAGES_FOR_EXPERIMENT],
            "quotas": {
                cfg.key: quota
                for cfg, quota in zip(LANGUAGES_FOR_EXPERIMENT, quotas, strict=False)
            },
            "notes": [
                "Abkhaz (abk) is not available in phone_similarity language registry.",
                "Adyghe (ady) is used as a proxy for Northwest Caucasian syllable complexity.",
            ],
        },
        "sampled_pages": sampled,
        "directed_results": directed_results,
        "asymmetry_results": asymmetry,
        "per_word_matches": per_word_matches,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
