#!/usr/bin/env python3
"""Wikipedia-scale multilingual pun mining.

This variant supports:
- fixed pages-per-language sampling
- custom language selection
- all-pairs or explicit pair subset evaluation

It reuses the core methodology from ``wikipedia_pun_mining.py``:
random page sampling -> dictionary-backed word extraction -> IPA lookup ->
cross-language nearest-neighbor distance via reverse dictionary lookup.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import unicodedata
from dataclasses import dataclass
from itertools import combinations
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
    macro_family: str


AVAILABLE_LANGUAGES: dict[str, LanguageConfig] = {
    # Requested core set
    "eng_uk": LanguageConfig("eng_uk", "eng-uk", "en", "English (UK)", "germanic"),
    "ger": LanguageConfig("ger", "ger", "de", "German", "germanic"),
    "dut": LanguageConfig("dut", "dut", "nl", "Dutch", "germanic"),
    "fra": LanguageConfig("fra", "fra", "fr", "French", "romance"),
    "por_po": LanguageConfig("por_po", "por-po", "pt", "Portuguese (PT)", "romance"),
    "spa": LanguageConfig("spa", "spa", "es", "Spanish", "romance"),
    "ita": LanguageConfig("ita", "ita", "it", "Italian", "romance"),
    # Optional extras for extension
    "eng_us": LanguageConfig("eng_us", "eng-us", "en", "English (US)", "germanic"),
    "jpn": LanguageConfig("jpn", "jpn", "ja", "Japanese", "japonic"),
    "ady": LanguageConfig("ady", "ady", "ady", "Adyghe", "northwest_caucasian"),
}


DEFAULT_LANGUAGE_KEYS = ["eng_uk", "ger", "dut", "fra", "por_po", "spa", "ita"]


def _tokenize_text(lang_key: str, text: str) -> list[str]:
    if lang_key == "jpn":
        return JAPANESE_TOKEN_RE.findall(text)
    return GENERIC_TOKEN_RE.findall(text)


def _candidate_forms(token: str) -> list[str]:
    forms = [token, token.casefold()]
    stripped = token.strip("'’`\"")
    forms.extend([stripped, stripped.casefold()])

    # NFKD-normalized variant can help for dictionary lookups.
    norm = unicodedata.normalize("NFKD", stripped)
    forms.extend([norm, norm.casefold()])

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


def _classify_pair_family(cfg_a: LanguageConfig, cfg_b: LanguageConfig) -> str:
    return "shared_history" if cfg_a.macro_family == cfg_b.macro_family else "cross_family"


def _parse_pair_specs(pair_specs: str, language_keys: list[str]) -> list[tuple[str, str]]:
    """Parse --pairs entries like 'eng_uk:ger,ger:dut'"""
    allowed = set(language_keys)
    out: list[tuple[str, str]] = []
    for raw in pair_specs.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair spec {raw!r}; expected format langA:langB")
        a, b = parts[0].strip(), parts[1].strip()
        if a not in allowed or b not in allowed:
            raise ValueError(f"Pair {raw!r} uses unknown language(s)")
        if a == b:
            continue
        out.append((a, b))
    # Deduplicate undirected by sorted tuple
    seen: set[tuple[str, str]] = set()
    uniq: list[tuple[str, str]] = []
    for a, b in out:
        k = tuple(sorted((a, b)))
        if k in seen:
            continue
        seen.add(k)
        uniq.append((k[0], k[1]))
    return uniq


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multilingual Wikipedia pun-mining experiment"
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=",".join(DEFAULT_LANGUAGE_KEYS),
        help="Comma-separated language keys",
    )
    parser.add_argument(
        "--pages-per-language",
        type=int,
        default=500,
        help="Target number of sampled pages per language",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="",
        help="Optional comma-separated pair list, e.g. eng_uk:ger,ger:dut",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-batches", type=int, default=220)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/wikipedia_pun_mining_multilingual_500each.json"),
    )
    args = parser.parse_args()

    language_keys = [k.strip() for k in args.languages.split(",") if k.strip()]
    if not language_keys:
        raise ValueError("No languages selected")

    selected_cfgs: list[LanguageConfig] = []
    for key in language_keys:
        cfg = AVAILABLE_LANGUAGES.get(key)
        if cfg is None:
            raise ValueError(f"Unsupported language key {key!r}")
        if key not in LANGUAGES:
            raise ValueError(f"Language key {key!r} not present in phone_similarity LANGUAGES")
        selected_cfgs.append(cfg)

    if args.pairs.strip():
        pair_list = _parse_pair_specs(args.pairs, language_keys)
    else:
        pair_list = [(a, b) for a, b in combinations(language_keys, 2)]

    print("Preparing language artifacts...")
    artifacts: dict[str, dict] = {}
    for cfg in selected_cfgs:
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
        for cfg in selected_cfgs:
            t0 = time.time()
            rows, batches = _collect_page_samples(
                session,
                cfg,
                artifacts[cfg.key]["lexicon"],
                args.pages_per_language,
                args.batch_size,
                args.max_batches,
            )
            sampled[cfg.key] = rows
            elapsed = time.time() - t0
            print(
                f"  {cfg.key:7s} target={args.pages_per_language:4d} collected={len(rows):4d} "
                f"batches={batches:3d} time={elapsed:6.1f}s"
            )

    print("\nRunning directed cross-language mining...")
    directed_results: list[dict] = []
    per_word_matches: list[dict] = []

    for a_key, b_key in pair_list:
        cfg_a = artifacts[a_key]["cfg"]
        cfg_b = artifacts[b_key]["cfg"]
        family = _classify_pair_family(cfg_a, cfg_b)

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
                f"family={family:14s} "
                f"n={summary['n_scored']:4d}/{len(src_samples):4d} "
                f"mean_sim={summary['mean_similarity']:.3f} "
                f"mean_dist={summary['mean_distance']:.3f}"
            )

    pair_lookup = {(r["source_lang"], r["target_lang"]): r for r in directed_results}
    asymmetry: list[dict] = []
    for a_key, b_key in pair_list:
        ab = pair_lookup[(a_key, b_key)]
        ba = pair_lookup[(b_key, a_key)]
        asymmetry.append(
            {
                "family": ab["family"],
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
            "languages": language_keys,
            "pages_per_language": args.pages_per_language,
            "pairs": pair_list,
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
