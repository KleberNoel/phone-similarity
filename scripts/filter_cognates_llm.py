#!/usr/bin/env python3
"""LLM adjudication for cognate-vs-pun filtering.

Consumes a JSONL queue (typically from scripts/filter_pun_candidates.py) and
classifies each pair as one of:
  - cognate
  - loanword
  - named_entity_transfer
  - chance_homophone
  - unknown

Backends:
  - hf: Hugging Face Inference API
  - vllm: local vLLM OpenAI-compatible endpoint
  - mock: deterministic offline mode for pipeline testing

Decision policy:
  - Remove if label in {cognate, loanword, named_entity_transfer} and confidence >= 0.70
  - Keep if label == chance_homophone and confidence >= 0.60
  - Otherwise route to manual review
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


ALLOWED_LABELS = {
    "cognate",
    "loanword",
    "named_entity_transfer",
    "chance_homophone",
    "unknown",
}
REMOVE_LABELS = {"cognate", "loanword", "named_entity_transfer"}

SYSTEM_PROMPT = (
    "You are a linguistics adjudicator for cross-language phonological matches. "
    "Return ONLY strict JSON with keys: label, confidence, reason, remove_from_pun_set. "
    "label must be one of: cognate, loanword, named_entity_transfer, chance_homophone, unknown. "
    "confidence must be a float in [0,1]. "
    "remove_from_pun_set must be true for cognate/loanword/named_entity_transfer and false otherwise."
)


@dataclass
class LLMResult:
    label: str
    confidence: float
    reason: str
    remove_from_pun_set: bool
    raw_text: str


def _canonical_label(label: str) -> str:
    x = label.strip().lower().replace("-", "_").replace(" ", "_")
    alias = {
        "chance_pun": "chance_homophone",
        "named_entity": "named_entity_transfer",
        "named_entity_transliteration": "named_entity_transfer",
    }
    x = alias.get(x, x)
    if x not in ALLOWED_LABELS:
        return "unknown"
    return x


def _extract_json_blob(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        obj = json.loads(blob)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def _normalize_result(raw_text: str) -> LLMResult:
    parsed = _extract_json_blob(raw_text)
    if not parsed:
        return LLMResult(
            label="unknown",
            confidence=0.0,
            reason="invalid_json_response",
            remove_from_pun_set=False,
            raw_text=raw_text,
        )

    label = _canonical_label(str(parsed.get("label", "unknown")))
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(parsed.get("reason", "")).strip() or "no_reason"
    remove_flag = bool(parsed.get("remove_from_pun_set", label in REMOVE_LABELS))
    # enforce policy-consistent remove flag
    remove_flag = label in REMOVE_LABELS if label in ALLOWED_LABELS else remove_flag

    return LLMResult(
        label=label,
        confidence=confidence,
        reason=reason,
        remove_from_pun_set=remove_flag,
        raw_text=raw_text,
    )


def _decide(result: LLMResult) -> str:
    if result.label in REMOVE_LABELS and result.confidence >= 0.70:
        return "remove"
    if result.label == "chance_homophone" and result.confidence >= 0.60:
        return "keep"
    return "manual_review"


def _hf_infer(
    prompt: str,
    model: str,
    token: str,
    max_new_tokens: int,
    temperature: float,
    retries: int,
    retry_sleep: float,
) -> str:
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": f"{SYSTEM_PROMPT}\n\n{prompt}",
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False,
        },
    }

    last_err = ""
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code >= 500:
                raise RuntimeError(f"hf_server_{resp.status_code}: {resp.text[:240]}")
            if resp.status_code == 429:
                raise RuntimeError("hf_rate_limited")
            if resp.status_code >= 400:
                raise RuntimeError(f"hf_client_{resp.status_code}: {resp.text[:240]}")

            data = resp.json()
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return str(data[0].get("generated_text", ""))
            if isinstance(data, dict):
                if "generated_text" in data:
                    return str(data.get("generated_text", ""))
                if "error" in data:
                    raise RuntimeError(f"hf_error: {data['error']}")
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if attempt >= retries:
                break
            time.sleep(retry_sleep * (2**attempt))

    return json.dumps(
        {
            "label": "unknown",
            "confidence": 0.0,
            "reason": f"hf_request_failed:{last_err}",
            "remove_from_pun_set": False,
        },
        ensure_ascii=False,
    )


def _vllm_infer(
    prompt: str,
    model: str,
    base_url: str,
    max_new_tokens: int,
    temperature: float,
    retries: int,
    retry_sleep: float,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }

    last_err = ""
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code >= 500:
                raise RuntimeError(f"vllm_server_{resp.status_code}: {resp.text[:240]}")
            if resp.status_code == 429:
                raise RuntimeError("vllm_rate_limited")
            if resp.status_code >= 400:
                raise RuntimeError(f"vllm_client_{resp.status_code}: {resp.text[:240]}")

            data = resp.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message", {})
                return str(msg.get("content", ""))
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if attempt >= retries:
                break
            time.sleep(retry_sleep * (2**attempt))

    return json.dumps(
        {
            "label": "unknown",
            "confidence": 0.0,
            "reason": f"vllm_request_failed:{last_err}",
            "remove_from_pun_set": False,
        },
        ensure_ascii=False,
    )


def _mock_infer(prompt: str) -> str:
    p = prompt.casefold()
    if any(k in p for k in ["district", "distrikt", "plaza", "praça", "batalla", "batalha"]):
        label = "cognate"
        confidence = 0.94
        reason = "high orthographic and phonological overlap in related languages"
    elif any(k in p for k in ["robinson", "ロービジョン", "апэрэ", "ガバナー"]):
        label = "named_entity_transfer"
        confidence = 0.81
        reason = "looks like transliterated name transfer"
    elif random.random() < 0.25:
        label = "chance_homophone"
        confidence = 0.66
        reason = "phonological proximity without obvious etymological identity"
    else:
        label = "unknown"
        confidence = 0.42
        reason = "insufficient certainty"

    return json.dumps(
        {
            "label": label,
            "confidence": confidence,
            "reason": reason,
            "remove_from_pun_set": label in REMOVE_LABELS,
        },
        ensure_ascii=False,
    )


def _two_pass_vote(a: LLMResult, b: LLMResult) -> LLMResult:
    if a.label == b.label:
        return LLMResult(
            label=a.label,
            confidence=(a.confidence + b.confidence) / 2.0,
            reason=f"two_pass_agree: {a.reason} | {b.reason}",
            remove_from_pun_set=a.label in REMOVE_LABELS,
            raw_text=a.raw_text + "\n---\n" + b.raw_text,
        )

    # disagreement -> conservative unknown
    return LLMResult(
        label="unknown",
        confidence=min(a.confidence, b.confidence),
        reason=f"two_pass_disagree: pass1={a.label}, pass2={b.label}",
        remove_from_pun_set=False,
        raw_text=a.raw_text + "\n---\n" + b.raw_text,
    )


def _infer_once(args: argparse.Namespace, prompt: str) -> LLMResult:
    if args.backend == "hf":
        token = os.environ.get(args.hf_token_env, "")
        if not token:
            raise RuntimeError(
                f"Missing Hugging Face token in env var {args.hf_token_env}. "
                "Set it before using --backend hf."
            )
        raw = _hf_infer(
            prompt,
            args.model,
            token,
            args.max_new_tokens,
            args.temperature,
            args.retries,
            args.retry_sleep,
        )
    elif args.backend == "vllm":
        raw = _vllm_infer(
            prompt,
            args.model,
            args.vllm_base_url,
            args.max_new_tokens,
            args.temperature,
            args.retries,
            args.retry_sleep,
        )
    else:
        raw = _mock_infer(prompt)
    return _normalize_result(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM cognate adjudication")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("notebooks/wikipedia_pun_mining_multilingual_500each_llm_queue.jsonl"),
    )
    parser.add_argument(
        "--backend",
        choices=["hf", "vllm", "mock"],
        default="mock",
        help="LLM backend; use mock for offline pipeline testing",
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--hf-token-env", type=str, default="HF_TOKEN")
    parser.add_argument("--vllm-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--two-pass-voting", action="store_true")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("notebooks/pun_candidates_llm_adjudicated.jsonl"),
    )
    parser.add_argument(
        "--removed-output-jsonl",
        type=Path,
        default=Path("notebooks/pun_candidates_removed_cognates_llm.jsonl"),
    )
    parser.add_argument(
        "--noncognate-output-jsonl",
        type=Path,
        default=Path("notebooks/pun_candidates_noncognate_llm.jsonl"),
    )
    parser.add_argument(
        "--manual-review-output-jsonl",
        type=Path,
        default=Path("notebooks/pun_candidates_manual_review_llm.jsonl"),
    )
    args = parser.parse_args()

    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_jsonl}")

    rows = []
    for line in args.input_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    if args.limit > 0:
        rows = rows[: args.limit]

    adjudicated = []
    removed = []
    kept = []
    manual = []

    for i, row in enumerate(rows, start=1):
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            prompt = (
                "Classify this cross-language phonological pair as one of: "
                "cognate, loanword, named_entity_transfer, chance_homophone, unknown. "
                "Return strict JSON with label, confidence, reason, remove_from_pun_set.\n\n"
                f"source_lang={row.get('source_lang')} target_lang={row.get('target_lang')}\n"
                f"source_word={row.get('source_word')} target_word={row.get('target_word')}\n"
                f"distance={row.get('distance')} similarity={row.get('similarity')}"
            )

        pass1 = _infer_once(args, prompt)
        if args.two_pass_voting:
            prompt2 = (
                prompt + "\n\nSecond pass instruction: be conservative; if uncertain between "
                "cognate/loanword/name-transfer vs chance_homophone, choose unknown."
            )
            pass2 = _infer_once(args, prompt2)
            final = _two_pass_vote(pass1, pass2)
        else:
            final = pass1

        decision = _decide(final)
        rec = {
            **row,
            "llm_label": final.label,
            "llm_confidence": final.confidence,
            "llm_reason": final.reason,
            "remove_from_pun_set": final.remove_from_pun_set,
            "decision": decision,
            "llm_raw_text": final.raw_text,
            "backend": args.backend,
            "model": args.model,
            "two_pass_voting": args.two_pass_voting,
        }
        adjudicated.append(rec)

        if decision == "remove":
            removed.append(rec)
        elif decision == "keep":
            kept.append(rec)
        else:
            manual.append(rec)

        if i % 50 == 0:
            print(f"processed {i}/{len(rows)}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for r in adjudicated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    args.removed_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.removed_output_jsonl.open("w", encoding="utf-8") as f:
        for r in removed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    args.noncognate_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.noncognate_output_jsonl.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    args.manual_review_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.manual_review_output_jsonl.open("w", encoding="utf-8") as f:
        for r in manual:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Input rows: {len(rows)}")
    print(f"Adjudicated: {len(adjudicated)}")
    print(f"Removed: {len(removed)}")
    print(f"Kept: {len(kept)}")
    print(f"Manual review: {len(manual)}")
    print(f"Saved adjudicated JSONL to {args.output_jsonl}")
    print(f"Saved removed JSONL to {args.removed_output_jsonl}")
    print(f"Saved non-cognate JSONL to {args.noncognate_output_jsonl}")
    print(f"Saved manual-review JSONL to {args.manual_review_output_jsonl}")


if __name__ == "__main__":
    main()
