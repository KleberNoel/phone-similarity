import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_llm = pytest.importorskip("scripts.filter_cognates_llm")
_decide = _llm._decide
_normalize_result = _llm._normalize_result


def test_llm_normalize_and_decision_remove_for_cognate_high_confidence():
    raw = json.dumps(
        {
            "label": "cognate",
            "confidence": 0.91,
            "reason": "close orthographic + phonological overlap",
            "remove_from_pun_set": True,
        }
    )
    rec = _normalize_result(raw)
    assert rec.label == "cognate"
    assert rec.confidence == 0.91
    assert rec.remove_from_pun_set is True
    assert _decide(rec) == "remove"


def test_llm_normalize_and_decision_keep_for_chance_homophone():
    raw = json.dumps(
        {
            "label": "chance_homophone",
            "confidence": 0.72,
            "reason": "phonological proximity without clear etymological identity",
            "remove_from_pun_set": False,
        }
    )
    rec = _normalize_result(raw)
    assert rec.label == "chance_homophone"
    assert rec.remove_from_pun_set is False
    assert _decide(rec) == "keep"


def test_llm_normalize_invalid_json_falls_back_to_unknown():
    rec = _normalize_result("not-json")
    assert rec.label == "unknown"
    assert rec.confidence == 0.0
    assert rec.remove_from_pun_set is False
    assert _decide(rec) == "manual_review"


def test_llm_alias_mapping_from_chance_pun_to_chance_homophone():
    raw = json.dumps(
        {
            "label": "chance_pun",
            "confidence": 0.64,
            "reason": "legacy label alias",
            "remove_from_pun_set": False,
        }
    )
    rec = _normalize_result(raw)
    assert rec.label == "chance_homophone"
    assert _decide(rec) == "keep"
