from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

Scorer = Callable[[str, dict[str, Any]], tuple[float, str]]

_TYPES = {"str": str, "int": int, "float": (int, float), "bool": bool, "list": list, "dict": dict}


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower().rstrip(".")


def _parse_json(output: str) -> Any:
    text = output.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return json.loads(fenced.group(1) if fenced else text)


def exact_match(output: str, case: dict[str, Any]) -> tuple[float, str]:
    ok = _norm(output) == _norm(case["expected"])
    return (1.0, "") if ok else (0.0, f"expected {case['expected']!r}")


def contains(output: str, case: dict[str, Any]) -> tuple[float, str]:
    ok = _norm(case["expected"]) in _norm(output)
    return (1.0, "") if ok else (0.0, f"missing {case['expected']!r}")


def regex(output: str, case: dict[str, Any]) -> tuple[float, str]:
    pattern = case["params"]["pattern"]
    ok = re.search(pattern, output.strip()) is not None
    return (1.0, "") if ok else (0.0, f"does not match /{pattern}/")


def json_equals(output: str, case: dict[str, Any]) -> tuple[float, str]:
    try:
        parsed = _parse_json(output)
    except (json.JSONDecodeError, ValueError):
        return 0.0, "output is not valid JSON"
    return (1.0, "") if parsed == case["expected"] else (0.0, f"expected {case['expected']!r}")


def json_keys(output: str, case: dict[str, Any]) -> tuple[float, str]:
    """Valid JSON object with the required keys and value types."""
    try:
        parsed = _parse_json(output)
    except (json.JSONDecodeError, ValueError):
        return 0.0, "output is not valid JSON"
    if not isinstance(parsed, dict):
        return 0.0, "JSON is not an object"
    for key, type_name in case["params"]["keys"].items():
        if key not in parsed:
            return 0.0, f"missing key {key!r}"
        value = parsed[key]
        if isinstance(value, bool) and type_name != "bool":
            return 0.0, f"key {key!r} should be {type_name}"
        if not isinstance(value, _TYPES[type_name]):
            return 0.0, f"key {key!r} should be {type_name}"
    return 1.0, ""


def number_close(output: str, case: dict[str, Any]) -> tuple[float, str]:
    match = re.search(r"-?\d+(?:\.\d+)?", output)
    if not match:
        return 0.0, "no number in output"
    tolerance = case.get("params", {}).get("tolerance", 1e-6)
    ok = abs(float(match.group()) - float(case["expected"])) <= tolerance
    return (1.0, "") if ok else (0.0, f"got {match.group()}, expected {case['expected']}")


SCORERS: dict[str, Scorer] = {
    "exact_match": exact_match,
    "contains": contains,
    "regex": regex,
    "json_equals": json_equals,
    "json_keys": json_keys,
    "number_close": number_close,
}


def score(output: str, case: dict[str, Any]) -> tuple[float, str]:
    name = case["scorer"]
    if name not in SCORERS:
        raise ValueError(f"Unknown scorer {name!r}. Available: {sorted(SCORERS)}")
    return SCORERS[name](output, case)
