from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from jsonschema import Draft202012Validator

from llm_eval.embeddings import cosine, hashed_ngram_embedder

Context = dict[str, Any]
Scorer = Callable[[str, dict[str, Any], Context], tuple[float, str]]

_TYPES = {"str": str, "int": int, "float": (int, float), "bool": bool, "list": list, "dict": dict}
# Graded scorers produce a score in [0, 1]; a case passes when score >= pass_at.
DEFAULT_PASS_AT = {"semantic_similarity": 0.85, "llm_judge": 0.7}
NEEDS_JUDGE = {"llm_judge"}


class MissingDependency(Exception):
    """The scorer needs something (e.g. a judge model) that was not configured."""


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower().rstrip(".")


def _parse_json(output: str) -> Any:
    text = output.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return json.loads(fenced.group(1) if fenced else text)


def exact_match(output: str, case: dict[str, Any], ctx: Context) -> tuple[float, str]:
    ok = _norm(output) == _norm(case["expected"])
    return (1.0, "") if ok else (0.0, f"expected {case['expected']!r}")


def contains(output: str, case: dict[str, Any], ctx: Context) -> tuple[float, str]:
    ok = _norm(case["expected"]) in _norm(output)
    return (1.0, "") if ok else (0.0, f"missing {case['expected']!r}")


def regex(output: str, case: dict[str, Any], ctx: Context) -> tuple[float, str]:
    pattern = case["params"]["pattern"]
    ok = re.search(pattern, output.strip()) is not None
    return (1.0, "") if ok else (0.0, f"does not match /{pattern}/")


def json_equals(output: str, case: dict[str, Any], ctx: Context) -> tuple[float, str]:
    try:
        parsed = _parse_json(output)
    except ValueError:
        return 0.0, "output is not valid JSON"
    return (1.0, "") if parsed == case["expected"] else (0.0, f"expected {case['expected']!r}")


def json_keys(output: str, case: dict[str, Any], ctx: Context) -> tuple[float, str]:
    """Valid JSON object with the required keys and value types."""
    try:
        parsed = _parse_json(output)
    except ValueError:
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


def json_schema(output: str, case: dict[str, Any], ctx: Context) -> tuple[float, str]:
    """Output must parse as JSON and validate against params.schema (JSON Schema 2020-12)."""
    try:
        parsed = _parse_json(output)
    except ValueError:
        return 0.0, "output is not valid JSON"
    errors = sorted(Draft202012Validator(case["params"]["schema"]).iter_errors(parsed),
                    key=lambda e: list(e.path))
    if errors:
        where = "/".join(str(p) for p in errors[0].path) or "(root)"
        return 0.0, f"schema violation at {where}: {errors[0].message}"
    return 1.0, ""


def number_close(output: str, case: dict[str, Any], ctx: Context) -> tuple[float, str]:
    match = re.search(r"-?\d+(?:\.\d+)?", output)
    if not match:
        return 0.0, "no number in output"
    tolerance = case.get("params", {}).get("tolerance", 1e-6)
    ok = abs(float(match.group()) - float(case["expected"])) <= tolerance
    return (1.0, "") if ok else (0.0, f"got {match.group()}, expected {case['expected']}")


def semantic_similarity(output: str, case: dict[str, Any], ctx: Context) -> tuple[float, str]:
    """Cosine similarity between embeddings of the output and the reference answer."""
    embed = ctx.get("embedder") or hashed_ngram_embedder
    got, want = embed([output, str(case["expected"])])
    sim = max(0.0, min(1.0, cosine(got, want)))
    return round(sim, 4), f"cosine similarity {sim:.2f}"


JUDGE_PROMPT = """You are grading an AI answer against a rubric.

Question: {question}
Answer to grade: {answer}
Rubric: {rubric}

Reply with ONLY a JSON object: {{"score": <integer 0-10>, "reason": "<one sentence>"}}"""


def llm_judge(output: str, case: dict[str, Any], ctx: Context) -> tuple[float, str]:
    judge = ctx.get("judge")
    if judge is None:
        raise MissingDependency("requires a judge model (--judge)")
    prompt = JUDGE_PROMPT.format(question=case["input"], answer=output,
                                 rubric=case["params"]["rubric"])
    try:
        verdict = _parse_json(str(judge(prompt)))
        value = max(0.0, min(10.0, float(verdict["score"]))) / 10
    except (ValueError, KeyError, TypeError):
        return 0.0, "judge returned an unparseable verdict"
    return round(value, 2), f"judge: {verdict.get('reason', '')}".strip()


SCORERS: dict[str, Scorer] = {
    "exact_match": exact_match,
    "contains": contains,
    "regex": regex,
    "json_equals": json_equals,
    "json_keys": json_keys,
    "json_schema": json_schema,
    "number_close": number_close,
    "semantic_similarity": semantic_similarity,
    "llm_judge": llm_judge,
}


def score(output: str, case: dict[str, Any], ctx: Context | None = None) -> tuple[float, str, bool]:
    """Returns (score, detail, passed)."""
    name = case["scorer"]
    if name not in SCORERS:
        raise ValueError(f"Unknown scorer {name!r}. Available: {sorted(SCORERS)}")
    value, detail = SCORERS[name](output, case, ctx or {})
    pass_at = case.get("pass_at", case.get("params", {}).get("threshold",
                                                            DEFAULT_PASS_AT.get(name, 1.0)))
    return value, detail, value >= pass_at
