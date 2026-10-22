from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from llm_eval.scorers import SCORERS, MissingDependency, score


@dataclass
class ModelResponse:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


Model = Callable[[str], "str | ModelResponse"]


NEEDS_EXPECTED = {"exact_match", "contains", "json_equals", "number_close", "semantic_similarity"}
NEEDS_PARAM = {"regex": "pattern", "json_keys": "keys", "json_schema": "schema", "llm_judge": "rubric"}


class InvalidSuite(ValueError):
    """The case file is malformed."""


def validate_case(case: Any, index: int) -> None:
    """Fail fast with a readable message instead of a per-case error at run time."""
    where = f"case #{index + 1}"
    if not isinstance(case, dict):
        raise InvalidSuite(f"{where}: must be a JSON object")
    for field in ("id", "input", "scorer"):
        if field not in case:
            raise ValueError(f"{where}: missing required field {field!r}")
    where = f"case {case['id']!r}"
    if case["scorer"] not in SCORERS:
        raise ValueError(f"{where}: unknown scorer {case['scorer']!r}; available: {sorted(SCORERS)}")
    if case["scorer"] in NEEDS_EXPECTED and "expected" not in case:
        raise ValueError(f"{where}: scorer {case['scorer']!r} requires 'expected'")
    param = NEEDS_PARAM.get(case["scorer"])
    if param and param not in case.get("params", {}):
        raise ValueError(f"{where}: scorer {case['scorer']!r} requires params.{param}")


def load_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    if path is None:
        text = resources.files("llm_eval").joinpath("data/sample_cases.jsonl").read_text()
    else:
        text = Path(path).read_text()
    cases = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {number}: invalid JSON ({exc.msg})") from exc
    for index, case in enumerate(cases):
        validate_case(case, index)
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Case ids must be unique")
    return cases


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def run_suite(
    name: str,
    model: Model,
    cases: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    pricing: tuple[float, float] | None = None,
    ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run `model` over `cases`.

    pricing is (USD per 1M prompt tokens, USD per 1M completion tokens), supplied by the caller;
    no built-in price table is shipped because prices change.
    """
    results = []
    for case in cases:
        started = time.perf_counter()
        error = None
        output, value, detail, passed, skipped = "", 0.0, "", False, False
        prompt_tokens = completion_tokens = None
        try:
            response = model(case["input"])
            if isinstance(response, ModelResponse):
                output = response.text
                prompt_tokens, completion_tokens = response.prompt_tokens, response.completion_tokens
            else:
                output = str(response)
            value, detail, passed = score(output, case, ctx)
        except MissingDependency as exc:
            skipped, detail = True, str(exc)
        except Exception as exc:  # noqa: BLE001 - a failing model call is a failed case
            value, detail, passed, error = 0.0, f"{type(exc).__name__}: {exc}", False, str(exc)
        results.append({
            "id": case["id"],
            "category": case.get("category", "uncategorized"),
            "input": case["input"],
            "output": output,
            "score": value,
            "passed": passed,
            "skipped": skipped,
            "detail": detail,
            "error": error,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })
    return {"model": name, "config": config or {}, **summarize(results, pricing)}


def summarize(results: list[dict[str, Any]], pricing: tuple[float, float] | None) -> dict[str, Any]:
    scored = [r for r in results if not r["skipped"]]
    passed = sum(r["passed"] for r in scored)
    latencies = [r["latency_ms"] for r in scored]
    p_tokens = [r["prompt_tokens"] for r in scored if r["prompt_tokens"] is not None]
    c_tokens = [r["completion_tokens"] for r in scored if r["completion_tokens"] is not None]
    by_category: dict[str, list[bool]] = defaultdict(list)
    for r in scored:
        by_category[r["category"]].append(r["passed"])
    telemetry: dict[str, Any] = {
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "prompt_tokens": sum(p_tokens) if p_tokens else None,
        "completion_tokens": sum(c_tokens) if c_tokens else None,
        "cost_usd": None,
    }
    if pricing and p_tokens and c_tokens:
        telemetry["cost_usd"] = round(
            sum(p_tokens) * pricing[0] / 1e6 + sum(c_tokens) * pricing[1] / 1e6, 6)
    return {
        "total": len(scored),
        "passed": passed,
        "skipped": len(results) - len(scored),
        "pass_rate": round(passed / len(scored), 4) if scored else 0.0,
        "by_category": {c: {"passed": sum(v), "total": len(v)} for c, v in sorted(by_category.items())},
        "telemetry": telemetry,
        "results": results,
    }


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare two runs by case id: what regressed, what improved."""
    by_a = {r["id"]: r for r in a["results"] if not r["skipped"]}
    by_b = {r["id"]: r for r in b["results"] if not r["skipped"]}
    shared = [i for i in by_a if i in by_b]
    return {
        "a": a["model"],
        "b": b["model"],
        "pass_rate_a": a["pass_rate"],
        "pass_rate_b": b["pass_rate"],
        "delta": round(b["pass_rate"] - a["pass_rate"], 4),
        "regressions": [i for i in shared if by_a[i]["passed"] and not by_b[i]["passed"]],
        "improvements": [i for i in shared if not by_a[i]["passed"] and by_b[i]["passed"]],
        "only_in_a": [i for i in by_a if i not in by_b],
        "only_in_b": [i for i in by_b if i not in by_a],
    }
