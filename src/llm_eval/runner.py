from __future__ import annotations

import json
import time
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

from llm_eval.scorers import score

Model = Callable[[str], str]


def load_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    if path is None:
        text = resources.files("llm_eval").joinpath("data/sample_cases.jsonl").read_text()
    else:
        text = Path(path).read_text()
    cases = [json.loads(line) for line in text.splitlines() if line.strip()]
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Case ids must be unique")
    return cases


def run_suite(name: str, model: Model, cases: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for case in cases:
        started = time.perf_counter()
        error = None
        output = ""
        try:
            output = str(model(case["input"]))
            value, detail = score(output, case)
        except Exception as exc:  # noqa: BLE001 - a failing model call is a failed case
            value, detail, error = 0.0, f"{type(exc).__name__}: {exc}", str(exc)
        results.append({
            "id": case["id"],
            "input": case["input"],
            "output": output,
            "score": value,
            "passed": value >= 1.0,
            "detail": detail,
            "error": error,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        })
    passed = sum(r["passed"] for r in results)
    return {
        "model": name,
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "results": results,
    }


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare two runs by case id: what regressed, what improved."""
    by_a = {r["id"]: r for r in a["results"]}
    by_b = {r["id"]: r for r in b["results"]}
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
