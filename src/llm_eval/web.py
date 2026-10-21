from __future__ import annotations

import os
from html import escape

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from llm_eval.models import BASELINES
from llm_eval.report import render_report
from llm_eval.runner import load_cases, run_suite

CASES = load_cases()


def _controls(a: str, b: str) -> str:
    def select(name: str, current: str) -> str:
        opts = "".join(
            f'<option value="{escape(m)}"{" selected" if m == current else ""}>{escape(m)}</option>'
            for m in BASELINES)
        return f'<label>{name} <select name="{name}">{opts}</select></label>'
    return (f'<form method="get">{select("a", a)}{select("b", b)}'
            f'<button>Compare</button></form>'
            f'<p class="sub">Demo models are deterministic rule-based baselines, not LLMs. '
            f'Use the CLI with <code>openai:MODEL@URL</code> to evaluate a real model.</p>')


async def index(request: Request) -> HTMLResponse:
    a = request.query_params.get("a", "rules-v1")
    b = request.query_params.get("b", "rules-v2")
    if a not in BASELINES or b not in BASELINES:
        return HTMLResponse("Unknown model.", status_code=400)
    run_a, run_b = (run_suite(m, BASELINES[m], CASES) for m in (a, b))
    return HTMLResponse(render_report(run_a, run_b, controls=_controls(a, b)))


app = Starlette(routes=[Route("/", index)])


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
