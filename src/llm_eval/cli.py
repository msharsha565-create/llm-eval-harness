from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from llm_eval.embeddings import sentence_transformer_embedder
from llm_eval.models import BASELINES, resolve_model
from llm_eval.report import render_report
from llm_eval.runner import compare, load_cases, run_suite


def _text(response: object) -> str:
    return getattr(response, "text", response)  # ModelResponse -> str, plain str unchanged


def _load_run(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-eval", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a model over a case file and save the results")
    run.add_argument("--model", required=True, help=f"{sorted(BASELINES)} or openai:MODEL@URL")
    run.add_argument("--cases", help="JSONL case file (default: built-in sample suite)")
    run.add_argument("--out", help="Write run JSON here")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--top-p", type=float)
    run.add_argument("--seed", type=int)
    run.add_argument("--max-tokens", type=int)
    run.add_argument("--system", help="System prompt for the model under test")
    run.add_argument("--price-in", type=float, help="USD per 1M prompt tokens")
    run.add_argument("--price-out", type=float, help="USD per 1M completion tokens")
    run.add_argument("--judge", help="Judge model spec (openai:MODEL@URL) for llm_judge cases")
    run.add_argument("--embedder", default="hashed",
                     help="'hashed' (default, no deps) or 'st:MODEL' (sentence-transformers)")

    cmp = sub.add_parser("compare", help="Compare two runs; exit 1 if anything regressed")
    cmp.add_argument("a")
    cmp.add_argument("b")

    rep = sub.add_parser("report", help="Write a self-contained HTML report")
    rep.add_argument("runs", nargs="+", help="One run, or two to compare")
    rep.add_argument("-o", "--out", default="report.html")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        key = os.environ.get("OPENAI_API_KEY")
        model = resolve_model(args.model, api_key=key, system=args.system,
                              temperature=args.temperature, top_p=args.top_p, seed=args.seed,
                              max_tokens=args.max_tokens)
        ctx: dict = {}
        if args.judge:
            judge_model = resolve_model(args.judge, api_key=key)
            ctx["judge"] = lambda prompt: _text(judge_model(prompt))
        if args.embedder.startswith("st:"):
            ctx["embedder"] = sentence_transformer_embedder(args.embedder[3:])
        pricing = (args.price_in, args.price_out) if args.price_in is not None and args.price_out is not None else None
        config = dict(getattr(model, "config", {}) or {"type": "baseline"})
        config["embedder"] = args.embedder
        if args.judge:
            config["judge"] = args.judge
        result = run_suite(args.model, model, load_cases(args.cases), config=config,
                           pricing=pricing, ctx=ctx)
        text = json.dumps(result, indent=2)
        if args.out:
            Path(args.out).write_text(text)
        t = result["telemetry"]
        skipped = f", {result['skipped']} skipped" if result["skipped"] else ""
        print(f"{args.model}: {result['passed']}/{result['total']} passed "
              f"({result['pass_rate']:.0%}){skipped}")
        print(f"  latency p50 {t['latency_p50_ms']} ms, p95 {t['latency_p95_ms']} ms"
              + (f", tokens {t['prompt_tokens']}/{t['completion_tokens']}" if t["prompt_tokens"] is not None else "")
              + (f", cost ${t['cost_usd']}" if t["cost_usd"] is not None else ""))
        return 0

    if args.cmd == "compare":
        diff = compare(_load_run(args.a), _load_run(args.b))
        print(f"{diff['a']} -> {diff['b']}: pass rate {diff['pass_rate_a']:.0%} -> "
              f"{diff['pass_rate_b']:.0%} ({diff['delta']:+.0%})")
        for label in ("regressions", "improvements"):
            if diff[label]:
                print(f"  {label}: {', '.join(diff[label])}")
        return 1 if diff["regressions"] else 0

    runs = [_load_run(p) for p in args.runs[:2]]
    Path(args.out).write_text(render_report(*runs))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
