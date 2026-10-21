from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from llm_eval.models import BASELINES, resolve_model
from llm_eval.report import render_report
from llm_eval.runner import compare, load_cases, run_suite


def _load_run(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-eval", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a model over a case file and save the results")
    run.add_argument("--model", required=True, help=f"{sorted(BASELINES)} or openai:MODEL@URL")
    run.add_argument("--cases", help="JSONL case file (default: built-in sample suite)")
    run.add_argument("--out", help="Write run JSON here")

    cmp = sub.add_parser("compare", help="Compare two runs; exit 1 if anything regressed")
    cmp.add_argument("a")
    cmp.add_argument("b")

    rep = sub.add_parser("report", help="Write a self-contained HTML report")
    rep.add_argument("runs", nargs="+", help="One run, or two to compare")
    rep.add_argument("-o", "--out", default="report.html")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        model = resolve_model(args.model, api_key=os.environ.get("OPENAI_API_KEY"))
        result = run_suite(args.model, model, load_cases(args.cases))
        text = json.dumps(result, indent=2)
        if args.out:
            Path(args.out).write_text(text)
        print(f"{args.model}: {result['passed']}/{result['total']} passed "
              f"({result['pass_rate']:.0%})")
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
