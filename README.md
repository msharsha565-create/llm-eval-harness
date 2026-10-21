# llm-eval-harness

Regression testing for LLM outputs. Define test cases, score model outputs, compare two runs, and
fail CI when something that used to pass now fails. Works with any OpenAI-compatible endpoint
(OpenAI, Ollama, vLLM, LM Studio) or any Python function.

![demo](docs/demo.png)

## Why

A single pass-rate number hides what changed. In the bundled demo, `rules-v1` and `rules-v2` both
score **88%**, yet v2 breaks 2 cases and fixes 2 others. Case-level comparison catches that;
`llm-eval compare` exits non-zero on any regression, so it works as a CI gate.

## Demo (no API key)

```bash
pip install -e ".[dev]"
llm-eval-web                       # http://127.0.0.1:8000
```

The demo models are deterministic **rule-based baselines, not LLMs**. They exist so the harness
has reproducible behaviour for the demo and the tests.

## CLI

```bash
llm-eval run --model rules-v1 --out v1.json
llm-eval run --model rules-v2 --out v2.json
llm-eval compare v1.json v2.json      # prints diff, exit code 1 if regressions
llm-eval report v1.json v2.json -o report.html
```

Evaluate a real model through an OpenAI-compatible endpoint:

```bash
# local Ollama, no key
llm-eval run --model "openai:llama3@http://localhost:11434/v1" --out llama.json
# OpenAI
OPENAI_API_KEY=... llm-eval run --model "openai:gpt-4o-mini" --out gpt.json
```

## Cases and scorers

Cases are JSONL (`--cases my.jsonl`): `{"id", "input", "expected", "scorer", "params"}`.

| Scorer | Passes when |
| --- | --- |
| `exact_match` | normalized output equals `expected` |
| `contains` | `expected` appears in the output |
| `regex` | output matches `params.pattern` |
| `json_equals` | output parses as JSON (fences allowed) and equals `expected` |
| `json_keys` | JSON object has `params.keys` with the right types |
| `number_close` | first number in output is within `params.tolerance` of `expected` |

A model call that raises (timeout, HTTP error) is recorded as a failed case, not a crashed run.

## Tests

```bash
python -m pytest
ruff check src tests
```

Covers every scorer, run comparison, report HTML escaping, the CLI exit codes, the web page, and
the OpenAI-compatible adapter against a mocked HTTP transport.

## Limitations

- The OpenAI-compatible adapter is tested against a mock server only; I have not run it against a
  live endpoint.
- Scoring is deterministic string/JSON/number matching. There is no LLM-as-judge scorer.
- Runs are sequential, and there is no retry or rate limiting.
