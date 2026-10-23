import json

import httpx
import pytest

from llm_eval import models
from llm_eval.models import OpenAICompatModel, resolve_model
from llm_eval.report import render_report
from llm_eval.runner import ModelResponse, compare, load_cases, percentile, run_suite
from llm_eval.scorers import score


def case(scorer, expected=None, **params):
    return {"id": "t", "input": "", "expected": expected, "scorer": scorer, "params": params}


@pytest.mark.parametrize("scorer,expected,params,output,passed", [
    ("exact_match", "42", {}, " 42. ", True),
    ("exact_match", "42", {}, "43", False),
    ("contains", "Paris", {}, "It is paris, France", True),
    ("contains", "Paris", {}, "Berlin", False),
    ("regex", None, {"pattern": r"^\d{4}-\d{2}-\d{2}$"}, "2024-03-05", True),
    ("regex", None, {"pattern": r"^\d{4}-\d{2}-\d{2}$"}, "March 5", False),
    ("json_equals", {"a": 1}, {}, '{"a": 1}', True),
    ("json_equals", {"a": 1}, {}, '```json\n{"a": 1}\n```', True),
    ("json_equals", {"a": 1}, {}, "not json", False),
    ("json_keys", None, {"keys": {"name": "str", "age": "int"}}, '{"name": "A", "age": 3}', True),
    ("json_keys", None, {"keys": {"age": "int"}}, '{"age": true}', False),
    ("json_keys", None, {"keys": {"age": "int"}}, '{"name": "A"}', False),
    ("json_keys", None, {"keys": {"age": "int"}}, "[1]", False),
    ("number_close", 3.14, {"tolerance": 0.01}, "about 3.142", True),
    ("number_close", 3.14, {"tolerance": 0.01}, "3.5", False),
    ("number_close", 3.14, {"tolerance": 0.01}, "none", False),
])
def test_scorers(scorer, expected, params, output, passed):
    _, _, ok = score(output, case(scorer, expected, **params))
    assert ok is passed


def test_unknown_scorer():
    with pytest.raises(ValueError):
        score("x", {"scorer": "nope"})


def test_sample_suite_is_large_categorized_and_unique():
    cases = load_cases()
    assert len(cases) >= 50
    assert len({c["id"] for c in cases}) == len(cases)
    assert len({c["category"] for c in cases}) >= 6


def test_run_suite_records_errors_instead_of_crashing():
    def flaky(prompt):
        raise RuntimeError("boom")

    result = run_suite("flaky", flaky, load_cases()[:2])
    assert result["passed"] == 0
    assert result["results"][0]["error"] == "boom"


def test_compare_finds_regressions_and_improvements_hidden_by_equal_pass_rate():
    cases = load_cases()
    v1 = run_suite("rules-v1", models.rules_v1, cases)
    v2 = run_suite("rules-v2", models.rules_v2, cases)
    diff = compare(v1, v2)
    assert v1["pass_rate"] == v2["pass_rate"]
    assert diff["only_in_a"] == [] and diff["only_in_b"] == []
    assert set(diff["regressions"]) == {"math-div", "approx-pi"}
    assert set(diff["improvements"]) == {"sent-negation", "capital-jp"}


def test_echo_baseline_fails_everything():
    assert run_suite("echo", models.echo, load_cases())["passed"] == 0


def test_openai_compat_adapter_sends_expected_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "42"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = OpenAICompatModel("m1", "http://localhost:11434/v1/", "k", system="Be brief", client=client)
    assert model("What is 6*7?").text == "42"
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "Be brief"}
    assert seen["body"]["temperature"] == 0


def test_adapter_http_error_becomes_failed_case(monkeypatch):
    monkeypatch.setattr("llm_eval.models.time.sleep", lambda s: None)
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    model = OpenAICompatModel("m", client=client)
    result = run_suite("remote", model, load_cases()[:1])
    assert result["results"][0]["passed"] is False
    assert "HTTPStatusError" in result["results"][0]["detail"]


def test_adapter_retries_rate_limits_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr("llm_eval.models.time.sleep", sleeps.append)
    replies = iter([httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(503),
                    httpx.Response(200, json={"choices": [{"message": {"content": "42"}}]})])
    client = httpx.Client(transport=httpx.MockTransport(lambda r: next(replies)))
    assert OpenAICompatModel("m", client=client)("q").text == "42"
    assert sleeps == [7.0, 10]


def test_resolve_model():
    assert resolve_model("echo") is models.echo
    assert isinstance(resolve_model("openai:llama3@http://localhost:11434/v1"), OpenAICompatModel)
    with pytest.raises(ValueError):
        resolve_model("gpt-unknown")


def test_report_escapes_html_and_marks_regressions():
    cases = load_cases()
    v1 = run_suite("rules-v1", models.rules_v1, cases)
    v2 = run_suite("rules-v2", models.rules_v2, cases)
    v2["results"][0]["output"] = "<script>alert(1)</script>"
    html = render_report(v1, v2)
    assert "<script>alert" not in html
    assert "regression" in html


PERSON_SCHEMA = {"type": "object", "required": ["name", "age"], "additionalProperties": False,
                 "properties": {"name": {"type": "string"},
                                "age": {"type": "integer", "minimum": 0, "maximum": 130}}}


@pytest.mark.parametrize("output,passed", [
    ('{"name": "Ann", "age": 30}', True),
    ('{"name": "Ann", "age": "30"}', False),
    ('{"name": "Ann", "age": 300}', False),
    ('{"name": "Ann"}', False),
    ('{"name": "Ann", "age": 3, "extra": 1}', False),
    ("not json", False),
])
def test_json_schema_scorer(output, passed):
    _, detail, ok = score(output, case("json_schema", schema=PERSON_SCHEMA))
    assert ok is passed
    assert (detail == "") is passed


def test_semantic_similarity_paraphrase_beats_unrelated():
    ref = "A load balancer distributes traffic across several servers."
    close = case("semantic_similarity", ref, threshold=0.5)
    v_close, _, ok_close = score("A load balancer spreads traffic over multiple servers.", close)
    v_far, _, ok_far = score("Bananas are yellow fruit grown in warm climates.", close)
    assert ok_close and not ok_far
    assert v_close > v_far


def test_semantic_similarity_accepts_custom_embedder():
    def embedder(texts):
        return [[1.0, 0.0] for _ in texts]

    value, _, ok = score("anything", case("semantic_similarity", "other"), {"embedder": embedder})
    assert value == 1.0 and ok


def test_llm_judge_scores_and_thresholds():
    judge_case = case("llm_judge", rubric="Must be accurate")
    judge_case["input"] = "q"
    good = {"judge": lambda prompt: '{"score": 9, "reason": "accurate"}'}
    bad = {"judge": lambda prompt: '{"score": 3, "reason": "wrong"}'}
    broken = {"judge": lambda prompt: "I refuse"}
    assert score("a", judge_case, good)[2] is True
    assert score("a", judge_case, bad)[2] is False
    value, detail, ok = score("a", judge_case, broken)
    assert (value, ok) == (0.0, False) and "unparseable" in detail


def test_judge_prompt_contains_question_answer_rubric():
    seen = []
    judge_case = case("llm_judge", rubric="RUBRIC-X")
    judge_case["input"] = "QUESTION-Y"
    score("ANSWER-Z", judge_case, {"judge": lambda p: seen.append(p) or '{"score": 10}'})
    assert all(token in seen[0] for token in ("RUBRIC-X", "QUESTION-Y", "ANSWER-Z"))


def test_judge_cases_are_skipped_without_a_judge_and_excluded_from_totals():
    cases = load_cases()
    result = run_suite("rules-v1", models.rules_v1, cases)
    judged = [c for c in cases if c["scorer"] == "llm_judge"]
    assert result["skipped"] == len(judged) > 0
    assert result["total"] == len(cases) - len(judged)
    assert all(r["skipped"] for r in result["results"] if r["id"].startswith("judge-"))


def test_judge_cases_run_when_judge_provided():
    judge = {"judge": lambda prompt: '{"score": 10, "reason": "ok"}'}
    result = run_suite("rules-v1", models.rules_v1, load_cases(), ctx=judge)
    assert result["skipped"] == 0


def test_percentile_nearest_rank():
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 100
    assert percentile([], 50) is None


def test_telemetry_tokens_latency_cost_and_categories():
    def model(prompt):
        return ModelResponse("42", prompt_tokens=100, completion_tokens=50)

    cases = [{"id": f"c{i}", "category": "math", "input": "x", "expected": "42",
              "scorer": "exact_match"} for i in range(4)]
    result = run_suite("m", model, cases, config={"temperature": 0.2}, pricing=(1.0, 2.0))
    t = result["telemetry"]
    assert (t["prompt_tokens"], t["completion_tokens"]) == (400, 200)
    assert t["cost_usd"] == pytest.approx(400 * 1.0 / 1e6 + 200 * 2.0 / 1e6)
    assert t["latency_p95_ms"] >= t["latency_p50_ms"] >= 0
    assert result["by_category"] == {"math": {"passed": 4, "total": 4}}
    assert result["config"] == {"temperature": 0.2}


def test_cost_is_none_without_pricing_or_token_usage():
    result = run_suite("m", lambda p: "42", load_cases()[:2])
    assert result["telemetry"]["cost_usd"] is None
    assert result["telemetry"]["prompt_tokens"] is None


def test_openai_adapter_records_usage_and_sampling_params():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "42"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = OpenAICompatModel("m1", temperature=0.7, top_p=0.9, seed=11, max_tokens=64,
                              system="sys", client=client)
    response = model("hi")
    assert (response.prompt_tokens, response.completion_tokens) == (12, 3)
    body = seen["body"]
    assert (body["temperature"], body["top_p"], body["seed"], body["max_tokens"]) == (0.7, 0.9, 11, 64)
    assert model.config["seed"] == 11 and model.config["system_prompt"] == "sys"


@pytest.mark.parametrize("line,message", [
    ("not json", "line 1: invalid JSON"),
    ('["a"]', "must be a JSON object"),
    ('{"id": "x", "input": "q"}', "missing required field 'scorer'"),
    ('{"id": "x", "input": "q", "scorer": "nope"}', "unknown scorer"),
    ('{"id": "x", "input": "q", "scorer": "exact_match"}', "requires 'expected'"),
    ('{"id": "x", "input": "q", "scorer": "regex"}', "requires params.pattern"),
    ('{"id": "x", "input": "q", "scorer": "llm_judge", "params": {}}', "requires params.rubric"),
])
def test_load_cases_rejects_malformed_suites_with_clear_messages(tmp_path, line, message):
    path = tmp_path / "bad.jsonl"
    path.write_text(line + "\n")
    with pytest.raises(ValueError, match=message):
        load_cases(path)


def test_load_cases_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "dup.jsonl"
    row = '{"id": "a", "input": "q", "scorer": "exact_match", "expected": "1"}\n'
    path.write_text(row * 2)
    with pytest.raises(ValueError, match="unique"):
        load_cases(path)


def test_adapter_does_not_retry_a_daily_quota(monkeypatch):
    sleeps = []
    monkeypatch.setattr("llm_eval.models.time.sleep", sleeps.append)
    body = '{"error": {"details": [{"retryDelay": "29919s"}]}}'
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(429, text=body)))
    with pytest.raises(httpx.HTTPStatusError, match="429"):
        OpenAICompatModel("m", client=client)("q")
    assert sleeps == []
