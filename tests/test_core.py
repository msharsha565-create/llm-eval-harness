import json

import httpx
import pytest

from llm_eval import models
from llm_eval.models import OpenAICompatModel, resolve_model
from llm_eval.report import render_report
from llm_eval.runner import compare, load_cases, run_suite
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
    value, _ = score(output, case(scorer, expected, **params))
    assert (value == 1.0) is passed


def test_unknown_scorer():
    with pytest.raises(ValueError):
        score("x", {"scorer": "nope"})


def test_sample_suite_loads_with_unique_ids():
    cases = load_cases()
    assert len(cases) >= 10
    assert len({c["id"] for c in cases}) == len(cases)


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
    assert model("What is 6*7?") == "42"
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "Be brief"}
    assert seen["body"]["temperature"] == 0


def test_adapter_http_error_becomes_failed_case():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    model = OpenAICompatModel("m", client=client)
    result = run_suite("remote", model, load_cases()[:1])
    assert result["results"][0]["passed"] is False
    assert "HTTPStatusError" in result["results"][0]["detail"]


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
