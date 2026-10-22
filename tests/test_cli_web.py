from starlette.testclient import TestClient

from llm_eval import web
from llm_eval.cli import main


def test_cli_run_compare_report(tmp_path, capsys):
    v1, v2, html = tmp_path / "v1.json", tmp_path / "v2.json", tmp_path / "r.html"
    assert main(["run", "--model", "rules-v1", "--out", str(v1)]) == 0
    assert main(["run", "--model", "rules-v2", "--out", str(v2)]) == 0
    assert main(["compare", str(v1), str(v2)]) == 1  # regressions -> non-zero exit
    assert main(["compare", str(v1), str(v1)]) == 0
    assert main(["report", str(v1), str(v2), "-o", str(html)]) == 0
    assert "regression" in html.read_text()
    assert "regressions: math-div" in capsys.readouterr().out


def test_web_page_and_validation():
    client = TestClient(web.app)
    page = client.get("/")
    assert page.status_code == 200 and "regression" in page.text
    assert client.get("/", params={"a": "nope"}).status_code == 400


def test_report_includes_telemetry_and_categories(tmp_path):
    v1, html = tmp_path / "v1.json", tmp_path / "r.html"
    main(["run", "--model", "rules-v1", "--out", str(v1)])
    main(["report", str(v1), "-o", str(html)])
    text = html.read_text()
    assert "p50" in text and "By category" in text and "json_extraction" in text


def test_api_key_is_read_from_the_named_environment_variable(monkeypatch, tmp_path):
    seen = []

    def fake_resolve(spec, api_key=None, **params):
        seen.append((spec, api_key))
        return lambda prompt: "42"

    monkeypatch.setattr("llm_eval.cli.resolve_model", fake_resolve)
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("OPENAI_API_KEY", "other-key")
    main(["run", "--model", "openai:m@http://x/v1", "--api-key-env", "GEMINI_API_KEY",
          "--out", str(tmp_path / "r.json")])
    assert seen == [("openai:m@http://x/v1", "gem-key")]
    seen.clear()
    main(["run", "--model", "openai:m@http://x/v1", "--out", str(tmp_path / "r.json")])
    assert seen[0][1] == "other-key"
