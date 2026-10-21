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
