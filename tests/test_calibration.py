import json
import math
import random

import pytest

torch = pytest.importorskip("torch")

from llm_eval import calibration
from llm_eval.cli import main


def synthetic(n=400, seed=0):
    """Synthetic judge scores (0-10) for a judge that is over-confident: true pass rate follows a
    shallower curve than the raw score implies. Not real model data."""
    rng = random.Random(seed)
    scores, labels = [], []
    for _ in range(n):
        score = rng.uniform(0, 10)
        p_true = 1 / (1 + math.exp(-0.6 * (score - 6.0)))
        scores.append(score)
        labels.append(rng.random() < p_true)
    return scores, labels


def test_ece_and_brier_basics():
    assert calibration.ece([1.0, 0.0], [True, False]) == 0.0
    assert calibration.brier([1.0, 0.0], [True, False]) == 0.0
    assert calibration.brier([0.0], [True]) == 1.0


def test_fit_recovers_the_generating_curve():
    scores, labels = synthetic()
    model = calibration.fit(scores, labels, scale=10.0)
    # true curve: a=0.6*10 on the normalised score, b=0.6
    assert model.a == pytest.approx(6.0, rel=0.5)
    assert model.b == pytest.approx(0.6, abs=0.1)


def test_calibration_improves_held_out_error():
    scores, labels = synthetic()
    report = calibration.calibration_report(scores, labels, scale=10.0)
    assert report["after"]["brier"] < report["before"]["brier"]
    assert report["after"]["ece"] < report["before"]["ece"]


def test_needs_both_classes_and_enough_cases():
    with pytest.raises(ValueError):
        calibration.fit([1.0, 2.0], [True, True])
    with pytest.raises(ValueError):
        calibration.calibration_report([1.0] * 5, [True, False] * 2 + [True])


def test_cli_calibrate(tmp_path, capsys):
    scores, labels = synthetic(200)
    run = {"results": [{"id": f"c{i}", "score": s, "skipped": False} for i, s in enumerate(scores)]}
    (tmp_path / "run.json").write_text(json.dumps(run))
    (tmp_path / "labels.json").write_text(json.dumps({f"c{i}": v for i, v in enumerate(labels)}))
    code = main(["calibrate", str(tmp_path / "run.json"), "--labels", str(tmp_path / "labels.json"),
                 "--scale", "10"])
    assert code == 0
    assert "after" in json.loads(capsys.readouterr().out)
