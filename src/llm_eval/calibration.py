"""Calibrate judge or similarity scores against human labels (needs PyTorch).

A raw score such as "judge gave 7/10" or "cosine similarity 0.81" is not a probability. This module
fits a two-parameter logistic map ``p(correct) = sigmoid(a * (score - b))`` (Platt scaling) with
PyTorch, then reports Expected Calibration Error (ECE) and the Brier score on held-out cases, so
you can see how far a threshold-based pass/fail can be trusted.

It needs ground truth: a mapping of case id -> bool you labelled yourself. The bundled suite ships
no human labels, so the demo in the tests uses synthetic scores and is labelled as such.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any


class MissingTorch(RuntimeError):
    """PyTorch is not installed."""


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise MissingTorch("calibration needs PyTorch: pip install 'llm-eval-harness[ml]'") from exc
    return torch


@dataclass
class Calibrator:
    """p(correct) = sigmoid(a * (score - b)); `scale` rescales raw scores (e.g. 10 for 0-10)."""

    a: float
    b: float
    scale: float = 1.0

    def predict(self, scores: list[float]) -> list[float]:
        import math

        return [1 / (1 + math.exp(-self.a * (s / self.scale - self.b))) for s in scores]


def fit(scores: list[float], labels: list[bool], scale: float = 1.0, steps: int = 500,
        seed: int = 0) -> Calibrator:
    """Fit the logistic map by minimising binary cross-entropy with L-BFGS."""
    torch = _torch()
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be the same non-zero length")
    if len(set(labels)) < 2:
        raise ValueError("need both passing and failing labels to calibrate")
    torch.manual_seed(seed)
    x = torch.tensor([s / scale for s in scores], dtype=torch.float64)
    y = torch.tensor([1.0 if v else 0.0 for v in labels], dtype=torch.float64)
    a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(float(x.mean()), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([a, b], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * (x - b), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return Calibrator(a=float(a), b=float(b), scale=scale)


def brier(probs: list[float], labels: list[bool]) -> float:
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in zip(probs, labels, strict=True)) / len(probs)


def ece(probs: list[float], labels: list[bool], bins: int = 10) -> float:
    """Expected Calibration Error: weighted gap between confidence and accuracy per bin."""
    total = len(probs)
    error = 0.0
    for k in range(bins):
        lo, hi = k / bins, (k + 1) / bins
        members = [(p, y) for p, y in zip(probs, labels, strict=True)
                   if (lo <= p < hi) or (k == bins - 1 and p == 1.0)]
        if members:
            confidence = sum(p for p, _ in members) / len(members)
            accuracy = sum(1.0 for _, y in members if y) / len(members)
            error += len(members) / total * abs(confidence - accuracy)
    return error


def naive_probs(scores: list[float], scale: float) -> list[float]:
    """The uncalibrated baseline: treat the normalised score itself as a probability."""
    return [min(1.0, max(0.0, s / scale)) for s in scores]


def split(n: int, test_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    cut = max(1, int(n * test_fraction))
    return indices[cut:], indices[:cut]


def calibration_report(scores: list[float], labels: list[bool], scale: float = 1.0,
                       test_fraction: float = 0.3, seed: int = 0) -> dict[str, Any]:
    """Fit on a train split and report ECE and Brier before and after on the held-out split."""
    if len(scores) < 10:
        raise ValueError("need at least 10 labelled cases for a train/test split")
    train, test = split(len(scores), test_fraction, seed)
    model = fit([scores[i] for i in train], [labels[i] for i in train], scale=scale, seed=seed)
    test_scores = [scores[i] for i in test]
    test_labels = [labels[i] for i in test]
    before = naive_probs(test_scores, scale)
    after = model.predict(test_scores)
    return {
        "calibrator": asdict(model),
        "train_size": len(train),
        "test_size": len(test),
        "before": {"ece": round(ece(before, test_labels), 4),
                   "brier": round(brier(before, test_labels), 4)},
        "after": {"ece": round(ece(after, test_labels), 4),
                  "brier": round(brier(after, test_labels), 4)},
    }


def from_run(run: dict[str, Any], labels: dict[str, bool], scale: float = 1.0) -> tuple[list[float], list[bool]]:
    """Pair each non-skipped result's raw score with your human label for that case id."""
    scores, truth = [], []
    for r in run["results"]:
        if r.get("skipped") or r["id"] not in labels:
            continue
        scores.append(float(r["score"]))
        truth.append(bool(labels[r["id"]]))
    return scores, truth
