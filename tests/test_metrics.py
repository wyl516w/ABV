"""Tests for evaluation metrics."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from eval.metrics import calibration_metrics, reliability_curve, speaker_similarity, stoi_placeholder, wer


def test_wer() -> None:
    assert wer(["a", "b"], ["a", "c"]) == 0.5


def test_stoi_placeholder() -> None:
    score = stoi_placeholder(torch.ones(2, 3), torch.ones(2, 3))
    assert score == 1.0


def test_reliability_curve() -> None:
    curve = reliability_curve(torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.3, 0.2, 0.1]))
    assert curve["error"].shape[0] == 3


def test_calibration_metrics() -> None:
    metrics = calibration_metrics(torch.zeros(3), torch.ones(3))
    assert metrics["ece"] > 0


def test_speaker_similarity() -> None:
    sim = speaker_similarity(torch.ones(2, 3), torch.ones(2, 3))
    assert sim == 1.0
