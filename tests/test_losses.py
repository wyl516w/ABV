"""Tests for training losses."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from train.losses import codec_ce_loss, spectral_loss, uncertainty_calibration_loss


def test_codec_ce_loss() -> None:
    logits = [torch.randn(2, 3, 5)]
    targets = [torch.randint(0, 5, (2, 3))]
    loss = codec_ce_loss(logits, targets)
    assert loss >= 0


def test_spectral_loss() -> None:
    pred = torch.randn(2, 160)
    target = torch.randn(2, 160)
    loss = spectral_loss(pred, target)
    assert loss >= 0


def test_uncertainty_calibration_loss() -> None:
    loss = uncertainty_calibration_loss(torch.zeros(2, 3), torch.ones(2, 3))
    assert loss > 0
