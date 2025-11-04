"""Tests for ReliabilityFusion."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.fusion.reliability_poe import ReliabilityFusion


def test_weight_monotonicity() -> None:
    fusion = ReliabilityFusion(alpha=1.0)
    feats = [torch.ones(2, 3, 4), torch.ones(2, 3, 4)]
    unc_low = torch.full((2, 3), 0.1)
    unc_high = torch.full((2, 3), 0.9)
    result = fusion(feats, [unc_low, unc_high])
    weights = result["weights"]
    assert (weights[:, 0] > weights[:, 1]).all()


def test_temperature_setter() -> None:
    fusion = ReliabilityFusion()
    fusion.set_temperature(0.5)
    assert fusion.temperature == 0.5
