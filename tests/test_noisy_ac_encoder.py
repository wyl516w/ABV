"""Tests for NoisyACEncoder."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.encoders.noisy_ac import NoisyACEncoder


def test_forward_shapes() -> None:
    encoder = NoisyACEncoder()
    audio = torch.randn(2, 320)
    output = encoder(audio)
    assert output["feat"].shape[0] == 2
    assert output["uncertainty"].shape[1] == output["feat"].shape[1]
    assert torch.isfinite(output["uncertainty"]).all()


def test_forward_none() -> None:
    encoder = NoisyACEncoder()
    output = encoder(None)
    assert output["feat"] is None
    assert output["uncertainty"] is None
