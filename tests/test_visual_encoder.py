"""Tests for VisualEncoder."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.encoders.vis import VisualEncoder


def test_forward_shapes() -> None:
    encoder = VisualEncoder()
    video = torch.randn(2, 4, 3, 96, 96)
    output = encoder(video)
    assert output["feat"].shape[:2] == (2, output["uncertainty"].shape[1])


def test_forward_none() -> None:
    encoder = VisualEncoder()
    output = encoder(None)
    assert output["feat"] is None
