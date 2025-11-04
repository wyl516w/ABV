"""Tests for BoneCondEncoder."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.encoders.bc import BoneCondEncoder


def test_forward_and_loss() -> None:
    encoder = BoneCondEncoder()
    audio = torch.randn(2, 320)
    output = encoder(audio)
    feat = output["feat"]
    loss = encoder.domain_alignment_loss(feat, torch.tensor([0, 1]))
    assert loss >= 0
    assert output["uncertainty"].shape[1] == feat.shape[1]


def test_no_adaptation() -> None:
    encoder = BoneCondEncoder(adaptation=False)
    audio = torch.randn(1, 320)
    feat = encoder(audio)["feat"]
    loss = encoder.domain_alignment_loss(feat, torch.tensor([0]))
    assert loss.item() == 0
