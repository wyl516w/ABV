"""Tests for training loop."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from train.train_loop import TrainerConfig, run_training


def test_run_training() -> None:
    config = TrainerConfig(batch_size=2, steps=1)
    loss = run_training(config)
    assert loss >= 0
