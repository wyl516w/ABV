"""Tests for CodecARTransformer."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.lm.codec_ar_transformer import CodecARTransformer


def test_forward_shapes() -> None:
    model = CodecARTransformer([8, 8])
    prefix = [torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3, dtype=torch.long)]
    ctx = torch.randn(2, 3, 64)
    output = model(prefix, ctx)
    assert len(output["logits"]) == 2
    assert output["logits"][0].shape == (2, 3, 8)
    assert len(output["cache"]) == 2
