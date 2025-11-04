"""Tests for streaming inference."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from infer.stream_infer import StreamingState
from models.lm.codec_ar_transformer import CodecARTransformer


def test_streaming_step() -> None:
    model = CodecARTransformer([8, 8])
    state = StreamingState(model)
    fused_ctx = torch.zeros(1, 1, 64)
    tokens = state.step(fused_ctx)
    assert len(tokens) == 2
