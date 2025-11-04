"""Tests for offline synthesis."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from codec.codec_adapter import AudioCodecAdapter
from infer.synthesize import SynthesisConfig, synthesize
from models.lm.codec_ar_transformer import CodecARTransformer


def test_synthesize_deterministic() -> None:
    codec = AudioCodecAdapter(has_weights=True)
    model = CodecARTransformer([8, 8])
    result_a = synthesize({}, model, codec, SynthesisConfig(max_tokens=2), seed=0)
    result_b = synthesize({}, model, codec, SynthesisConfig(max_tokens=2), seed=0)
    assert torch.equal(result_a["waveform"], result_b["waveform"])
