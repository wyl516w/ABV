"""Tests for codec adapter."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from codec.codec_adapter import AudioCodecAdapter, MissingWeightsError


def test_missing_weights_error() -> None:
    adapter = AudioCodecAdapter()
    with pytest.raises(MissingWeightsError):
        adapter.encode(torch.zeros(160))


def test_encode_decode_roundtrip() -> None:
    adapter = AudioCodecAdapter(has_weights=True)
    tokens = adapter.encode(torch.zeros(320))
    waveform = adapter.decode(tokens)
    assert tokens[0].shape[0] == 1  # 20ms hop -> 1 token for 320 samples at 16k
    assert waveform.shape[0] == 320
