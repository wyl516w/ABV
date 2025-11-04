"""Tests for collate_batch."""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

from datasets.collate_fn import collate_batch


def make_sample(audio_len: int, video_len: int, text_len: int) -> Dict[str, Any]:
    audio = torch.arange(audio_len, dtype=torch.float32)
    video = torch.zeros(video_len, 3, 96, 96) if video_len > 0 else None
    text = torch.arange(text_len, dtype=torch.long) if text_len > 0 else None
    codec = [torch.arange(audio_len // 2 + 1, dtype=torch.long)] if audio_len > 0 else None
    return {
        "utt_id": "id",
        "audio_sr": 16000,
        "video_fps": 30,
        "audio_ac": audio,
        "audio_bc": None,
        "audio_clean": None,
        "video": video,
        "text": text,
        "codec_tokens": codec,
        "spk_id": None,
        "spk_emb": None,
        "vad_mask": None,
        "align": None,
        "meta": None,
    }


def test_collate_pads_audio_and_video() -> None:
    samples = [make_sample(4, 2, 3), make_sample(2, 0, 0)]
    batch = collate_batch(samples)
    assert batch["audio_ac"].shape == (2, 4)
    assert batch["video"].shape == (2, 2, 3, 96, 96)


def test_text_padding() -> None:
    samples = [make_sample(4, 2, 2), make_sample(4, 2, 3)]
    batch = collate_batch(samples)
    assert batch["text"].shape == (2, 3)


def test_codec_padding_handles_missing() -> None:
    sample_a = make_sample(4, 2, 2)
    sample_b = make_sample(0, 0, 0)
    sample_b["codec_tokens"] = None
    batch = collate_batch([sample_a, sample_b])
    assert batch["codec_tokens"][0].shape == (2, 3)
