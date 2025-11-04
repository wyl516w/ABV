"""Tests for the audio-visual dataset."""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

from datasets.av_dataset import AVDataset


def random_crop(audio: torch.Tensor, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    half = audio.shape[0] // 2
    if half == 0:
        return audio, video
    return audio[:half], video[:half]


@pytest.fixture
def metadata_items() -> List[Dict[str, Any]]:
    video = torch.zeros(4, 3, 96, 96)
    return [
        {
            "utt_id": "a",
            "audio_ac": torch.linspace(-1.0, 1.0, 8),
            "video": video,
            "codec_tokens": [[1, 2], [0, 1]],
        },
        {
            "utt_id": "b",
            "audio_ac": torch.zeros(6),
        },
        {
            "utt_id": "c",
            "video": video.clone(),
        },
    ]


def test_sample_shapes(metadata_items: List[Dict[str, Any]]) -> None:
    dataset = AVDataset(metadata_items)
    sample = dataset[0]
    assert sample["audio_ac"].shape[0] == 8
    assert sample["video"].shape == (4, 3, 96, 96)


def test_random_crop_applied(metadata_items: List[Dict[str, Any]]) -> None:
    dataset = AVDataset(metadata_items, random_crop=random_crop)
    sample = dataset[0]
    assert sample["audio_ac"].shape[0] == 4
    assert sample["video"].shape[0] == 2


def test_missing_fields_allowed(metadata_items: List[Dict[str, Any]]) -> None:
    dataset = AVDataset(metadata_items)
    assert dataset[1]["video"] is None
    assert dataset[2]["audio_ac"] is None


def test_requires_audio_or_video(metadata_items: List[Dict[str, Any]]) -> None:
    bad = {"utt_id": "bad"}
    dataset = AVDataset(metadata_items + [bad])
    with pytest.raises(ValueError):
        _ = dataset[3]
