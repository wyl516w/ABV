"""Tests for BaseMultiModalDataset."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

torch = pytest.importorskip("torch")

from datasets.base_dataset import BaseMultiModalDataset


class DummyDataset(BaseMultiModalDataset):
    def _load_audio(self, source: Any, sr: int) -> Optional[torch.Tensor]:
        if isinstance(source, (list, tuple)):
            return torch.tensor(source, dtype=torch.float32)
        if isinstance(source, torch.Tensor):
            return source.float()
        raise TypeError("Unsupported audio source")

    def _load_video(self, source: Any, fps: int, size: tuple[int, int]) -> Optional[torch.Tensor]:
        if isinstance(source, torch.Tensor):
            return source.float()
        if isinstance(source, dict) and "tensor" in source:
            return source["tensor"].float()
        raise TypeError("Unsupported video source")

    def _load_text(self, source: Any) -> Optional[torch.Tensor]:
        return torch.tensor(source, dtype=torch.long)

    def _load_codec_tokens(self, source: Any) -> Optional[List[torch.Tensor]]:
        return [torch.tensor(tokens, dtype=torch.long) for tokens in source]


@pytest.fixture
def sample_items() -> List[Dict[str, Any]]:
    video = torch.zeros(2, 3, 96, 96)
    return [
        {
            "utt_id": "utt-1",
            "audio_ac": [0.0, 0.1, -0.1],
            "video": {"tensor": video},
            "codec_tokens": [[1, 2, 3], [0, 1, 1]],
        },
        {
            "utt_id": "utt-2",
            "audio_clean": torch.zeros(4),
            "text": [1, 2],
            "codec_tokens": [[1], [2]],
        },
    ]


def test_len(sample_items: List[Dict[str, Any]]) -> None:
    dataset = DummyDataset(sample_items)
    assert len(dataset) == 2


def test_getitem_shapes(sample_items: List[Dict[str, Any]]) -> None:
    dataset = DummyDataset(sample_items)
    sample = dataset[0]
    assert sample["audio_ac"].shape == (3,)
    assert sample["video"].shape == (2, 3, 96, 96)
    assert len(sample["codec_tokens"]) == 2


def test_missing_audio_requires_codec(sample_items: List[Dict[str, Any]]) -> None:
    dataset = DummyDataset(sample_items)
    sample = dataset[1]
    assert sample["audio_clean"].shape == (4,)
    assert sample["audio_ac"] is None
    assert sample["video"] is None


def test_sanity_check_requires_audio_or_codec(sample_items: List[Dict[str, Any]]) -> None:
    bad = {
        "utt_id": "bad",
    }
    dataset = DummyDataset(sample_items)
    dataset._items.append(bad)  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        _ = dataset[2]
