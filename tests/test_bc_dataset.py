"""Tests for the bone-conduction dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

from datasets.bc_dataset import BCDataset


@pytest.fixture
def metadata_items() -> List[Dict[str, Any]]:
    return [
        {
            "utt_id": "bc1",
            "audio_bc": torch.ones(5),
        },
        {
            "utt_id": "bc2",
            "audio_ac": torch.zeros(3),
            "codec_tokens": [[1, 0]],
        },
    ]


def test_sample_contains_bc(metadata_items: List[Dict[str, Any]]) -> None:
    dataset = BCDataset(metadata_items)
    assert dataset[0]["audio_bc"].shape == (5,)


def test_optional_fields(metadata_items: List[Dict[str, Any]]) -> None:
    dataset = BCDataset(metadata_items)
    assert dataset[1]["audio_bc"] is None
    assert dataset[1]["codec_tokens"][0].shape[0] == 2


def test_requires_audio_or_codec(metadata_items: List[Dict[str, Any]]) -> None:
    bad = {"utt_id": "bad"}
    dataset = BCDataset(metadata_items + [bad])
    with pytest.raises(ValueError):
        _ = dataset[2]


def test_codec_tokens_loaded_from_path(tmp_path: Path) -> None:
    token_file = tmp_path / "tokens.json"
    token_file.write_text("[[1, 2, 3]]", encoding="utf-8")
    metadata = {
        "utt_id": "codec", 
        "audio_bc": torch.ones(2),
        "codec_tokens": str(token_file),
    }
    dataset = BCDataset([metadata])
    sample = dataset[0]
    assert sample["codec_tokens"][0].tolist() == [1, 2, 3]
