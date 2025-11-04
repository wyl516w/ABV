"""Tests for offline tokenization script."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from codec.offline_tokenize import tokenize_dataset


def test_tokenize_dataset(tmp_path: Path) -> None:
    index = tmp_path / "index.jsonl"
    entries = [
        {"utt_id": "u1", "audio_clean": "clean.wav"},
        {"utt_id": "u2", "audio_ac": "noisy.wav"},
        {"utt_id": "u3"},
    ]
    with index.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    output = tmp_path / "out.jsonl"
    tokenize_dataset(index, output)
    contents = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 3
    summary = json.loads((output.with_suffix(".summary.json")).read_text(encoding="utf-8"))
    assert summary["success"] == 1
    assert summary["fallback"] == 2
