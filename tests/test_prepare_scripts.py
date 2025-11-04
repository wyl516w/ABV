"""Tests for dataset preparation helper scripts."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from scripts.prepare_index_abcs import build_abcs_index
from scripts.prepare_index_av import build_generic_av_index
from scripts.prepare_index_bc import build_generic_bc_index
from scripts.prepare_index_cmlr import build_cmlr_index
from scripts.prepare_index_lrs2 import build_lrs2_index
from scripts.prepare_index_lrs3 import build_lrs3_index


@pytest.fixture()
def av_root(tmp_path: Path) -> Path:
    root = tmp_path / "av"
    audio_dir = root / "audio"
    video_dir = root / "video"
    audio_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    (audio_dir / "sample.wav").write_bytes(b"")
    (video_dir / "sample.mp4").write_bytes(b"")
    (root / "transcripts.txt").write_text("audio/sample|hello", encoding="utf-8")
    return root


def test_build_generic_av_index(av_root: Path) -> None:
    entries = build_generic_av_index(av_root, "generic", av_root / "transcripts.txt")
    assert entries
    sample = entries[0]
    assert sample["audio_ac"].endswith("sample.wav")
    assert sample["video"].endswith("sample.mp4")
    assert sample["text"] == "hello"


def test_build_generic_bc_index(tmp_path: Path) -> None:
    root = tmp_path / "bc"
    bc_dir = root / "bc"
    bc_dir.mkdir(parents=True)
    (bc_dir / "utt.wav").write_bytes(b"")
    (root / "transcripts.txt").write_text("bc/utt|骨传导", encoding="utf-8")
    entries = build_generic_bc_index(root, "generic_bc", root / "transcripts.txt")
    assert entries[0]["audio_bc"].endswith("utt.wav")
    assert entries[0]["text"] == "骨传导"


def test_build_abcs_index(tmp_path: Path) -> None:
    root = tmp_path / "abcs"
    ac_dir = root / "AC"
    bc_dir = root / "BC"
    (ac_dir / "spk1" / "utt1.wav").parent.mkdir(parents=True, exist_ok=True)
    (bc_dir / "spk1" / "utt1.wav").parent.mkdir(parents=True, exist_ok=True)
    (ac_dir / "spk1" / "utt1.wav").write_bytes(b"")
    (bc_dir / "spk1" / "utt1.wav").write_bytes(b"")
    (root / "transcripts.txt").write_text("spk1/utt1|test", encoding="utf-8")
    entries = build_abcs_index(root, transcripts=root / "transcripts.txt")
    sample = entries[0]
    assert sample["spk_id"] == "spk1"
    assert sample["audio_ac"].endswith("utt1.wav")
    assert sample["audio_bc"].endswith("utt1.wav")


def test_build_cmlr_index(tmp_path: Path) -> None:
    root = tmp_path / "cmlr"
    split_dir = root / "train"
    (split_dir / "wav" / "speaker" / "utt.wav").parent.mkdir(parents=True, exist_ok=True)
    (split_dir / "mp4" / "speaker" / "utt.mp4").parent.mkdir(parents=True, exist_ok=True)
    (split_dir / "text" / "speaker" / "utt.txt").parent.mkdir(parents=True, exist_ok=True)
    (split_dir / "wav" / "speaker" / "utt.wav").write_bytes(b"")
    (split_dir / "mp4" / "speaker" / "utt.mp4").write_bytes(b"")
    (split_dir / "text" / "speaker" / "utt.txt").write_text("你好", encoding="utf-8")
    entries = build_cmlr_index(root, split="train")
    sample = entries[0]
    assert sample["spk_id"] == "speaker"
    assert sample["text"] == "你好"


def test_build_lrs2_index(tmp_path: Path) -> None:
    root = tmp_path / "lrs2"
    split_dir = root / "train"
    (split_dir / "speaker" / "utt.mp4").parent.mkdir(parents=True, exist_ok=True)
    video_path = split_dir / "speaker" / "utt.mp4"
    video_path.write_bytes(b"")
    (split_dir / "speaker" / "utt.txt").write_text("hello", encoding="utf-8")
    entries = build_lrs2_index(root, split="train")
    sample = entries[0]
    assert sample["video"].endswith("utt.mp4")
    assert sample["text"] == "hello"


def test_build_lrs3_index(tmp_path: Path) -> None:
    root = tmp_path / "lrs3"
    split_dir = root / "trainval"
    (split_dir / "speaker" / "utt.mp4").parent.mkdir(parents=True, exist_ok=True)
    video_path = split_dir / "speaker" / "utt.mp4"
    video_path.write_bytes(b"")
    (split_dir / "speaker" / "utt.txt").write_text("world", encoding="utf-8")
    entries = build_lrs3_index(root, split="trainval")
    sample = entries[0]
    assert sample["video"].endswith("utt.mp4")
    assert sample["text"] == "world"
