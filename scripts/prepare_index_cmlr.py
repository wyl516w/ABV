"""Prepare index files for the CMLR lip-reading dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from scripts.index_utils import write_index

VIDEO_EXTS = {".mp4", ".avi"}
AUDIO_EXTS = {".wav", ".flac"}


def _scan_dir(root: Path, extensions: set[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not root or not root.exists():
        return mapping
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        key = path.relative_to(root).with_suffix("").as_posix()
        mapping[key] = path
    return mapping


def _default_dir(root: Path, split: str, candidates: list[str]) -> Path | None:
    for candidate in candidates:
        directory = root / split / candidate
        if directory.exists():
            return directory
    return None


def build_cmlr_index(
    root: Path,
    *,
    split: str = "train",
    transcripts_dir: Path | None = None,
    audio_dir: Path | None = None,
    video_dir: Path | None = None,
) -> list[dict[str, object]]:
    transcripts = {}
    if transcripts_dir is None:
        transcripts_dir = _default_dir(root, split, ["txt", "text", "label"])
    if transcripts_dir and transcripts_dir.exists():
        for path in transcripts_dir.rglob("*.txt"):
            key = path.relative_to(transcripts_dir).with_suffix("").as_posix()
            transcripts[key] = path.read_text(encoding="utf-8").strip()
    else:
        transcripts = {}

    audio_dir = audio_dir or _default_dir(root, split, ["wav", "audio"])
    video_dir = video_dir or _default_dir(root, split, ["mp4", "video"])

    audio_files = _scan_dir(audio_dir, AUDIO_EXTS)
    video_files = _scan_dir(video_dir, VIDEO_EXTS)
    keys = sorted(set(audio_files) | set(video_files) | set(transcripts))

    entries: list[dict[str, object]] = []
    for key in keys:
        audio_path = audio_files.get(key)
        video_path = video_files.get(key)
        text = transcripts.get(key) or transcripts.get(key.split("/")[-1])
        speaker = key.split("/")[0]
        entries.append(
            {
                "utt_id": f"cmlr-{split}-{key.replace('/', '_')}",
                "audio_ac": str(audio_path) if audio_path else None,
                "video": str(video_path) if video_path else None,
                "text": text,
                "spk_id": speaker,
                "meta": {"dataset": "CMLR", "split": split},
            }
        )
    return entries


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Prepare CMLR dataset index")
    parser.add_argument("root", type=Path, help="CMLR dataset root")
    parser.add_argument("output", type=Path, help="Output JSONL index path")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--transcripts-dir", type=Path, default=None, help="Directory containing text files")
    parser.add_argument("--audio-dir", type=Path, default=None, help="Directory containing audio files")
    parser.add_argument("--video-dir", type=Path, default=None, help="Directory containing video files")
    args = parser.parse_args()

    entries = build_cmlr_index(
        args.root,
        split=args.split,
        transcripts_dir=args.transcripts_dir,
        audio_dir=args.audio_dir,
        video_dir=args.video_dir,
    )
    write_index(entries, args.output)
    print(f"Wrote {len(entries)} CMLR entries to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["build_cmlr_index"]
