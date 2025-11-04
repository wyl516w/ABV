"""Prepare index files for the LRS2 audio-visual dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from scripts.index_utils import write_index

VIDEO_EXTS = {".mp4", ".mkv"}
AUDIO_EXTS = {".wav", ".flac", ".m4a"}


def _collect_videos(root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        key = path.relative_to(root).with_suffix("").as_posix()
        mapping[key] = path
    return mapping


def _find_audio(video_path: Path) -> Path | None:
    for ext in AUDIO_EXTS:
        candidate = video_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def _read_transcript(video_path: Path) -> str | None:
    txt_path = video_path.with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    return None


def build_lrs2_index(
    root: Path,
    *,
    split: str = "train",
    subset_root: Path | None = None,
) -> list[dict[str, object]]:
    base = subset_root if subset_root else root / split
    if not base.exists():
        raise FileNotFoundError(f"Split directory not found: {base}")

    videos = _collect_videos(base)
    entries: list[dict[str, object]] = []
    for key, video_path in videos.items():
        audio_path = _find_audio(video_path)
        transcript = _read_transcript(video_path)
        speaker = key.split("/")[0]
        entries.append(
            {
                "utt_id": f"lrs2-{split}-{key.replace('/', '_')}",
                "audio_ac": str(audio_path) if audio_path else None,
                "video": str(video_path),
                "text": transcript,
                "spk_id": speaker,
                "meta": {"dataset": "LRS2", "split": split},
            }
        )
    return entries


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Prepare LRS2 dataset index")
    parser.add_argument("root", type=Path, help="LRS2 dataset root")
    parser.add_argument("output", type=Path, help="Output JSONL index path")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--subset-root", type=Path, default=None, help="Optional explicit split directory")
    args = parser.parse_args()

    entries = build_lrs2_index(args.root, split=args.split, subset_root=args.subset_root)
    write_index(entries, args.output)
    print(f"Wrote {len(entries)} LRS2 entries to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["build_lrs2_index"]
