"""Prepare a generic audio-visual dataset index."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from scripts.index_utils import guess_transcript_path, load_transcripts, write_index

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}
AUDIO_EXTS = {".wav", ".flac", ".m4a"}


def _collect_files(root: Path, extensions: set[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        key = path.relative_to(root).with_suffix("").as_posix()
        mapping[key] = path
    return mapping


def _lookup_transcript(transcripts: Dict[str, str], key: str) -> str | None:
    candidates = [key, key.split("/")[-1]]
    for candidate in candidates:
        if candidate in transcripts:
            return transcripts[candidate]
    return None


def _lookup_path(mapping: Dict[str, Path], key: str) -> Path | None:
    if key in mapping:
        return mapping[key]
    tail = key.split("/")[-1]
    if tail in mapping:
        return mapping[tail]
    for candidate_key, path in mapping.items():
        if candidate_key.endswith(f"/{tail}"):
            return path
    return None


def build_generic_av_index(root: Path, dataset_name: str, transcripts_path: Path | None) -> list[dict[str, object]]:
    transcripts = load_transcripts(transcripts_path or guess_transcript_path(root))
    audio_files = _collect_files(root, AUDIO_EXTS)
    video_files = _collect_files(root, VIDEO_EXTS)
    all_keys = sorted(set(audio_files) | set(video_files))

    entries: list[dict[str, object]] = []
    for key in all_keys:
        audio_path = _lookup_path(audio_files, key)
        video_path = _lookup_path(video_files, key)
        relative_parts = key.split("/")
        split = relative_parts[0] if len(relative_parts) > 1 else "unspecified"
        utt_id = f"{dataset_name}-{key.replace('/', '_')}"
        entries.append(
            {
                "utt_id": utt_id,
                "audio_ac": str(audio_path) if audio_path else None,
                "video": str(video_path) if video_path else None,
                "text": _lookup_transcript(transcripts, key),
                "meta": {"dataset": dataset_name, "split": split},
            }
        )
    return entries


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Prepare a generic AV dataset index")
    parser.add_argument("root", type=Path, help="Dataset root containing audio/video files")
    parser.add_argument("output", type=Path, help="Destination JSONL index path")
    parser.add_argument("--dataset-name", default="generic", help="Name to record in metadata")
    parser.add_argument("--transcripts", type=Path, default=None, help="Optional transcript file")
    args = parser.parse_args()

    entries = build_generic_av_index(args.root, args.dataset_name, args.transcripts)
    write_index(entries, args.output)
    print(f"Wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["build_generic_av_index"]
