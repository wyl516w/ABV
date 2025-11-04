"""Prepare index files for the ABCS (air & bone-conducted speech) corpus."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

from scripts.index_utils import guess_transcript_path, load_transcripts, write_index

AUDIO_EXTS = {".wav", ".flac"}


def _collect(directory: Path | None) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if directory is None:
        return mapping
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        key = path.relative_to(directory).with_suffix("").as_posix()
        mapping[key] = path
    return mapping


def _detect_directory(root: Path, hints: Iterable[str]) -> Path | None:
    for hint in hints:
        candidate = root / hint
        if candidate.exists():
            return candidate
    return None


def build_abcs_index(
    root: Path,
    *,
    transcripts: Path | None = None,
    ac_dir: Path | None = None,
    bc_dir: Path | None = None,
    video_dir: Path | None = None,
) -> list[dict[str, object]]:
    transcript_file = transcripts or guess_transcript_path(root)
    transcripts_map = load_transcripts(transcript_file) if transcript_file else {}
    ac_dir = ac_dir or _detect_directory(root, ["AC", "ac", "audio_ac", "air"])
    bc_dir = bc_dir or _detect_directory(root, ["BC", "bc", "audio_bc", "bone"])
    if ac_dir is None and bc_dir is None:
        raise ValueError("At least one of AC or BC directories must be provided for ABCS")
    video_dir = video_dir or _detect_directory(root, ["video", "mp4"])

    ac_files = _collect(ac_dir)
    bc_files = _collect(bc_dir)
    video_files: Dict[str, Path] = {}
    if video_dir:
        for path in sorted(video_dir.rglob("*.mp4")):
            key = path.relative_to(video_dir).with_suffix("").as_posix()
            video_files[key] = path

    all_keys = sorted(set(ac_files) | set(bc_files) | set(video_files))
    entries: list[dict[str, object]] = []
    for key in all_keys:
        ac_path = ac_files.get(key)
        bc_path = bc_files.get(key)
        video_path = video_files.get(key)
        speaker = key.split("/")[0]
        transcript = transcripts_map.get(key) or transcripts_map.get(key.split("/")[-1])
        entries.append(
            {
                "utt_id": f"abcs-{key.replace('/', '_')}",
                "audio_ac": str(ac_path) if ac_path else None,
                "audio_bc": str(bc_path) if bc_path else None,
                "video": str(video_path) if video_path else None,
                "text": transcript,
                "spk_id": speaker,
                "meta": {
                    "dataset": "ABCS",
                    "environment": "anechoic_chamber",
                },
            }
        )
    return entries


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Prepare ABCS corpus index")
    parser.add_argument("root", type=Path, help="ABCS dataset root")
    parser.add_argument("output", type=Path, help="Output JSONL index path")
    parser.add_argument("--transcripts", type=Path, default=None, help="Optional transcript file")
    parser.add_argument("--ac-dir", type=Path, default=None, help="Directory containing air-conduction audio")
    parser.add_argument("--bc-dir", type=Path, default=None, help="Directory containing bone-conduction audio")
    parser.add_argument("--video-dir", type=Path, default=None, help="Directory containing optional video files")
    args = parser.parse_args()

    entries = build_abcs_index(
        args.root,
        transcripts=args.transcripts,
        ac_dir=args.ac_dir,
        bc_dir=args.bc_dir,
        video_dir=args.video_dir,
    )
    write_index(entries, args.output)
    print(f"Wrote {len(entries)} ABCS entries to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["build_abcs_index"]
