"""Prepare a generic bone-conduction dataset index."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from scripts.index_utils import guess_transcript_path, load_transcripts, write_index

AUDIO_EXTS = {".wav", ".flac"}


def _collect_audio(root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        key = path.relative_to(root).with_suffix("").as_posix()
        mapping[key] = path
    return mapping


def build_generic_bc_index(root: Path, dataset_name: str, transcripts_path: Path | None) -> list[dict[str, object]]:
    transcripts = load_transcripts(transcripts_path or guess_transcript_path(root))
    bc_files = _collect_audio(root)
    entries: list[dict[str, object]] = []
    for key, bc_path in bc_files.items():
        split = key.split("/")[0] if "/" in key else "unspecified"
        utt_id = f"{dataset_name}-{key.replace('/', '_')}"
        entries.append(
            {
                "utt_id": utt_id,
                "audio_bc": str(bc_path),
                "text": transcripts.get(key) or transcripts.get(key.split("/")[-1]),
                "meta": {"dataset": dataset_name, "split": split},
            }
        )
    return entries


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Prepare a generic BC dataset index")
    parser.add_argument("root", type=Path, help="Dataset root with bone-conduction audio files")
    parser.add_argument("output", type=Path, help="Destination JSONL index path")
    parser.add_argument("--dataset-name", default="generic_bc", help="Name to record in metadata")
    parser.add_argument("--transcripts", type=Path, default=None, help="Optional transcript file")
    args = parser.parse_args()

    entries = build_generic_bc_index(args.root, args.dataset_name, args.transcripts)
    write_index(entries, args.output)
    print(f"Wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["build_generic_bc_index"]
