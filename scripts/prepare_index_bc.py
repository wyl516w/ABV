"""Script to prepare BC dataset index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def scan_directory(root: Path) -> list[dict[str, str | None]]:
    entries = []
    for idx, path in enumerate(sorted(root.glob("*.wav"))):
        entry = {
            "utt_id": f"bc-{idx}",
            "audio_bc": str(path),
        }
        entries.append(entry)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BC index")
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    entries = scan_directory(args.root)
    with args.output.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
