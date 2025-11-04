"""Utility helpers shared by dataset index preparation scripts."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def load_transcripts(path: Optional[Path]) -> Dict[str, str]:
    """Load a mapping from utterance id to transcript string.

    The helper accepts CSV/TSV (columns ``utt_id`` and ``text``), JSON/JSONL
    files following the same schema, or plain text with ``utt_id|transcript``
    per line. Missing or empty files return an empty mapping.
    """

    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")

    suffix = path.suffix.lower()
    transcripts: Dict[str, str] = {}

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                utt = str(record["utt_id"])
                transcripts[utt] = str(record.get("text", "")).strip()
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            iterable: Iterable = payload.items()
        else:
            iterable = payload
        for item in iterable:
            if isinstance(item, dict):
                utt = str(item["utt_id"])
                text = str(item.get("text", ""))
            else:
                utt, text = item
            transcripts[str(utt)] = str(text).strip()
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if "utt_id" not in reader.fieldnames or "text" not in reader.fieldnames:
                raise ValueError("Transcript CSV/TSV must contain 'utt_id' and 'text' columns")
            for row in reader:
                utt = str(row["utt_id"])
                transcripts[utt] = str(row.get("text", "")).strip()
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if "|" in line:
                    utt, text = line.split("|", 1)
                else:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 1:
                        utt, text = parts[0], ""
                    else:
                        utt, text = parts
                transcripts[str(utt)] = text.strip()
    return transcripts


def write_index(entries: Iterable[Dict[str, object]], output: Path) -> None:
    """Write entries to ``output`` as JSONL."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def guess_transcript_path(root: Path, candidates: Optional[List[str]] = None) -> Optional[Path]:
    """Return the first transcript file found under ``root``."""

    search = candidates or [
        "transcripts.jsonl",
        "transcripts.json",
        "transcripts.csv",
        "transcripts.tsv",
        "metadata.csv",
        "metadata.json",
        "text.txt",
    ]
    for name in search:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None
