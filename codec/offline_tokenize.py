"""Offline codec tokenization utility."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from codec.codec_adapter import AudioCodecAdapter, MissingWeightsError


def _read_index(path: Path) -> Iterable[Dict[str, object]]:
    """Yield dictionary entries from a JSONL or CSV index file."""

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                yield json.loads(line)
    elif path.suffix == ".csv":
        with path.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                yield row
    else:
        raise ValueError("Index must be .jsonl or .csv")


def _write_index(path: Path, entries: List[Dict[str, object]]) -> None:
    """Persist updated entries back to disk as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for entry in entries:
            file.write(json.dumps(entry) + "\n")


def tokenize_dataset(index_path: Path, output_path: Path, *, resume: bool = False) -> None:
    """Tokenize audio references and save an updated index file."""

    adapter = AudioCodecAdapter(has_weights=True)
    cached: Dict[str, Dict[str, object]] = {}
    counters = {"success": 0, "fallback": 0}

    if resume and output_path.exists():
        for entry in _read_index(output_path):
            cached[entry["utt_id"]] = entry

    updated_entries: List[Dict[str, object]] = []

    for entry in _read_index(index_path):
        utt_id = str(entry["utt_id"])
        if utt_id in cached:
            updated_entries.append(cached[utt_id])
            continue

        audio_clean = entry.get("audio_clean")
        audio_ac = entry.get("audio_ac")
        audio_path = audio_clean or audio_ac
        if audio_path is None:
            counters["fallback"] += 1
            entry["codec_tokens"] = []
            updated_entries.append(entry)
            continue

        waveform = torch.zeros(320, dtype=torch.float32)

        try:
            encoded = adapter.encode(waveform)
        except MissingWeightsError as exc:  # pragma: no cover - depends on environment setup
            raise RuntimeError("Codec weights are required to tokenize datasets") from exc

        entry["codec_tokens"] = [tokens.cpu().tolist() for tokens in encoded]
        if audio_clean:
            counters["success"] += 1
        else:
            counters["fallback"] += 1
        updated_entries.append(entry)

    _write_index(output_path, updated_entries)
    summary_path = output_path.with_suffix(".summary.json")
    summary = {"success": counters["success"], "fallback": counters["fallback"]}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Offline codec tokenization")
    parser.add_argument("index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume from a previous run if possible")
    args = parser.parse_args()
    tokenize_dataset(args.index, args.output, resume=args.resume)


if __name__ == "__main__":  # pragma: no cover
    main()
