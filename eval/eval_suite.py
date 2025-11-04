"""Evaluation suite for unified auditory token space."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

from eval.metrics import calibration_metrics, reliability_curve, speaker_similarity, stoi_placeholder, wer


@dataclass
class EvalConfig:
    output_dir: Path


def run_eval(samples: List[Dict[str, torch.Tensor]], config: EvalConfig) -> Path:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for sample in samples:
        report = {
            "utt_id": sample["utt_id"],
            "wer": wer([sample["hyp"]], [sample["ref"]]),
            "stoi": stoi_placeholder(sample["pred_audio"], sample["ref_audio"]),
            "speaker_sim": speaker_similarity(sample["pred_emb"], sample["ref_emb"]),
        }
        reports.append(report)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(reports), encoding="utf-8")
    return report_path


__all__ = ["run_eval", "EvalConfig"]
