"""Evaluation metrics scaffolding."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor


def wer(hyp: List[str], ref: List[str]) -> float:
    if len(hyp) != len(ref):
        raise ValueError("hyp and ref must match in length")
    return sum(h != r for h, r in zip(hyp, ref)) / max(1, len(ref))


def stoi_placeholder(pred: Tensor, target: Tensor) -> float:
    return float(torch.cosine_similarity(pred, target, dim=-1).mean().item())


def reliability_curve(errors: Tensor, uncertainties: Tensor) -> Dict[str, Tensor]:
    sorted_unc, indices = torch.sort(uncertainties)
    sorted_err = errors[indices]
    cumulative = torch.cumsum(sorted_err, dim=0) / torch.arange(1, len(sorted_err) + 1)
    return {"uncertainty": sorted_unc, "error": cumulative}


def calibration_metrics(pred: Tensor, target: Tensor) -> Dict[str, float]:
    ece = torch.abs(pred - target).mean().item()
    return {"ece": ece, "ace": ece / 2}


def speaker_similarity(emb_a: Tensor, emb_b: Tensor) -> float:
    return float(torch.cosine_similarity(emb_a, emb_b).mean().item())


__all__ = [
    "wer",
    "stoi_placeholder",
    "reliability_curve",
    "calibration_metrics",
    "speaker_similarity",
]
