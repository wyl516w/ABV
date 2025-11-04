"""Training losses for the unified model."""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn, Tensor


def codec_ce_loss(logits: List[Tensor], targets: List[Tensor]) -> Tensor:
    losses = []
    for logit, target in zip(logits, targets):
        losses.append(nn.functional.cross_entropy(logit.view(-1, logit.shape[-1]), target.view(-1), ignore_index=0))
    return torch.stack(losses).mean()


def spectral_loss(pred: Tensor, target: Tensor) -> Tensor:
    pred_spec = torch.fft.rfft(pred, dim=-1)
    target_spec = torch.fft.rfft(target, dim=-1)
    return torch.mean(torch.abs(pred_spec - target_spec))


def uncertainty_calibration_loss(uncertainty: Tensor, reference: Tensor) -> Tensor:
    return nn.functional.mse_loss(uncertainty, reference)


def domain_alignment_loss(feat: Tensor, domain_labels: Tensor, classifier: nn.Module) -> Tensor:
    logits = classifier(feat)
    return nn.functional.cross_entropy(logits, domain_labels)


__all__ = [
    "codec_ce_loss",
    "spectral_loss",
    "uncertainty_calibration_loss",
    "domain_alignment_loss",
]
