"""Bone conduction encoder with simple domain adaptation hooks."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn, Tensor


class BoneCondEncoder(nn.Module):
    """Encode bone-conduction audio with optional domain adaptation."""

    def __init__(self, hidden_dim: int = 32, adaptation: bool = True) -> None:
        super().__init__()
        self.adaptation = adaptation
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, audio: Optional[Tensor]) -> Dict[str, Tensor | None]:
        if audio is None:
            return {"feat": None, "uncertainty": None, "quality": None}
        if audio.dim() != 2:
            raise ValueError("audio must have shape [B, T]")
        x = self.conv(audio.unsqueeze(1)).transpose(1, 2)
        feat = self.proj(x)
        uncertainty = torch.sigmoid(self.uncertainty_head(feat)).squeeze(-1)
        energy = torch.clamp(audio.abs().mean(dim=1, keepdim=True), min=1e-4)
        return {
            "feat": feat,
            "uncertainty": uncertainty,
            "quality": {"contact": (1 / energy).squeeze(-1)},
        }

    def domain_alignment_loss(self, feat: Tensor, domain_labels: Tensor) -> Tensor:
        if not self.adaptation:
            return torch.zeros((), device=feat.device)
        logits = self.domain_classifier(feat.mean(dim=1))
        return nn.functional.cross_entropy(logits, domain_labels)


__all__ = ["BoneCondEncoder"]
