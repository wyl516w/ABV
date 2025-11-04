"""Noisy air-conduction encoder."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn, Tensor


class NoisyACEncoder(nn.Module):
    """Encode noisy air-conduction waveforms into token-aligned features."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(self, audio: Optional[Tensor]) -> Dict[str, Tensor | None]:
        if audio is None:
            return {"feat": None, "uncertainty": None, "quality": None}
        if audio.dim() != 2:
            raise ValueError("audio must have shape [B, T]")
        x = audio.unsqueeze(1)
        features = self.frontend(x)
        features = features.transpose(1, 2)
        feat = self.proj(features)
        quality = torch.clamp(audio.std(dim=1, keepdim=True), min=1e-4)
        uncertainty = torch.sigmoid(self.uncertainty_head(feat)).squeeze(-1)
        return {
            "feat": feat,
            "uncertainty": uncertainty,
            "quality": {"snr_proxy": (1 / quality).squeeze(-1)},
        }


__all__ = ["NoisyACEncoder"]
