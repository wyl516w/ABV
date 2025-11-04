"""Visual encoder for 96x96 video frames."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn, Tensor


class VisualEncoder(nn.Module):
    """Encode visual frames into token-aligned features."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(16, hidden_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(self, video: Optional[Tensor]) -> Dict[str, Tensor | None]:
        if video is None:
            return {"feat": None, "uncertainty": None, "quality": None}
        if video.dim() != 5:
            raise ValueError("video must have shape [B, T, C, 96, 96]")
        b, t, c, h, w = video.shape
        x = video.permute(0, 2, 1, 3, 4)
        features = self.conv(x)
        features = features.mean(dim=(-1, -2))
        feat = self.proj(features.permute(0, 2, 1))
        blur = torch.clamp(video.var(dim=(-1, -2)).mean(dim=(2)), min=1e-4)
        uncertainty = torch.sigmoid(self.uncertainty_head(feat)).squeeze(-1)
        return {
            "feat": feat,
            "uncertainty": uncertainty,
            "quality": {"blur": (1 / blur).squeeze(-1)},
        }


__all__ = ["VisualEncoder"]
