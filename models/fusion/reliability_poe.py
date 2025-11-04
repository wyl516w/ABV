"""Reliability aware product-of-experts fusion."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn, Tensor


class ReliabilityFusion(nn.Module):
    """Fuse modality features based on uncertainty estimates."""

    def __init__(self, alpha: float = 1.0, use_cross_attention: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.use_cross_attention = use_cross_attention
        self.temperature = 1.0
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=2, batch_first=True) if use_cross_attention else None

    def set_temperature(self, value: float) -> None:
        if value <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = value

    def forward(self, cond_list: List[Tensor], unc_list: List[Tensor]) -> Dict[str, Tensor]:
        if len(cond_list) != len(unc_list):
            raise ValueError("cond_list and unc_list must have same length")
        logits = []
        for unc in unc_list:
            mean_unc = unc.mean(dim=1)
            logits.append((-self.alpha * mean_unc).unsqueeze(-1))
        stacked = torch.cat(logits, dim=-1)
        w = torch.softmax(stacked / self.temperature, dim=-1)
        fused = torch.zeros_like(cond_list[0])
        for idx, feat in enumerate(cond_list):
            weight = w[:, idx].unsqueeze(-1).unsqueeze(-1)
            fused = fused + feat * weight
        if self.use_cross_attention and self.attn is not None:
            query = cond_list[0]
            key = torch.cat(cond_list, dim=1)
            value = key
            attn_out, _ = self.attn(query, key, value)
            fused = fused + attn_out
        return {"fused_ctx": fused, "weights": w.mean(dim=1)}


__all__ = ["ReliabilityFusion"]
