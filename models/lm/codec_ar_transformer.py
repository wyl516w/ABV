"""Autoregressive transformer for codec tokens."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn, Tensor


class CodecARTransformer(nn.Module):
    """Minimal transformer predicting codec tokens."""

    def __init__(self, vocab_sizes: List[int], hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embeddings = nn.ModuleList([nn.Embedding(v, hidden_dim) for v in vocab_sizes])
        encoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerDecoder(encoder_layer, num_layers=num_layers)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, v) for v in vocab_sizes])

    def forward(
        self,
        prefix_tokens: List[Tensor],
        fused_ctx: Tensor,
        kv_cache: Optional[List[Tensor]] = None,
    ) -> Dict[str, List[Tensor] | List[Tensor]]:
        embeds = []
        for idx, tokens in enumerate(prefix_tokens):
            embed = self.embeddings[idx](tokens)
            embeds.append(embed)
        combined = torch.stack(embeds, dim=0).sum(dim=0)
        memory = fused_ctx
        tgt = combined
        decoded = self.transformer(tgt, memory)
        logits = [head(decoded) for head in self.heads]
        cache = [decoded.detach()] * len(self.vocab_sizes)
        return {"logits": logits, "cache": cache}


__all__ = ["CodecARTransformer"]
