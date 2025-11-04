"""Streaming inference utilities."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor

from codec.codec_adapter import AudioCodecAdapter
from models.lm.codec_ar_transformer import CodecARTransformer


@dataclass
class StreamingConfig:
    window_ms: int = 40
    hop_ms: int = 20
    ema: float = 0.9


class StreamingState:
    def __init__(self, model: CodecARTransformer) -> None:
        self.model = model
        self.prefix = [torch.zeros(1, 1, dtype=torch.long) for _ in model.vocab_sizes]
        self.cache: Optional[List[Tensor]] = None
        self.weights = torch.ones(len(model.vocab_sizes)) / len(model.vocab_sizes)

    def step(self, fused_ctx: Tensor) -> List[Tensor]:
        output = self.model(self.prefix, fused_ctx, self.cache)
        self.cache = output["cache"]
        logits = output["logits"]
        tokens = []
        for idx, logit in enumerate(logits):
            token = torch.argmax(torch.softmax(logit[:, -1], dim=-1), dim=-1)
            self.prefix[idx] = torch.cat([self.prefix[idx], token.unsqueeze(0)], dim=1)
            tokens.append(token)
        return tokens


def stream_demo(iterations: int = 2) -> None:
    codec = AudioCodecAdapter(has_weights=True)
    model = CodecARTransformer([8, 8])
    state = StreamingState(model)
    fused_ctx = torch.zeros(1, 1, 64)
    for _ in range(iterations):
        tokens = state.step(fused_ctx)
        codec.decode([prefix.squeeze(0) for prefix in state.prefix])
        print("Generated", [t.item() for t in tokens])


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming inference demo")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    stream_demo(args.steps)


if __name__ == "__main__":  # pragma: no cover
    main()
