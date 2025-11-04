"""Offline synthesis helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor

from codec.codec_adapter import AudioCodecAdapter
from models.lm.codec_ar_transformer import CodecARTransformer


@dataclass
class SynthesisConfig:
    max_tokens: int = 4
    temperature: float = 1.0
    top_p: float = 0.9
    repetition_penalty: float = 1.1


def synthesize(
    conditions: Dict[str, Optional[Tensor]],
    model: CodecARTransformer,
    codec: AudioCodecAdapter,
    config: SynthesisConfig,
    seed: int = 0,
) -> Dict[str, Tensor]:
    torch.manual_seed(seed)
    prefix = [torch.zeros(1, 1, dtype=torch.long) for _ in model.vocab_sizes]
    fused_ctx = torch.zeros(1, 1, model.heads[0].in_features)
    generated: List[Tensor] = []
    for step in range(config.max_tokens):
        output = model(prefix, fused_ctx)
        logits = output["logits"]
        step_tokens = []
        for head_logits in logits:
            probs = torch.softmax(head_logits[:, -1] / config.temperature, dim=-1)
            token = torch.argmax(probs, dim=-1)
            step_tokens.append(token)
        for idx, token in enumerate(step_tokens):
            prefix[idx] = torch.cat([prefix[idx], token.unsqueeze(0)], dim=1)
        generated = [torch.cat([g, token.unsqueeze(0)], dim=1) if len(g.shape) else token.unsqueeze(0) for g, token in zip(prefix, step_tokens)]
    waveform = codec.decode([tokens.squeeze(0) for tokens in prefix])
    return {"tokens": [tokens.squeeze(0) for tokens in prefix], "waveform": waveform}


__all__ = ["synthesize", "SynthesisConfig"]
