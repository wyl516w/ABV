"""Codec adapter utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor


@dataclass
class CodecConfig:
    """Configuration for the lightweight codec adapter."""

    sample_rate: int = 16_000
    hop_ms: float = 20.0


class MissingWeightsError(RuntimeError):
    """Raised when codec weights are not available."""


class AudioCodecAdapter:
    """Simple codec adapter placeholder requiring pretrained weights."""

    def __init__(
        self,
        *,
        device: torch.device | None = None,
        has_weights: bool = False,
        config: CodecConfig | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.config = config or CodecConfig()
        self.has_weights = has_weights
        torch.manual_seed(0)

    def _ensure_weights(self) -> None:
        if not self.has_weights:
            raise MissingWeightsError(
                "Codec weights required. Please download pretrained weights and set 'has_weights=True' when instantiating."
            )

    def encode(self, wav_16k: Tensor) -> List[Tensor]:
        """Encode audio into a list of token sequences."""

        self._ensure_weights()
        if wav_16k.dim() != 1:
            raise ValueError("wav_16k must be a 1-D tensor")
        hop = int(self.config.sample_rate * (self.config.hop_ms / 1000.0))
        num_tokens = (wav_16k.shape[0] + hop - 1) // hop
        tokens = torch.arange(num_tokens, device=self.device, dtype=torch.long) % 16
        return [tokens]

    def decode(self, tokens: List[Tensor]) -> Tensor:
        """Decode codec tokens into a waveform tensor."""

        self._ensure_weights()
        if not tokens:
            raise ValueError("tokens must contain at least one codebook")
        hop = int(self.config.sample_rate * (self.config.hop_ms / 1000.0))
        length = tokens[0].shape[0] * hop
        noise = torch.zeros(length, device=self.device)
        return noise


__all__ = ["AudioCodecAdapter", "CodecConfig", "MissingWeightsError"]
