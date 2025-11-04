"""Bone-conduction dataset implementation."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .base_dataset import BaseMultiModalDataset


class BCDataset(BaseMultiModalDataset):
    """Dataset for samples prioritising bone-conduction audio."""

    def __init__(self, items: Sequence[Dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(items, **kwargs)

    def _load_audio(self, source: Any, sr: int) -> Optional[Tensor]:
        if source is None:
            return None
        if isinstance(source, Tensor):
            tensor = source.float()
        elif isinstance(source, (list, tuple)):
            tensor = torch.tensor(source, dtype=torch.float32)
        else:
            raise TypeError("Unsupported audio source type")
        if tensor.dim() != 1:
            raise ValueError("Audio must be 1-D")
        return tensor

    def _load_video(self, source: Any, fps: int, size: Tuple[int, int]) -> Optional[Tensor]:
        if source is None:
            return None
        if isinstance(source, Tensor):
            video = source.float()
        elif isinstance(source, dict) and "frames" in source:
            video = torch.tensor(source["frames"], dtype=torch.float32)
        else:
            raise TypeError("Unsupported video source")
        if video.shape[2:] != size:
            raise ValueError("Video frame size mismatch")
        return video

    def _load_text(self, source: Any) -> Optional[Tensor]:
        if source is None:
            return None
        return torch.tensor(source, dtype=torch.long)

    def _load_codec_tokens(self, source: Any) -> Optional[List[Tensor]]:
        if source is None:
            return None
        return [torch.tensor(tokens, dtype=torch.long) for tokens in source]

    def _prepare_sample(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        sample = super()._prepare_sample(meta)
        if sample.get("audio_bc") is None and sample.get("audio_ac") is None and sample.get("codec_tokens") is None:
            raise ValueError("BCDataset requires audio_bc/audio_ac or codec tokens")
        return sample


__all__ = ["BCDataset"]
