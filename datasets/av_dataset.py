"""Audio-visual dataset implementation."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .base_dataset import BaseMultiModalDataset


class AVDataset(BaseMultiModalDataset):
    """Dataset that loads audio-visual samples.

    Parameters
    ----------
    items:
        Sequence of metadata dictionaries. Each must contain ``utt_id`` and
        references for at least one of ``audio_ac`` or ``video``.
    random_crop:
        Optional callable receiving an audio/video tensor pair and returning
        cropped versions. ``None`` keeps inputs untouched.
    """

    def __init__(self, items: Sequence[Dict[str, Any]], *, random_crop: Optional[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]] = None,
                 **kwargs: Any) -> None:
        self.random_crop = random_crop
        super().__init__(items, **kwargs)

    def _load_audio(self, source: Any, sr: int) -> Optional[Tensor]:
        if source is None:
            return None
        if isinstance(source, Tensor):
            audio = source.float()
        elif isinstance(source, (list, tuple)):
            audio = torch.tensor(source, dtype=torch.float32)
        else:
            raise TypeError("Unsupported audio source type")
        if audio.dim() != 1:
            raise ValueError("Audio must be mono 1-D")
        if audio.abs().max() > 1.5:
            raise ValueError("Audio samples must be in [-1, 1]")
        return audio

    def _load_video(self, source: Any, fps: int, size: Tuple[int, int]) -> Optional[Tensor]:
        if source is None:
            return None
        if isinstance(source, Tensor):
            video = source.float()
        elif isinstance(source, dict) and "frames" in source:
            video = torch.tensor(source["frames"], dtype=torch.float32)
        else:
            raise TypeError("Unsupported video source")
        if video.dim() != 4:
            raise ValueError("Video must have shape [T, C, H, W]")
        if tuple(video.shape[2:]) != size:
            raise ValueError("Video frames must match frame size")
        if not torch.isfinite(video).all():
            raise ValueError("Video tensor contains NaNs or infs")
        return video

    def _load_text(self, source: Any) -> Optional[Tensor]:
        if source is None:
            return None
        if isinstance(source, Tensor):
            return source.long()
        return torch.tensor(source, dtype=torch.long)

    def _load_codec_tokens(self, source: Any) -> Optional[List[Tensor]]:
        if source is None:
            return None
        tokens: List[Tensor] = []
        for codebook in source:
            tensor = torch.tensor(codebook, dtype=torch.long)
            tokens.append(tensor)
        return tokens

    def _prepare_sample(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        sample = super()._prepare_sample(meta)
        audio = sample.get("audio_ac")
        video = sample.get("video")
        if self.random_crop and audio is not None and video is not None:
            cropped_audio, cropped_video = self.random_crop(audio, video)
            if cropped_audio.shape[0] == 0 or cropped_video.shape[0] == 0:
                raise ValueError("Random crop must not produce empty tensors")
            sample["audio_ac"] = cropped_audio
            sample["video"] = cropped_video
        if sample["audio_ac"] is None and sample["video"] is None:
            raise ValueError("AVDataset requires at least audio_ac or video")
        return sample


__all__ = ["AVDataset"]
