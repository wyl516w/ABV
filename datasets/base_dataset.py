"""Base multimodal dataset abstractions for unified auditory token space project."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


@dataclass
class BaseDatasetConfig:
    """Configuration for :class:`BaseMultiModalDataset`.

    Attributes
    ----------
    sample_rate:
        Audio sample rate expected by datasets. Defaults to 16 kHz.
    video_fps:
        Frame rate for video streams. Defaults to 30 FPS.
    frame_size:
        Height and width for each frame. Defaults to 96 x 96 pixels.
    codec_hop_ms:
        Temporal hop size for codec tokens in milliseconds. Defaults to 20 ms.
    """

    sample_rate: int = 16_000
    video_fps: int = 30
    frame_size: Tuple[int, int] = (96, 96)
    codec_hop_ms: float = 20.0


class BaseMultiModalDataset(torch.utils.data.Dataset):
    """Abstract dataset with helper utilities for multimodal samples.

    Subclasses are responsible for implementing the ``_load_*`` hooks and
    providing metadata for each index. The dataset enforces a unified return
    dictionary where any missing field must be :data:`None`.
    """

    DEFAULT_CONFIG = BaseDatasetConfig()

    def __init__(self, items: Sequence[Dict[str, Any]], *, transforms: Optional[Dict[str, Callable[[Any], Any]]] = None,
                 config: Optional[BaseDatasetConfig] = None) -> None:
        if not isinstance(items, Sequence):
            raise TypeError("items must be a sequence of sample metadata")
        self._items = list(items)
        if not self._items:
            raise ValueError("items must contain at least one element")
        self.config = config or self.DEFAULT_CONFIG
        if self.config.frame_size != (96, 96):
            raise ValueError("frame_size must be (96, 96)")
        self.transforms = transforms or {}

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        meta = self._items[index]
        if "utt_id" not in meta:
            raise KeyError("Each item must provide an 'utt_id'.")
        sample = self._prepare_sample(meta)
        self._sanity_check(sample)
        return sample

    def _prepare_sample(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Load a sample from metadata and apply optional transforms."""

        sr = meta.get("audio_sr", self.config.sample_rate)
        fps = meta.get("video_fps", self.config.video_fps)
        frame_size = self.config.frame_size

        def maybe_transform(key: str, value: Any) -> Any:
            transform = self.transforms.get(key)
            return transform(value) if transform else value

        sample = {
            "utt_id": meta["utt_id"],
            "audio_sr": sr,
            "video_fps": fps,
            "audio_clean": self._load_audio(meta.get("audio_clean"), sr) if meta.get("audio_clean") is not None else None,
            "audio_ac": self._load_audio(meta.get("audio_ac"), sr) if meta.get("audio_ac") is not None else None,
            "audio_bc": self._load_audio(meta.get("audio_bc"), sr) if meta.get("audio_bc") is not None else None,
            "video": self._load_video(meta.get("video"), fps, frame_size) if meta.get("video") is not None else None,
            "text": self._load_text(meta.get("text")) if meta.get("text") is not None else None,
            "codec_tokens": self._load_codec_tokens(meta.get("codec_tokens")) if meta.get("codec_tokens") is not None else None,
            "spk_id": meta.get("spk_id"),
            "spk_emb": self._maybe_tensor(meta.get("spk_emb")),
            "vad_mask": self._maybe_tensor(meta.get("vad_mask"), dtype=torch.bool),
            "align": meta.get("align"),
            "meta": meta.get("meta"),
        }

        for key, value in sample.items():
            sample[key] = maybe_transform(key, value)
        return sample

    def empty_like(self) -> Dict[str, Any]:
        """Return an empty sample containing ``None`` for optional fields."""

        return {
            "utt_id": "",
            "audio_sr": self.config.sample_rate,
            "video_fps": self.config.video_fps,
            "audio_clean": None,
            "audio_ac": None,
            "audio_bc": None,
            "video": None,
            "text": None,
            "codec_tokens": None,
            "spk_id": None,
            "spk_emb": None,
            "vad_mask": None,
            "align": None,
            "meta": None,
        }

    # abstract hooks -----------------------------------------------------

    @abc.abstractmethod
    def _load_audio(self, source: Any, sr: int) -> Optional[Tensor]:
        """Load audio from ``source`` and return a tensor shaped ``[T]``."""

    @abc.abstractmethod
    def _load_video(self, source: Any, fps: int, size: Tuple[int, int]) -> Optional[Tensor]:
        """Load video frames and return a tensor shaped ``[T, C, 96, 96]``."""

    @abc.abstractmethod
    def _load_text(self, source: Any) -> Optional[Tensor]:
        """Load tokenised text and return a 1-D long tensor."""

    @abc.abstractmethod
    def _load_codec_tokens(self, source: Any) -> Optional[List[Tensor]]:
        """Load codec tokens as a list of ``LongTensor`` items."""

    # utilities ----------------------------------------------------------

    def audio_samples_to_tokens(self, num_samples: int) -> int:
        hop = int(self.config.sample_rate * (self.config.codec_hop_ms / 1000.0))
        return (num_samples + hop - 1) // hop

    def video_frames_to_tokens(self, num_frames: int) -> int:
        ratio = self.config.video_fps / (1000.0 / self.config.codec_hop_ms)
        return int(round(num_frames * ratio))

    def tokens_to_audio_samples(self, num_tokens: int) -> int:
        hop = int(self.config.sample_rate * (self.config.codec_hop_ms / 1000.0))
        return num_tokens * hop

    def _maybe_tensor(self, value: Any, *, dtype: Optional[torch.dtype] = None) -> Optional[Tensor]:
        if value is None:
            return None
        if not isinstance(value, Tensor):
            raise TypeError("Expected tensor for provided metadata field")
        return value.to(dtype=dtype) if dtype is not None else value

    def _sanity_check(self, sample: Dict[str, Any]) -> None:
        audio_fields = ["audio_clean", "audio_ac", "audio_bc"]
        if all(sample[field] is None for field in audio_fields) and sample.get("codec_tokens") is None:
            raise ValueError("At least one audio source or codec_tokens must be provided")
        video = sample.get("video")
        if video is not None:
            if video.dim() != 4 or video.shape[2:] != (96, 96):
                raise ValueError("video must have shape [T, C, 96, 96]")
        for key in ("audio_clean", "audio_ac", "audio_bc"):
            tensor = sample.get(key)
            if tensor is not None and tensor.dim() != 1:
                raise ValueError(f"{key} must be 1-D audio tensor")


__all__ = ["BaseMultiModalDataset", "BaseDatasetConfig"]
