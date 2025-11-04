"""I/O helpers for dataset implementations."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor


def _ensure_path(source: str | Path) -> Path:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def load_audio(source: str | Path, target_sr: int) -> Tensor:
    """Load audio as a mono float tensor in ``[-1, 1]``."""

    path = _ensure_path(source)
    waveform: Tensor
    sample_rate: int

    try:
        import torchaudio  # type: ignore[import-not-found]

        waveform, sample_rate = torchaudio.load(path)
    except (ImportError, OSError):  # pragma: no cover - optional dependency path
        try:
            import soundfile as sf  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Loading audio from disk requires torchaudio or soundfile"
            ) from exc
        waveform_np, sample_rate = sf.read(path)
        waveform = torch.tensor(waveform_np, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim != 2:
            raise ValueError("Unsupported audio shape returned by soundfile")
        waveform = waveform.transpose(0, 1)

    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    elif waveform.ndim != 1:
        raise ValueError("Audio loader returned tensor with unexpected shape")

    if waveform.abs().max() > 1.5:
        waveform = waveform / waveform.abs().max().clamp(min=1.0)

    if sample_rate != target_sr:
        try:
            import torchaudio.functional as F  # type: ignore[import-not-found]

            waveform = F.resample(waveform.unsqueeze(0), sample_rate, target_sr).squeeze(0)
        except ImportError:  # pragma: no cover - optional dependency path
            warnings.warn(
                "Sample rate mismatch but torchaudio is unavailable; returning original waveform.",
                RuntimeWarning,
            )
    return waveform.float()


def _resize_frames(frames: Tensor, size: Tuple[int, int]) -> Tensor:
    if tuple(frames.shape[-2:]) == size:
        return frames
    frames = frames.unsqueeze(0)
    frames = torch.nn.functional.interpolate(frames, size=size, mode="bilinear", align_corners=False)
    return frames.squeeze(0)


def load_video(source: str | Path, size: Tuple[int, int]) -> Tensor:
    """Load video frames as ``[T, C, H, W]`` float tensor in ``[0, 1]`` range."""

    path = _ensure_path(source)
    frames: Optional[Tensor] = None

    try:
        import decord  # type: ignore[import-not-found]
        import numpy as np

        vr = decord.VideoReader(str(path))
        if len(vr) == 0:
            raise ValueError(f"Video file contains no frames: {path}")
        batch = vr.get_batch(list(range(len(vr)))).asnumpy()
        frames = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
    except ImportError:  # pragma: no cover - optional dependency path
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Loading video requires either decord or opencv-python"
            ) from exc
        capture = cv2.VideoCapture(str(path))
        frame_list: List[Tensor] = []
        success, frame = capture.read()
        while success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            frame_list.append(tensor)
            success, frame = capture.read()
        capture.release()
        if not frame_list:
            raise ValueError(f"Video file contains no frames: {path}")
        frames = torch.stack(frame_list, dim=0)

    if frames is None:
        raise RuntimeError(f"Failed to load video: {path}")

    frames = _resize_frames(frames, size)
    if not torch.isfinite(frames).all():
        raise ValueError("Video frames contain NaN/Inf values")
    return frames


def load_codec_tokens(source: str | Path) -> List[Tensor]:
    """Load codec tokens from JSON files."""

    path = _ensure_path(source)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        values = data.get("codec_tokens")
        if values is None:
            raise ValueError("JSON codec file missing 'codec_tokens' key")
        data = values
    tokens: List[Tensor] = []
    for entry in data:
        tokens.append(torch.tensor(entry, dtype=torch.long))
    return tokens


__all__ = ["load_audio", "load_video", "load_codec_tokens"]
