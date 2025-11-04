"""Batch collation utilities."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence


def _pad_audio(tensors: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    valid = [t for t in tensors if t is not None]
    if not valid:
        return None
    max_len = max(t.shape[0] for t in valid)
    padded = torch.zeros(len(tensors), max_len)
    for idx, tensor in enumerate(tensors):
        if tensor is None:
            continue
        padded[idx, : tensor.shape[0]] = tensor
    return padded


def _pad_video(tensors: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    valid = [t for t in tensors if t is not None]
    if not valid:
        return None
    max_len = max(t.shape[0] for t in valid)
    c, h, w = valid[0].shape[1:]
    batch = torch.zeros(len(tensors), max_len, c, h, w)
    for idx, tensor in enumerate(tensors):
        if tensor is None:
            continue
        length = tensor.shape[0]
        batch[idx, :length] = tensor
    return batch


def _pad_codec(tokens_list: List[Optional[List[torch.Tensor]]]) -> Optional[List[torch.Tensor]]:
    valid = [t for t in tokens_list if t is not None]
    if not valid:
        return None
    num_codebooks = len(valid[0])
    padded: List[torch.Tensor] = []
    for q in range(num_codebooks):
        codebook_tokens = []
        for tokens in tokens_list:
            if tokens is None:
                codebook_tokens.append(torch.zeros(1, dtype=torch.long))
            else:
                codebook_tokens.append(tokens[q])
        max_len = max(t.shape[0] for t in codebook_tokens)
        batch = torch.zeros(len(tokens_list), max_len, dtype=torch.long)
        for idx, tensor in enumerate(codebook_tokens):
            batch[idx, : tensor.shape[0]] = tensor
        padded.append(batch)
    return padded


def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of multimodal samples."""

    batch: Dict[str, Any] = {}
    keys = samples[0].keys()
    for key in keys:
        values = [sample.get(key) for sample in samples]
        if key in {"audio_ac", "audio_bc", "audio_clean"}:
            batch[key] = _pad_audio(values)
        elif key == "video":
            batch[key] = _pad_video(values)
        elif key == "text":
            valid = [v for v in values if v is not None]
            batch[key] = pad_sequence(valid, batch_first=True) if valid else None
        elif key == "codec_tokens":
            batch[key] = _pad_codec(values)
        else:
            batch[key] = values
    return batch


__all__ = ["collate_batch"]
