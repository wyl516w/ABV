"""Utility functions for evaluation metrics.

The real project is expected to plug in dataset or model specific evaluation
hooks (e.g. an ASR model for WER, STOI/ViSQOL packages, ECAPA/x-vector
embeddings, etc.).  The helpers below provide a lightweight yet well documented
stand-in that is safe to run inside unit tests.  Each helper validates its
inputs and documents the tensor shapes that are expected so that downstream
callers receive informative errors when a modality is missing.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Dict, Iterable, List, Sequence

import torch
from torch import Tensor


def _normalise_transcript(transcript: Iterable[str] | str) -> List[str]:
    """Normalise the transcript input to a list of tokens.

    Parameters
    ----------
    transcript:
        Either an iterable of string tokens or a whitespace separated string.

    Returns
    -------
    list of str
        The transcript expressed as a list of tokens.
    """

    if isinstance(transcript, str):
        transcript = transcript.strip().split()
    else:
        transcript = list(transcript)
    return transcript


def _normalise_characters(transcript: Iterable[str] | str) -> List[str]:
    """Normalise the transcript to a list of characters without whitespace."""

    if isinstance(transcript, str):
        tokens = transcript
    else:
        tokens = "".join(str(token) for token in transcript)
    return [ch for ch in tokens if not ch.isspace()]


def wer(hypothesis: Iterable[str] | str, reference: Iterable[str] | str) -> float:
    """Compute the word-error-rate (WER) between two transcripts.

    The function performs a Levenshtein edit-distance computation and normalises
    the number of edits by the reference length.  Empty references return ``0``
    by convention.

    Parameters
    ----------
    hypothesis, reference:
        The hypothesis and reference token sequences.  Either a pre-tokenised
        iterable of strings or a whitespace delimited string.

    Returns
    -------
    float
        The WER in the range ``[0, 1]``.
    """

    hyp_tokens = _normalise_transcript(hypothesis)
    ref_tokens = _normalise_transcript(reference)
    if not ref_tokens:
        return 0.0

    # Classic Levenshtein DP.
    dp = torch.zeros((len(ref_tokens) + 1, len(hyp_tokens) + 1), dtype=torch.int32)
    dp[0] = torch.arange(len(hyp_tokens) + 1)
    dp[:, 0] = torch.arange(len(ref_tokens) + 1)
    for i, ref_token in enumerate(ref_tokens, start=1):
        for j, hyp_token in enumerate(hyp_tokens, start=1):
            cost = 0 if ref_token == hyp_token else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,  # deletion
                dp[i, j - 1] + 1,  # insertion
                dp[i - 1, j - 1] + cost,  # substitution
            )

    return float(dp[-1, -1].item() / len(ref_tokens))


def cer(hypothesis: Iterable[str] | str, reference: Iterable[str] | str) -> float:
    """Compute the character-error-rate (CER) between two transcripts."""

    hyp_chars = _normalise_characters(hypothesis)
    ref_chars = _normalise_characters(reference)
    if not ref_chars:
        return 0.0

    dp = torch.zeros((len(ref_chars) + 1, len(hyp_chars) + 1), dtype=torch.int32)
    dp[0] = torch.arange(len(hyp_chars) + 1)
    dp[:, 0] = torch.arange(len(ref_chars) + 1)
    for i, ref_char in enumerate(ref_chars, start=1):
        for j, hyp_char in enumerate(hyp_chars, start=1):
            cost = 0 if ref_char == hyp_char else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )

    return float(dp[-1, -1].item() / len(ref_chars))


def stoi_placeholder(prediction: Tensor, target: Tensor) -> float:
    """Placeholder short-time objective intelligibility score.

    A real implementation would dispatch to the `pystoi` package.  For unit
    testing we compute the mean cosine similarity between flattened audio
    frames.  Inputs are validated to ensure that callers provide non-empty,
    matching tensors of shape ``[..., T]``.
    """

    if prediction.ndim == 0 or target.ndim == 0:
        raise ValueError("prediction and target must be at least 1-D tensors")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share the same shape")

    pred_flat = prediction.reshape(prediction.shape[0], -1)
    tgt_flat = target.reshape(target.shape[0], -1)
    score = torch.cosine_similarity(pred_flat, tgt_flat, dim=-1).mean()
    return float(score.clamp(min=-1.0, max=1.0).item())


def stoi(prediction: Tensor, target: Tensor, sample_rate: int, extended: bool = False) -> float:
    """Compute STOI/ESTOI via :mod:`pystoi` with lazy importing."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    try:
        from pystoi import stoi as stoi_fn  # type: ignore import
    except ImportError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "pystoi is required for STOI/ESTOI computation. Install with `pip install pystoi`."
        ) from exc

    prediction_np = prediction.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    if prediction_np.shape != target_np.shape:
        raise ValueError("prediction and target must share the same shape")

    return float(stoi_fn(target_np, prediction_np, sample_rate, extended=extended))


def estoi(prediction: Tensor, target: Tensor, sample_rate: int) -> float:
    """Extended short-time objective intelligibility using :func:`stoi`."""

    return stoi(prediction, target, sample_rate, extended=True)


def pesq_score(
    prediction: Tensor,
    target: Tensor,
    sample_rate: int,
    mode: str = "wb",
) -> float:
    """Compute PESQ using the optional :mod:`pesq` package."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    try:
        from pesq import pesq as pesq_fn  # type: ignore import
    except ImportError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "pesq is required for PESQ computation. Install with `pip install pesq`."
        ) from exc

    prediction_np = prediction.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    if prediction_np.shape != target_np.shape:
        raise ValueError("prediction and target must share the same shape")

    if mode not in {"wb", "nb"}:
        raise ValueError("mode must be either 'wb' (wide-band) or 'nb' (narrow-band)")

    return float(pesq_fn(sample_rate, target_np, prediction_np, mode))


def reliability_curve(errors: Tensor, uncertainties: Tensor) -> Dict[str, Tensor]:
    """Generate a reliability curve from error/uncertainty pairs.

    Parameters
    ----------
    errors:
        1-D tensor containing absolute or squared errors per sample.
    uncertainties:
        1-D tensor containing predictive uncertainties for each sample.

    Returns
    -------
    dict
        ``{"uncertainty": Tensor, "error": Tensor}`` where the error tensor
        holds the cumulative mean error sorted by ascending uncertainty.
    """

    if errors.ndim != 1 or uncertainties.ndim != 1:
        raise ValueError("errors and uncertainties must be 1-D tensors")
    if errors.shape != uncertainties.shape:
        raise ValueError("errors and uncertainties must match in length")
    if errors.numel() == 0:
        raise ValueError("errors tensor must not be empty")

    sorted_uncertainty, order = torch.sort(uncertainties)
    sorted_error = errors[order]
    cumulative_error = torch.cumsum(sorted_error, dim=0)
    cumulative_error = cumulative_error / torch.arange(1, len(sorted_error) + 1, dtype=sorted_error.dtype)
    return {"uncertainty": sorted_uncertainty, "error": cumulative_error}


def calibration_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    num_bins: int = 10,
) -> Dict[str, float]:
    """Compute simple ECE/ACE style calibration metrics.

    Parameters
    ----------
    prediction, target:
        1-D tensors containing probabilities and binary ground truth labels.
    num_bins:
        Number of equally spaced calibration bins in ``[0, 1]``.  The default is
        ``10`` which keeps the calculation stable even on tiny batch sizes.
    """

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share the same shape")
    if prediction.numel() == 0:
        raise ValueError("prediction must not be empty")
    if num_bins <= 0:
        raise ValueError("num_bins must be a positive integer")

    prediction = prediction.detach().clamp(0.0, 1.0)
    target = target.detach().clamp(0.0, 1.0)

    bin_boundaries = torch.linspace(0.0, 1.0, num_bins + 1, device=prediction.device)
    ece = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    ace = torch.zeros_like(ece)

    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (prediction >= lower) & (prediction < upper if upper < 1 else prediction <= upper)
        if not torch.any(in_bin):
            continue
        prob_mean = prediction[in_bin].mean()
        acc_mean = target[in_bin].mean()
        bin_prob = in_bin.float().mean()
        ece += bin_prob * torch.abs(acc_mean - prob_mean)
        ace += bin_prob * (acc_mean - prob_mean)

    return {"ece": float(ece.abs().item()), "ace": float(ace.item())}


def speaker_similarity(embedding_a: Tensor, embedding_b: Tensor) -> float:
    """Compute cosine similarity between two sets of speaker embeddings.

    The function accepts tensors of shape ``[N, D]`` or ``[D]`` and broadcasts
    the smaller tensor when possible.  Unlike the original placeholder it
    validates the shapes and raises an informative :class:`ValueError` when the
    comparison is ill defined.
    """

    if embedding_a.shape != embedding_b.shape:
        if embedding_a.ndim == 2 and embedding_b.ndim == 1 and embedding_a.shape[1] == embedding_b.shape[0]:
            embedding_b = embedding_b.unsqueeze(0).expand_as(embedding_a)
        elif embedding_b.ndim == 2 and embedding_a.ndim == 1 and embedding_b.shape[1] == embedding_a.shape[0]:
            embedding_a = embedding_a.unsqueeze(0).expand_as(embedding_b)
        else:
            raise ValueError("embedding_a and embedding_b must be broadcastable to the same shape")

    similarity = torch.cosine_similarity(embedding_a, embedding_b, dim=-1)
    return float(similarity.mean().clamp(min=-1.0, max=1.0).item())


__all__ = [
    "wer",
    "cer",
    "stoi_placeholder",
    "stoi",
    "estoi",
    "pesq_score",
    "reliability_curve",
    "calibration_metrics",
    "speaker_similarity",
]
