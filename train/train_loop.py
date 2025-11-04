"""Training loop for the unified auditory token space model."""
# If ``audio_clean`` is largely missing, prefer ``codec_tokens``; otherwise use ``audio_ac`` as last-resort targets.
# With unpaired AV and BC, train mixed batches; use adversarial or distance-based alignment so BC features match AC.
# Uncertainty heads: weak supervision—AC via SNR/spectral entropy, V via mouth quality, BC via energy stability/"contact" score.
# Fusion: under noise/occlusion, weights should shift to BC/V; in clean settings, shift back to AC.
# Never feed ``audio_clean`` as a conditioning input.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:  # pragma: no cover - optional dependency guard
    import lightning as L
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    L = None

from datasets.collate_fn import collate_batch
from models.encoders.noisy_ac import NoisyACEncoder
from models.encoders.bc import BoneCondEncoder
from models.encoders.vis import VisualEncoder
from models.fusion.reliability_poe import ReliabilityFusion
from models.lm.codec_ar_transformer import CodecARTransformer
from train.losses import codec_ce_loss


@dataclass
class TrainerConfig:
    """Configuration for the lightweight Lightning training loop."""

    batch_size: int = 2
    steps: int = 2
    learning_rate: float = 1e-3
    seed: int = 0
    accelerator: str = "cpu"
    devices: int = 1
    enable_progress_bar: bool = False
    deterministic: bool = True


class DummyDataset(Dataset):
    """Minimal dataset producing synthetic multimodal batches for tests."""

    def __len__(self) -> int:  # pragma: no cover - trivial
        return 4

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        audio = torch.randn(160)
        video = torch.randn(4, 3, 96, 96)
        tokens = [torch.randint(0, 8, (8,), dtype=torch.long), torch.randint(0, 8, (8,), dtype=torch.long)]
        return {
            "utt_id": f"utt-{idx}",
            "audio_sr": 16000,
            "video_fps": 30,
            "audio_ac": audio,
            "audio_bc": audio * 0.5,
            "audio_clean": None,
            "video": video,
            "text": None,
            "codec_tokens": tokens,
            "spk_id": None,
            "spk_emb": None,
            "vad_mask": None,
            "align": None,
            "meta": None,
        }


def build_model() -> Dict[str, nn.Module]:
    """Construct the component modules used by the Lightning trainer."""

    ac = NoisyACEncoder()
    bc = BoneCondEncoder()
    vis = VisualEncoder()
    fusion = ReliabilityFusion()
    lm = CodecARTransformer([8, 8])
    return {"ac": ac, "bc": bc, "vis": vis, "fusion": fusion, "lm": lm}


if L is not None:

    class UnifiedAuditoryModule(L.LightningModule):
        """LightningModule wiring the encoders, fusion block, and AR codec LM."""

        def __init__(self, learning_rate: float = 1e-3) -> None:
            super().__init__()
            modules = build_model()
            self.ac_encoder: NoisyACEncoder = modules["ac"]
            self.bc_encoder: BoneCondEncoder = modules["bc"]
            self.vis_encoder: VisualEncoder = modules["vis"]
            self.fusion: ReliabilityFusion = modules["fusion"]
            self.lm: CodecARTransformer = modules["lm"]
            self.learning_rate = learning_rate
            self.last_loss: torch.Tensor = torch.tensor(float("nan"))

        def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
            return self._compute_loss(batch)

        def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:  # noqa: D401 - Lightning API
            processed = self._move_batch_to_device(batch)
            loss = self._compute_loss(processed)
            self.last_loss = loss.detach()
            self.log("train/loss", loss, on_step=True, prog_bar=False)
            return loss

        def configure_optimizers(self) -> torch.optim.Optimizer:  # noqa: D401 - Lightning API
            params = [
                *self.ac_encoder.parameters(),
                *self.bc_encoder.parameters(),
                *self.vis_encoder.parameters(),
                *self.fusion.parameters(),
                *self.lm.parameters(),
            ]
            return torch.optim.Adam(params, lr=self.learning_rate)

        def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
            device = self.device
            processed: Dict[str, Any] = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    processed[key] = value.to(device)
                elif isinstance(value, list):
                    processed[key] = [
                        item.to(device) if isinstance(item, torch.Tensor) else item for item in value
                    ]
                else:
                    processed[key] = value
            return processed

        def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
            features: List[torch.Tensor] = []
            uncertainties: List[torch.Tensor] = []

            audio_ac = batch.get("audio_ac")
            if isinstance(audio_ac, torch.Tensor):
                out = self.ac_encoder(audio_ac)
                if out["feat"] is not None:
                    features.append(out["feat"])
                    uncertainties.append(out["uncertainty"])

            audio_bc = batch.get("audio_bc")
            if isinstance(audio_bc, torch.Tensor):
                out = self.bc_encoder(audio_bc)
                if out["feat"] is not None:
                    features.append(out["feat"])
                    uncertainties.append(out["uncertainty"])

            video = batch.get("video")
            if isinstance(video, torch.Tensor):
                out = self.vis_encoder(video)
                if out["feat"] is not None:
                    features.append(out["feat"])
                    uncertainties.append(out["uncertainty"])

            if not features:
                raise RuntimeError("At least one conditioning modality must be available for fusion")

            fused_ctx = self.fusion(features, uncertainties)["fused_ctx"]

            codec_tokens = batch.get("codec_tokens")
            if codec_tokens is None:
                raise RuntimeError("codec_tokens are required as training targets in this scaffold")
            targets = [tokens.long() for tokens in codec_tokens]

            logits = self.lm(targets, fused_ctx)["logits"]
            loss = codec_ce_loss(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss became non-finite during training")
            return loss


    class DummyDataModule(L.LightningDataModule):
        """LightningDataModule wrapping the :class:`DummyDataset`."""

        def __init__(self, batch_size: int) -> None:
            super().__init__()
            self.batch_size = batch_size
            self._dataset: Optional[Dataset] = None

        def setup(self, stage: Optional[str] = None) -> None:  # noqa: D401 - Lightning API
            self._dataset = DummyDataset()

        def train_dataloader(self) -> DataLoader:  # noqa: D401 - Lightning API
            if self._dataset is None:
                self.setup()
            assert self._dataset is not None
            return DataLoader(self._dataset, batch_size=self.batch_size, collate_fn=collate_batch)

else:  # pragma: no cover - makes module importable without Lightning

    UnifiedAuditoryModule = None  # type: ignore[assignment]
    DummyDataModule = None  # type: ignore[assignment]


def run_training(config: TrainerConfig) -> float:
    """Execute a tiny Lightning training loop and return the final loss."""

    if L is None or UnifiedAuditoryModule is None or DummyDataModule is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "The `lightning` package (>=2.5) is required to run the training loop."
        )

    L.seed_everything(config.seed)
    model = UnifiedAuditoryModule(learning_rate=config.learning_rate)
    data_module = DummyDataModule(batch_size=config.batch_size)
    trainer = L.Trainer(
        max_steps=config.steps,
        max_epochs=1,
        accelerator=config.accelerator,
        devices=config.devices,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=config.enable_progress_bar,
        deterministic=config.deterministic,
        limit_train_batches=config.steps,
    )
    trainer.fit(model, datamodule=data_module)
    return float(model.last_loss.detach().cpu())


__all__ = ["run_training", "TrainerConfig", "UnifiedAuditoryModule", "DummyDataModule"]
