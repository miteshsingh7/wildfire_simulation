"""Training pipeline for wind-conditioned U-Net surrogate model.

Features:
- Reproducible random seeds (NumPy, PyTorch, Python).
- Automatic Mixed Precision (AMP) via torch.amp.autocast.
- AdamW optimizer with CosineAnnealingLR schedule.
- Multi-task loss (Pinball loss, non-crossing ordering constraint, burn BCE).
- Checkpointing at every epoch to disk (e.g. /kaggle/working/checkpoints/ or local).
- Leave-fire-out cross-validation split utilities.
"""

import logging
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from wf052.surrogate.dataset import (
    SurrogateDataset,
    SurrogateSample,
    collate_surrogate_batch,
)
from wf052.surrogate.evaluation import (
    evaluate_model_on_samples,
)
from wf052.surrogate.losses import LossBreakdown, SurrogateMultiTaskLoss
from wf052.surrogate.model import WindConditionedUNet

logger = logging.getLogger(__name__)


def set_reproducible_seed(seed: int = 42000) -> None:
    """Set seeds for Python, NumPy, and PyTorch (CPU, CUDA, MPS)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def split_leave_fire_out(
    samples: Sequence[SurrogateSample],
    held_out_fire: str,
) -> tuple[list[SurrogateSample], list[SurrogateSample]]:
    """Split samples into train and test sets by leaving one fire landscape out."""
    train_samples: list[SurrogateSample] = []
    val_samples: list[SurrogateSample] = []

    for s in samples:
        if s.fire_name == held_out_fire:
            val_samples.append(s)
        else:
            train_samples.append(s)

    return train_samples, val_samples


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameter configuration."""

    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    lambda_order: float = 1.0
    lambda_bce: float = 0.5
    seed: int = 42000
    checkpoint_dir: str = "checkpoints"


class SurrogateTrainer:
    """Trainer for WindConditionedUNet with AMP, multi-task losses, and checkpointing."""

    def __init__(
        self,
        model: WindConditionedUNet,
        train_samples: list[SurrogateSample],
        val_samples: list[SurrogateSample],
        config: TrainConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or TrainConfig()
        set_reproducible_seed(self.config.seed)

        # Strict training data constraint (Phase 2C):
        # Training strictly uses teacher_v1 labels only.
        # Ellipse samples are flagged baseline_only and rejected by the trainer.
        rejected = [
            s.sample_id
            for s in train_samples
            if getattr(s, "baseline_only", False)
            or getattr(s, "source", "") == "analytic_wind_ellipse"
        ]
        if rejected:
            raise ValueError(
                f"SurrogateTrainer rejected {len(rejected)} sample(s) with baseline_only=True or "
                f"source='analytic_wind_ellipse'. Training strictly requires teacher_v1 labels only. "
                f"First rejected sample: '{rejected[0]}'."
            )

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = device

        self.model = model.to(self.device)
        self.loss_fn = SurrogateMultiTaskLoss(
            lambda_order=self.config.lambda_order,
            lambda_bce=self.config.lambda_bce,
        )

        self.train_dataset = SurrogateDataset(train_samples)
        self.val_samples = val_samples

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_surrogate_batch,
            drop_last=(len(train_samples) >= self.config.batch_size),
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.config.epochs),
            eta_min=1e-6,
        )

        # GradScaler for AMP (CUDA only)
        self.use_amp = True
        self.device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one single epoch."""
        self.model.train()
        total_loss = 0.0
        pinball_loss = 0.0
        ordering_loss = 0.0
        bce_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device_type, enabled=self.use_amp):
                out = self.model(batch.spatial_features, batch.wind_series)
                losses: LossBreakdown = self.loss_fn(
                    pred_quantiles=out.quantiles,
                    pred_burn_logits=out.burn_logits,
                    target_arrival=batch.target_arrival,
                    censor_mask=batch.censor_mask,
                )

            if self.device.type == "cuda":
                self.scaler.scale(losses.total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses.total_loss.backward()
                self.optimizer.step()

            total_loss += losses.total_loss.item()
            pinball_loss += losses.pinball_loss.item()
            ordering_loss += losses.ordering_loss.item()
            bce_loss += losses.bce_loss.item()
            num_batches += 1

        self.scheduler.step()

        denom = max(1, num_batches)
        metrics = {
            "epoch": epoch,
            "train_total_loss": total_loss / denom,
            "train_pinball_loss": pinball_loss / denom,
            "train_ordering_loss": ordering_loss / denom,
            "train_bce_loss": bce_loss / denom,
            "lr": self.scheduler.get_last_lr()[0],
        }
        return metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, Any],
        filename: str | None = None,
    ) -> Path:
        """Save training checkpoint to disk."""
        if filename is None:
            filename = f"surrogate_epoch_{epoch:03d}.pt"
        ckpt_path = self.checkpoint_dir / filename

        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": asdict(self.config),
        }
        torch.save(state, ckpt_path)
        logger.info("Saved checkpoint to %s", ckpt_path)
        return ckpt_path

    def fit(self) -> list[dict[str, Any]]:
        """Run full training routine across epochs with per-epoch checkpointing."""
        history: list[dict[str, Any]] = []

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.perf_counter()
            train_metrics = self.train_epoch(epoch)
            elapsed = time.perf_counter() - t0

            epoch_record = dict(train_metrics)
            epoch_record["elapsed_sec"] = elapsed

            # Evaluate on validation samples if provided
            if len(self.val_samples) > 0:
                val_fidelity = evaluate_model_on_samples(
                    model=self.model,
                    samples=self.val_samples,
                    device=self.device,
                    batch_size=self.config.batch_size,
                )
                epoch_record["val_uncensored_mae"] = val_fidelity.uncensored_mae_minutes
                epoch_record["val_p10_p90_cov"] = val_fidelity.p10_p90_coverage
                epoch_record["val_12h_f1"] = val_fidelity.horizon_metrics["12h"].f1
                epoch_record["val_12h_iou"] = val_fidelity.horizon_metrics["12h"].iou

            # Save checkpoint each epoch
            self.save_checkpoint(epoch=epoch, metrics=epoch_record)
            history.append(epoch_record)

            logger.info(
                "Epoch %d/%d - Loss: %.4f (Pinball: %.4f, Order: %.4f, BCE: %.4f) - %.2fs",
                epoch,
                self.config.epochs,
                train_metrics["train_total_loss"],
                train_metrics["train_pinball_loss"],
                train_metrics["train_ordering_loss"],
                train_metrics["train_bce_loss"],
                elapsed,
            )

        return history
