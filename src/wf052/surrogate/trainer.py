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
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from wf052.surrogate.dataset import (
    CompactSurrogateDataset,
    FIRE_SPLITS,
    SCHEMA_VERSION,
    SurrogateDataset,
    SurrogateSample,
    collate_surrogate_batch,
)
from wf052.surrogate.standin_generator import (
    EXPECTED_CANONICAL_SIM_HASH,
    EXPECTED_TEACHER_CONFIG_HASH,
    validate_shard_metadata,
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
        train_samples: Sequence[SurrogateSample] | torch.utils.data.Dataset,
        val_samples: Sequence[SurrogateSample] | torch.utils.data.Dataset,
        config: TrainConfig | None = None,
        device: torch.device | None = None,
        allow_same_fire_leak: bool = False,
    ) -> None:
        self.config = config or TrainConfig()
        set_reproducible_seed(self.config.seed)
        self.allow_same_fire_leak = allow_same_fire_leak

        # Data leakage prevention: refuse any split placing the same fire in more than one split
        def _extract_fires(data: Sequence[SurrogateSample] | torch.utils.data.Dataset) -> set[str]:
            if isinstance(data, CompactSurrogateDataset):
                return set(data.fire_ids)
            if hasattr(data, "samples") and isinstance(getattr(data, "samples"), Sequence):
                return {getattr(s, "fire_name", getattr(s, "fire_id", "")) for s in data.samples if getattr(s, "fire_name", getattr(s, "fire_id", ""))}
            if isinstance(data, Sequence):
                return {getattr(s, "fire_name", getattr(s, "fire_id", "")) for s in data if getattr(s, "fire_name", getattr(s, "fire_id", ""))}
            return set()

        train_fires = _extract_fires(train_samples)
        val_fires = _extract_fires(val_samples)
        shared_fires = train_fires.intersection(val_fires)

        if shared_fires:
            leak_msg = (
                f"Data leakage detected! Leaking fire(s) {sorted(list(shared_fires))} placed in more than one split "
                f"(found in both train={sorted(list(train_fires))} and val={sorted(list(val_fires))})."
            )
            if not self.allow_same_fire_leak:
                raise ValueError(
                    f"{leak_msg} The trainer strictly refuses splits that place the same fire in more than one split. "
                    "Set allow_same_fire_leak=True to override with explicit warning."
                )
            else:
                warn_msg = f"WARNING: Data leakage permitted via allow_same_fire_leak=True! {leak_msg}"
                logger.warning(warn_msg)
                print(warn_msg)

        # Strict training data constraint (Phase 2C & Phase 2D):
        # 1. Ellipse samples are flagged baseline_only and rejected by the trainer.
        # 2. Frozen teacher samples must match frozen hashes and teacher label sources.
        if isinstance(train_samples, Sequence):
            rejected = [
                s.sample_id
                for s in train_samples
                if getattr(s, "baseline_only", False)
                or getattr(s, "source", "") == "analytic_wind_ellipse"
            ]
            if rejected:
                raise ValueError(
                    f"SurrogateTrainer rejected {len(rejected)} sample(s) with baseline_only=True or "
                    f"source='analytic_wind_ellipse'. Training strictly requires teacher_v1 or frozen teacher labels only. "
                    f"First rejected sample: '{rejected[0]}'."
                )

            # Enforce frozen teacher hashes and valid label sources
            rejected_frozen = []
            for s in train_samples:
                src = getattr(s, "source", "")
                cfg_h = getattr(s, "teacher_config_hash", None)
                sim_h = getattr(s, "canonical_sim_hash", None)

                # If sample specifies source outside recognized teacher sources
                if src not in ("teacher_v2_ensemble", "teacher_v2_embers", "teacher_v1_cellular"):
                    rejected_frozen.append((s.sample_id, f"source='{src}' is not a recognized teacher source"))
                elif src in ("teacher_v2_ensemble", "teacher_v2_embers"):
                    if cfg_h != EXPECTED_TEACHER_CONFIG_HASH:
                        rejected_frozen.append((s.sample_id, f"teacher_config_hash='{cfg_h}' != '{EXPECTED_TEACHER_CONFIG_HASH}'"))
                    elif sim_h != EXPECTED_CANONICAL_SIM_HASH:
                        rejected_frozen.append((s.sample_id, f"canonical_sim_hash='{sim_h}' != '{EXPECTED_CANONICAL_SIM_HASH}'"))

            if rejected_frozen:
                s_id, reason = rejected_frozen[0]
                raise ValueError(
                    f"SurrogateTrainer rejected {len(rejected_frozen)} sample(s) whose hash differs or whose "
                    f"label source is not the frozen teacher. First rejection: '{s_id}' ({reason})."
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

        if isinstance(train_samples, torch.utils.data.Dataset):
            self.train_dataset = train_samples
        else:
            self.train_dataset = SurrogateDataset(train_samples)
        self.val_samples = val_samples

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_surrogate_batch,
            drop_last=(len(self.train_dataset) >= self.config.batch_size),
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

    @classmethod
    def validate_shard(
        cls,
        shard_metadata: pd.DataFrame | str | Path,
        expected_schema_version: str = SCHEMA_VERSION,
    ) -> None:
        """Validate that a shard's metadata matches the frozen teacher hash, label source, and schema_version.

        Raises:
            AssertionError or ValueError if hash differs, label source is not frozen teacher, or schema_version mismatch.
        """
        validate_shard_metadata(shard_metadata, expected_schema_version=expected_schema_version)

    @classmethod
    def from_shards(
        cls,
        model: WindConditionedUNet,
        train_shards: list[str | Path],
        val_shards: list[str | Path],
        config: TrainConfig | None = None,
        device: torch.device | None = None,
        allow_same_fire_leak: bool = False,
        fires_base_dir: str | Path = "data/tiles/fires",
        expected_schema_version: str = SCHEMA_VERSION,
    ) -> "SurrogateTrainer":
        """Construct SurrogateTrainer directly from compact .npz shard files."""
        for sp in train_shards:
            cls.validate_shard(sp, expected_schema_version=expected_schema_version)
        for sp in val_shards:
            cls.validate_shard(sp, expected_schema_version=expected_schema_version)

        train_ds = CompactSurrogateDataset(
            train_shards,
            fires_base_dir=fires_base_dir,
            expected_schema_version=expected_schema_version,
        )
        val_ds = CompactSurrogateDataset(
            val_shards,
            fires_base_dir=fires_base_dir,
            expected_schema_version=expected_schema_version,
        )
        return cls(
            model=model,
            train_samples=train_ds,
            val_samples=val_ds,
            config=config,
            device=device,
            allow_same_fire_leak=allow_same_fire_leak,
        )

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
