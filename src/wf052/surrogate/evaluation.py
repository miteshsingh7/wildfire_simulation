"""Evaluation module for surrogate fire spread models (Phase 2A).

Metrics and Protocols:
- Leave-fire-out splits across the 7 training fires.
- Metrics:
    - Arrival MAE (minutes) on uncensored cells.
    - IoU and F1 at 1 h (60 min), 3 h (180 min), 6 h (360 min), and 12 h (720 min).
    - P10-P90 coverage fraction (target ~80%) on uncensored cells.
    - CPU and GPU inference latency vs teacher benchmark.
- Terminology rule: All outputs report "fidelity to teacher" (never "accuracy").
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from wf052.surrogate.dataset import (
    SurrogateDataset,
    SurrogateSample,
    collate_surrogate_batch,
)


@dataclass(frozen=True)
class HorizonMetrics:
    """Metrics at a specific simulation horizon (e.g. 1h, 3h, 6h, 12h)."""

    horizon_minutes: float
    iou: float
    f1: float
    tp: int
    fp: int
    fn: int


@dataclass(frozen=True)
class SurrogateFidelityMetrics:
    """Strongly typed container for surrogate fidelity to teacher benchmark.

    NOTE: Results represent fidelity to teacher simulation, never ground-truth accuracy.
    """

    model_name: str
    fire_name: str
    num_samples: int
    uncensored_mae_minutes: float
    p10_p90_coverage: float  # Fraction of uncensored arrivals in [q10, q90]
    horizon_metrics: dict[str, HorizonMetrics]  # "1h", "3h", "6h", "12h"
    inference_latency_ms: float
    cells_per_second: float
    speedup_vs_teacher: float


def compute_binary_metrics_at_horizon(
    pred_arrival: np.ndarray,
    target_arrival: np.ndarray,
    censor_mask: np.ndarray,
    horizon_min: float,
) -> HorizonMetrics:
    """Compute IoU and F1 score for fire footprint at a specified time horizon."""
    # True burned: arrived at or before horizon and not censored
    y_true = (target_arrival <= horizon_min) & (censor_mask == 0.0)
    # Predicted burned: predicted median arrival at or before horizon
    y_pred = pred_arrival <= horizon_min

    tp = int(np.logical_and(y_true, y_pred).sum())
    fp = int(np.logical_and(~y_true, y_pred).sum())
    fn = int(np.logical_and(y_true, ~y_pred).sum())

    denom_iou = tp + fp + fn
    iou = float(tp / denom_iou) if denom_iou > 0 else 1.0

    denom_f1 = 2 * tp + fp + fn
    f1 = float((2 * tp) / denom_f1) if denom_f1 > 0 else 1.0

    return HorizonMetrics(
        horizon_minutes=horizon_min,
        iou=iou,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def evaluate_predictions_fidelity(
    pred_quantiles: np.ndarray,  # (N, 3, H, W)
    target_arrival: np.ndarray,  # (N, H, W)
    censor_mask: np.ndarray,  # (N, H, W)
    model_name: str,
    fire_name: str,
    elapsed_ms: float,
    teacher_elapsed_ms: float = 1.0,
) -> SurrogateFidelityMetrics:
    """Evaluate surrogate predictions against teacher target arrival surface.

    Labels all outputs as 'fidelity to teacher'.
    """
    n, _, h, w = pred_quantiles.shape
    total_cells = n * h * w

    q10 = pred_quantiles[:, 0, :, :]
    q50 = pred_quantiles[:, 1, :, :]
    q90 = pred_quantiles[:, 2, :, :]

    # Uncensored cells: burned within 12h horizon
    uncensored = censor_mask == 0.0
    num_uncensored = int(uncensored.sum())

    if num_uncensored > 0:
        # MAE between median quantile (q50) and teacher arrival time
        abs_diff = np.abs(q50[uncensored] - target_arrival[uncensored])
        mae = float(np.mean(abs_diff))

        # Coverage: true arrival inside [q10, q90]
        in_interval = (target_arrival[uncensored] >= q10[uncensored]) & (
            target_arrival[uncensored] <= q90[uncensored]
        )
        coverage = float(np.mean(in_interval))
    else:
        mae = 0.0
        coverage = 1.0

    # Horizon metrics at 1h (60m), 3h (180m), 6h (360m), 12h (720m)
    horizons = {"1h": 60.0, "3h": 180.0, "6h": 360.0, "12h": 720.0}
    horizon_results: dict[str, HorizonMetrics] = {}
    for h_name, h_min in horizons.items():
        horizon_results[h_name] = compute_binary_metrics_at_horizon(
            pred_arrival=q50,
            target_arrival=target_arrival,
            censor_mask=censor_mask,
            horizon_min=h_min,
        )

    cells_per_sec = (total_cells / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0
    speedup = (teacher_elapsed_ms / elapsed_ms) if elapsed_ms > 0 else 1.0

    return SurrogateFidelityMetrics(
        model_name=model_name,
        fire_name=fire_name,
        num_samples=n,
        uncensored_mae_minutes=mae,
        p10_p90_coverage=coverage,
        horizon_metrics=horizon_results,
        inference_latency_ms=elapsed_ms / max(1, n),
        cells_per_second=cells_per_sec,
        speedup_vs_teacher=speedup,
    )


def evaluate_model_on_samples(
    model: nn.Module,
    samples: Sequence[SurrogateSample] | torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int = 4,
    teacher_ref_ms: float = 35.0,
    model_name: str = "wind_conditioned_unet",
) -> SurrogateFidelityMetrics:
    """Run inference with the surrogate model on samples and compute fidelity metrics."""
    model.eval()
    if isinstance(samples, torch.utils.data.Dataset):
        dataset = samples
        if hasattr(samples, "fire_ids") and len(samples.fire_ids) > 0:
            fire_label = str(samples.fire_ids[0])
        else:
            fire_label = "unknown"
    else:
        dataset = SurrogateDataset(samples)
        fire_label = samples[0].fire_name if len(samples) > 0 else "unknown"

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_surrogate_batch,
    )

    all_preds_q: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_censors: list[np.ndarray] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # Support AMP on device
            device_type = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=device_type):
                out = model(batch.spatial_features, batch.wind_series)

            all_preds_q.append(out.quantiles.detach().float().cpu().numpy())
            all_targets.append(batch.target_arrival.detach().float().cpu().numpy())
            all_censors.append(batch.censor_mask.detach().float().cpu().numpy())

    total_elapsed_ms = (time.perf_counter() - t0) * 1000.0

    pred_q_arr = np.concatenate(all_preds_q, axis=0)
    target_arr = np.concatenate(all_targets, axis=0)
    censor_arr = np.concatenate(all_censors, axis=0)

    fire_label = samples[0].fire_name if len(samples) > 0 else "unknown"

    return evaluate_predictions_fidelity(
        pred_quantiles=pred_q_arr,
        target_arrival=target_arr,
        censor_mask=censor_arr,
        model_name=model_name,
        fire_name=fire_label,
        elapsed_ms=total_elapsed_ms,
        teacher_elapsed_ms=teacher_ref_ms * len(samples),
    )


def generate_fidelity_report_markdown(
    eval_results: Sequence[SurrogateFidelityMetrics],
    baseline_results: Sequence[SurrogateFidelityMetrics] | None = None,
) -> str:
    """Generate markdown report explicitly titled 'Fidelity to Teacher'."""
    lines: list[str] = [
        "# Surrogate Model Evaluation: Fidelity to Teacher Benchmark",
        "",
        "> [!IMPORTANT]",
        "> All metrics reported below measure **fidelity to teacher simulation**, NOT ground truth.",
        "> Stand-in training data was generated via analytic wind-ellipse and teacher_v1 cellular automaton scaffold.",
        "",
        "## Summary Metrics by Fire (Leave-Fire-Out)",
        "",
        "| Fire Landscape | Model | Uncensored MAE (min) | P10-P90 Cov. (tgt ~80%) | 1h F1 | 3h F1 | 6h F1 | 12h F1 | 12h IoU | Latency (ms/tile) | Speedup vs Teacher |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for res in eval_results:
        m1 = res.horizon_metrics.get("1h", HorizonMetrics(60, 0, 0, 0, 0, 0))
        m3 = res.horizon_metrics.get("3h", HorizonMetrics(180, 0, 0, 0, 0, 0))
        m6 = res.horizon_metrics.get("6h", HorizonMetrics(360, 0, 0, 0, 0, 0))
        m12 = res.horizon_metrics.get("12h", HorizonMetrics(720, 0, 0, 0, 0, 0))

        lines.append(
            f"| {res.fire_name} | {res.model_name} | {res.uncensored_mae_minutes:.2f} | "
            f"{res.p10_p90_coverage * 100.0:.1f}% | {m1.f1:.3f} | {m3.f1:.3f} | {m6.f1:.3f} | "
            f"{m12.f1:.3f} | {m12.iou:.3f} | {res.inference_latency_ms:.2f} ms | {res.speedup_vs_teacher:.1f}x |"
        )

    if baseline_results:
        lines.append("")
        lines.append("## Baselines Comparison (Speed and Fidelity Reference)")
        lines.append("")
        lines.append(
            "| Baseline | Fire | Uncensored MAE (min) | P10-P90 Cov. | 12h F1 | 12h IoU | Latency (ms/tile) | Speedup |"
        )
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for b_res in baseline_results:
            m12 = b_res.horizon_metrics.get("12h", HorizonMetrics(720, 0, 0, 0, 0, 0))
            lines.append(
                f"| {b_res.model_name} | {b_res.fire_name} | {b_res.uncensored_mae_minutes:.2f} | "
                f"{b_res.p10_p90_coverage * 100.0:.1f}% | {m12.f1:.3f} | {m12.iou:.3f} | "
                f"{b_res.inference_latency_ms:.2f} ms | {b_res.speedup_vs_teacher:.1f}x |"
            )

    return "\n".join(lines)
