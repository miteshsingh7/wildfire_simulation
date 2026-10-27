"""Evaluation module for wildland fire spread surrogate models (Phase 3A).

METRIC DEFINITIONS AND PROTOCOLS (Docstring Reference):
------------------------------------------------------
1. Fidelity to Teacher:
   All metrics evaluate surrogate fidelity to the teacher simulation benchmark
   (frozen teacher_v2_ensemble), NEVER real-world accuracy.

2. Stratified Bins:
   Five K-mean burned-fraction strata defined by edges [0.0, 0.002, 0.02, 0.1, 0.5, 1.0]:
     - Bin 0: [0.0, 0.002)   (unburned / minimal spot)
     - Bin 1: [0.002, 0.02)  (small spread)
     - Bin 2: [0.02, 0.1)    (moderate spread)
     - Bin 3: [0.1, 0.5)     (large spread)
     - Bin 4: [0.5, 1.0]     (extreme / tile-scale run)
   Metrics are always evaluated and reported per bin, per split, and per fire.
   Never reported as unstratified population averages.
   Any stratum with sample count n < 5 is explicitly flagged as "LOW-N".

3. burn_prob Metrics:
   - Brier Score: Mean squared error over all grid cells:
       Brier = mean((p_pred - target_burn_prob)^2)
   - IoU (Intersection over Union):
       IoU of (p_pred >= 0.5) vs (target_burn_prob >= 0.5).
       If BOTH sets are empty, IoU is defined as float('nan') (NaN, not 0.0).

4. Arrival Metrics (q10, q50, q90):
   - Arrival MAE: Mean absolute error in raw minutes between predicted arrival and
     target arrival, evaluated strictly on finite-target cells.
     A cell is finite iff target_burn_prob >= ceil(tau * K) / K (uncensored).
   - False Unburned Share: Share of finite cells where predicted p < 0.5
     (i.e. model fails to flag arrival on a burning cell).

5. Quantile Ordering Constraint:
   - Ordering Violation Rate: Fraction of cells where q10 > q50 or q50 > q90.
     Evaluated over finite-target cells (and reported across all cells).

6. Baselines:
   - Baseline B0: Constant burn_prob equal to the train-set scalar mean burn_prob.
     Constant arrival time of 720.0 minutes for all quantiles.
   - Baseline B1: Non-parametric look-up baseline fitted on training fires only.
     Predicts burn_prob and q50 as a function of:
       (a) Euclidean distance to ignition cell (1 km bins up to 8 km: [0,1), [1,2), ..., [7,8), >=8 km).
       (b) Peak sustained wind speed bin of the 12 h series (<4, 4-8, >=8 m/s).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from wf052.surrogate.dataset import (
    CompactSurrogateDataset,
    collate_surrogate_batch,
)


BIN_EDGES = [0.0, 0.002, 0.02, 0.1, 0.5, 1.0]
BIN_NAMES = ["[0, 0.002)", "[0.002, 0.02)", "[0.02, 0.1)", "[0.1, 0.5)", "[0.5, 1.0]"]
DIST_BINS_KM = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, float("inf")]
WIND_BINS_MS = [0.0, 4.0, 8.0, float("inf")]


def compute_sample_metrics(
    pred_quantiles: np.ndarray,  # (3, H, W) in minutes
    pred_burn_prob: np.ndarray,  # (H, W) in [0, 1]
    target_quantiles: np.ndarray,  # (3, H, W) in minutes
    censor_quantiles: np.ndarray,  # (3, H, W) {0=uncensored, 1=censored}
    target_burn_prob: np.ndarray,  # (H, W) in [0, 1]
    k_ensemble: int = 4,
) -> dict[str, float]:
    """Compute all evaluation metrics for a single scenario sample."""
    h, w = target_burn_prob.shape

    # 1. Brier score over all cells
    brier = float(np.mean((pred_burn_prob - target_burn_prob) ** 2))

    # 2. IoU of (p >= 0.5) vs (target >= 0.5)
    pred_burned = pred_burn_prob >= 0.5
    true_burned = target_burn_prob >= 0.5
    intersection = np.logical_and(pred_burned, true_burned).sum()
    union = np.logical_or(pred_burned, true_burned).sum()

    if union == 0:
        iou = float("nan")
    else:
        iou = float(intersection / union)

    # 3. Quantile arrival MAEs on finite cells
    # Finite condition: target_burn_prob >= ceil(tau * K) / K (or censor_quantiles == 0)
    taus = [0.1, 0.5, 0.9]
    maes = []
    false_unburned_shares = []

    for q_idx, tau in enumerate(taus):
        req_prob = np.ceil(tau * k_ensemble) / float(k_ensemble)
        # Finite cells for this quantile
        finite_mask = (censor_quantiles[q_idx] == 0) & (target_burn_prob >= req_prob - 1e-5)
        n_finite = int(np.sum(finite_mask))

        if n_finite > 0:
            diff = np.abs(pred_quantiles[q_idx][finite_mask] - target_quantiles[q_idx][finite_mask])
            mae = float(np.mean(diff))
            maes.append(mae)

            # Share of finite cells where predicted p < 0.5
            p_under = float(np.mean(pred_burn_prob[finite_mask] < 0.5))
            false_unburned_shares.append(p_under)
        else:
            maes.append(float("nan"))
            false_unburned_shares.append(float("nan"))

    mae_q10, mae_q50, mae_q90 = maes
    share_unburned_q50 = false_unburned_shares[1]

    # 4. Ordering violations on finite cells of q50
    finite_q50 = censor_quantiles[1] == 0
    n_finite_q50 = int(np.sum(finite_q50))
    if n_finite_q50 > 0:
        v1 = pred_quantiles[0][finite_q50] > pred_quantiles[1][finite_q50]
        v2 = pred_quantiles[1][finite_q50] > pred_quantiles[2][finite_q50]
        order_viol_rate = float(np.mean(v1 | v2))
    else:
        # Check across all cells
        v1 = pred_quantiles[0] > pred_quantiles[1]
        v2 = pred_quantiles[1] > pred_quantiles[2]
        order_viol_rate = float(np.mean(v1 | v2))

    # 5. Evaluation guards (Phase 3C Step 4)
    pred_bp_std = float(np.std(pred_burn_prob))
    pred_q50_std = float(np.std(pred_quantiles[1]))
    pred_area_p05 = float(np.mean(pred_burn_prob >= 0.5))
    pred_area_p01 = float(np.mean(pred_burn_prob >= 0.1))
    target_area_mean = float(np.mean(target_burn_prob))
    target_area_p05 = float(np.mean(target_burn_prob >= 0.5))
    target_area_p01 = float(np.mean(target_burn_prob >= 0.1))

    return {
        "brier": brier,
        "iou": iou,
        "mae_q10": mae_q10,
        "mae_q50": mae_q50,
        "mae_q90": mae_q90,
        "false_unburned_share": share_unburned_q50,
        "ordering_violation_rate": order_viol_rate,
        "pred_bp_std": pred_bp_std,
        "pred_q50_std": pred_q50_std,
        "pred_area_p05": pred_area_p05,
        "pred_area_p01": pred_area_p01,
        "target_area_mean": target_area_mean,
        "target_area_p05": target_area_p05,
        "target_area_p01": target_area_p01,
    }


class BaselineB0:
    """Baseline B0: Constant train-mean burn probability, constant arrival 720 min."""

    def __init__(self, train_mean_burn_prob: float = 0.05) -> None:
        self.train_mean_burn_prob = train_mean_burn_prob
        self.arrival_min = 720.0

    def predict(self, shape: tuple[int, int] = (256, 256)) -> tuple[np.ndarray, np.ndarray]:
        """Return (pred_quantiles (3, H, W), pred_burn_prob (H, W))."""
        pred_bp = np.full(shape, self.train_mean_burn_prob, dtype=np.float32)
        pred_q = np.full((3, shape[0], shape[1]), self.arrival_min, dtype=np.float32)
        return pred_q, pred_bp


class BaselineB1:
    """Baseline B1: Look-up table on distance to ignition and wind-peak bin."""

    def __init__(
        self,
        table_burn_prob: np.ndarray,  # shape (n_dist_bins, n_wind_bins)
        table_q10: np.ndarray,        # shape (n_dist_bins, n_wind_bins)
        table_q50: np.ndarray,        # shape (n_dist_bins, n_wind_bins)
        table_q90: np.ndarray,        # shape (n_dist_bins, n_wind_bins)
    ) -> None:
        self.table_burn_prob = table_burn_prob
        self.table_q10 = table_q10
        self.table_q50 = table_q50
        self.table_q90 = table_q90

    def predict(
        self,
        ign_r: int,
        ign_c: int,
        peak_wind_ms: float,
        shape: tuple[int, int] = (256, 256),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict burn_prob and quantiles for a tile given ignition and peak wind."""
        h, w = shape
        # Wind bin: <4, 4-8, >=8
        if peak_wind_ms < 4.0:
            w_idx = 0
        elif peak_wind_ms < 8.0:
            w_idx = 1
        else:
            w_idx = 2

        # Distance grid in km
        rows, cols = np.ogrid[:h, :w]
        dist_km = np.hypot(rows - ign_r, cols - ign_c) * 0.060  # 60m resolution
        dist_idx = np.digitize(dist_km, DIST_BINS_KM[1:-1])  # 0 to 8

        pred_bp = self.table_burn_prob[dist_idx, w_idx]
        pred_q10 = self.table_q10[dist_idx, w_idx]
        pred_q50 = self.table_q50[dist_idx, w_idx]
        pred_q90 = self.table_q90[dist_idx, w_idx]

        pred_q = np.stack([pred_q10, pred_q50, pred_q90], axis=0)
        return pred_q, pred_bp


def fit_baseline_b0(dataset: CompactSurrogateDataset, train_fires: Sequence[str]) -> BaselineB0:
    """Fit Baseline B0 on training fires."""
    total_cells = 0
    total_bp_sum = 0.0

    for idx, f in enumerate(dataset.fire_ids):
        if f in train_fires:
            s = dataset[idx]
            bp = s["burn_prob"].numpy()
            total_bp_sum += float(np.sum(bp))
            total_cells += bp.size

    mean_bp = total_bp_sum / max(1, total_cells)
    return BaselineB0(train_mean_burn_prob=mean_bp)


def fit_baseline_b1(dataset: CompactSurrogateDataset, train_fires: Sequence[str]) -> BaselineB1:
    """Fit Baseline B1 (distance and peak wind lookup table) on training fires."""
    n_dist = len(DIST_BINS_KM) - 1  # 9 bins
    n_wind = len(WIND_BINS_MS) - 1  # 3 bins

    sum_bp = np.zeros((n_dist, n_wind), dtype=np.float64)
    cnt_bp = np.zeros((n_dist, n_wind), dtype=np.int64)

    sum_q10 = np.zeros((n_dist, n_wind), dtype=np.float64)
    cnt_q10 = np.zeros((n_dist, n_wind), dtype=np.int64)
    sum_q50 = np.zeros((n_dist, n_wind), dtype=np.float64)
    cnt_q50 = np.zeros((n_dist, n_wind), dtype=np.int64)
    sum_q90 = np.zeros((n_dist, n_wind), dtype=np.float64)
    cnt_q90 = np.zeros((n_dist, n_wind), dtype=np.int64)

    for idx, f in enumerate(dataset.fire_ids):
        if f in train_fires:
            s_idx, i = dataset.samples_index[idx]
            shard = dataset.loaded_shards[s_idx]

            ign_r, ign_c = int(shard["ignition_cells"][i, 0]), int(shard["ignition_cells"][i, 1])
            wind_uv = shard["wind_series"][i]
            peak_wind = float(np.max(np.hypot(wind_uv[:, 0], wind_uv[:, 1])))

            if peak_wind < 4.0:
                w_idx = 0
            elif peak_wind < 8.0:
                w_idx = 1
            else:
                w_idx = 2

            rows, cols = np.ogrid[:256, :256]
            dist_km = np.hypot(rows - ign_r, cols - ign_c) * 0.060
            d_idx = np.digitize(dist_km, DIST_BINS_KM[1:-1])

            bp = shard["burn_prob"][i].astype(np.float32) / 255.0
            q10 = shard["arrival_quantiles"][i, 0].astype(np.float32)
            q50 = shard["arrival_quantiles"][i, 1].astype(np.float32)
            q90 = shard["arrival_quantiles"][i, 2].astype(np.float32)
            c_packed_q10 = shard["censor_quantiles_packed"][i, 0]
            c_packed_q50 = shard["censor_quantiles_packed"][i, 1]
            c_packed_q90 = shard["censor_quantiles_packed"][i, 2]
            c_q10 = np.unpackbits(c_packed_q10).reshape(256, 256)
            c_q50 = np.unpackbits(c_packed_q50).reshape(256, 256)
            c_q90 = np.unpackbits(c_packed_q90).reshape(256, 256)

            for d in range(n_dist):
                mask_d = d_idx == d
                if np.any(mask_d):
                    sum_bp[d, w_idx] += float(np.sum(bp[mask_d]))
                    cnt_bp[d, w_idx] += int(np.sum(mask_d))

                    finite_q10 = mask_d & (c_q10 == 0)
                    if np.any(finite_q10):
                        sum_q10[d, w_idx] += float(np.sum(q10[finite_q10]))
                        cnt_q10[d, w_idx] += int(np.sum(finite_q10))

                    finite_q50 = mask_d & (c_q50 == 0)
                    if np.any(finite_q50):
                        sum_q50[d, w_idx] += float(np.sum(q50[finite_q50]))
                        cnt_q50[d, w_idx] += int(np.sum(finite_q50))

                    finite_q90 = mask_d & (c_q90 == 0)
                    if np.any(finite_q90):
                        sum_q90[d, w_idx] += float(np.sum(q90[finite_q90]))
                        cnt_q90[d, w_idx] += int(np.sum(finite_q90))

    table_bp = np.zeros((n_dist, n_wind), dtype=np.float32)
    table_q10 = np.full((n_dist, n_wind), 720.0, dtype=np.float32)
    table_q50 = np.full((n_dist, n_wind), 720.0, dtype=np.float32)
    table_q90 = np.full((n_dist, n_wind), 720.0, dtype=np.float32)

    for d in range(n_dist):
        for w in range(n_wind):
            if cnt_bp[d, w] > 0:
                table_bp[d, w] = float(sum_bp[d, w] / cnt_bp[d, w])
            if cnt_q10[d, w] > 0:
                table_q10[d, w] = float(sum_q10[d, w] / cnt_q10[d, w])
            if cnt_q50[d, w] > 0:
                table_q50[d, w] = float(sum_q50[d, w] / cnt_q50[d, w])
            if cnt_q90[d, w] > 0:
                table_q90[d, w] = float(sum_q90[d, w] / cnt_q90[d, w])

    return BaselineB1(table_burn_prob=table_bp, table_q10=table_q10, table_q50=table_q50, table_q90=table_q90)


def evaluate_model_on_dataset(
    model: torch.nn.Module,
    dataset: CompactSurrogateDataset,
    target_indices: Sequence[int] | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Evaluate a trained PyTorch surrogate model on specified dataset indices."""
    if device is None:
        device = torch.device(
            "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model.eval()
    model.to(device)

    indices = list(range(len(dataset))) if target_indices is None else list(target_indices)
    records = []

    with torch.no_grad():
        for idx in indices:
            item = dataset[idx]
            spatial = item["spatial_features"].unsqueeze(0).to(device)
            wind = item["wind_series"].unsqueeze(0).to(device)

            out = model(spatial, wind)
            pred_q = out.quantiles.squeeze(0).cpu().numpy()
            pred_bp = torch.sigmoid(out.burn_logits).squeeze().cpu().numpy()

            t_q = item["target_quantiles"].numpy()
            c_q = item["censor_quantiles"].numpy()
            t_bp = item["burn_prob"].numpy()
            fire = item["fire_name"]
            sample_id = item["sample_id"]

            s_idx, i = dataset.samples_index[idx]
            shard = dataset.loaded_shards[s_idx]
            # Retrieve or calculate bin_kmean
            mean_bf = float(np.mean(t_bp))
            bin_kmean = int(np.digitize(mean_bf, BIN_EDGES[1:-1]))

            m = compute_sample_metrics(
                pred_quantiles=pred_q,
                pred_burn_prob=pred_bp,
                target_quantiles=t_q,
                censor_quantiles=c_q,
                target_burn_prob=t_bp,
                k_ensemble=4,
            )
            records.append({
                "sample_id": sample_id,
                "fire": fire,
                "bin_kmean": bin_kmean,
                "bin_name": BIN_NAMES[bin_kmean],
                "brier": m["brier"],
                "iou": m["iou"],
                "mae_q10": m["mae_q10"],
                "mae_q50": m["mae_q50"],
                "mae_q90": m["mae_q90"],
                "false_unburned_share": m["false_unburned_share"],
                "ordering_violation_rate": m["ordering_violation_rate"],
                "pred_bp_std": m["pred_bp_std"],
                "pred_q50_std": m["pred_q50_std"],
                "pred_area_p05": m["pred_area_p05"],
                "pred_area_p01": m["pred_area_p01"],
                "target_area_mean": m["target_area_mean"],
                "target_area_p05": m["target_area_p05"],
                "target_area_p01": m["target_area_p01"],
            })

    return pd.DataFrame(records)


def evaluate_baseline_on_dataset(
    baseline: BaselineB0 | BaselineB1,
    dataset: CompactSurrogateDataset,
    target_indices: Sequence[int],
    baseline_name: str = "B0",
) -> pd.DataFrame:
    """Evaluate Baseline B0 or B1 on specified dataset indices."""
    records = []

    for idx in target_indices:
        item = dataset[idx]
        t_q = item["target_quantiles"].numpy()
        c_q = item["censor_quantiles"].numpy()
        t_bp = item["burn_prob"].numpy()
        fire = item["fire_name"]
        sample_id = item["sample_id"]

        s_idx, i = dataset.samples_index[idx]
        shard = dataset.loaded_shards[s_idx]

        ign_r, ign_c = int(shard["ignition_cells"][i, 0]), int(shard["ignition_cells"][i, 1])
        wind_uv = shard["wind_series"][i]
        peak_wind = float(np.max(np.hypot(wind_uv[:, 0], wind_uv[:, 1])))

        if isinstance(baseline, BaselineB0):
            pred_q, pred_bp = baseline.predict()
        else:
            pred_q, pred_bp = baseline.predict(ign_r=ign_r, ign_c=ign_c, peak_wind_ms=peak_wind)

        mean_bf = float(np.mean(t_bp))
        bin_kmean = int(np.digitize(mean_bf, BIN_EDGES[1:-1]))

        m = compute_sample_metrics(
            pred_quantiles=pred_q,
            pred_burn_prob=pred_bp,
            target_quantiles=t_q,
            censor_quantiles=c_q,
            target_burn_prob=t_bp,
            k_ensemble=4,
        )
        records.append({
            "model": baseline_name,
            "sample_id": sample_id,
            "fire": fire,
            "bin_kmean": bin_kmean,
            "bin_name": BIN_NAMES[bin_kmean],
            "brier": m["brier"],
            "iou": m["iou"],
            "mae_q10": m["mae_q10"],
            "mae_q50": m["mae_q50"],
            "mae_q90": m["mae_q90"],
            "false_unburned_share": m["false_unburned_share"],
            "ordering_violation_rate": m["ordering_violation_rate"],
            "pred_bp_std": m["pred_bp_std"],
            "pred_q50_std": m["pred_q50_std"],
            "pred_area_p05": m["pred_area_p05"],
            "pred_area_p01": m["pred_area_p01"],
            "target_area_mean": m["target_area_mean"],
            "target_area_p05": m["target_area_p05"],
            "target_area_p01": m["target_area_p01"],
        })

    return pd.DataFrame(records)


def check_evaluation_guards(df_val_samples: pd.DataFrame) -> list[str]:
    """Check evaluation guards on validation samples (Phase 3C Step 4).

    Prints a loud WARNING if:
    1. Mean std of predicted burn_prob across val samples is below 1e-3.
    2. Mean predicted p>=0.1 area exceeds 5x target area.

    Returns list of warning strings.
    """
    warnings: list[str] = []
    if df_val_samples.empty:
        return warnings

    mean_bp_std = float(df_val_samples["pred_bp_std"].mean())
    if mean_bp_std < 1e-3:
        warnings.append(
            f"GUARD FAILURE: Mean std of predicted burn_prob across val samples is {mean_bp_std:.6e} (< 1e-3)! "
            "Model predictions are essentially flat / uninformative!"
        )

    mean_pred_p01 = float(df_val_samples["pred_area_p01"].mean())
    mean_target_area = float(df_val_samples["target_area_mean"].mean())
    if mean_pred_p01 > 5.0 * max(mean_target_area, 1e-5):
        ratio = mean_pred_p01 / max(mean_target_area, 1e-5)
        warnings.append(
            f"GUARD FAILURE: Predicted p>=0.1 area ({mean_pred_p01:.4f}) exceeds 5x target area ({mean_target_area:.4f})! "
            f"Ratio: {ratio:.2f}x > 5.0x"
        )

    if warnings:
        print("\n" + "!" * 78)
        print("!!! LOUD EVALUATION GUARD WARNING !!!")
        for w in warnings:
            print(f"!!! WARNING: {w}")
        print("!" * 78 + "\n")

    return warnings


def aggregate_per_bin_and_fire_metrics(
    df_records: pd.DataFrame,
    fires: Sequence[str] | None = None,
    model_name: str = "Surrogate (UNet)",
) -> pd.DataFrame:
    """Aggregate records into table with breakdown per bin AND per fire, with n and LOW-N flags."""
    rows = []
    if "model" not in df_records.columns:
        df_records = df_records.copy()
        df_records["model"] = model_name

    m_name = df_records["model"].iloc[0] if not df_records.empty else model_name

    available_fires = list(fires) if fires is not None else sorted(df_records["fire"].unique())
    # Fire breakdown: each individual fire plus 'ALL'
    fire_subsets = [(f, df_records[df_records["fire"] == f]) for f in available_fires]
    fire_subsets.append(("ALL", df_records))

    for f_label, f_df in fire_subsets:
        all_bins = range(len(BIN_NAMES))
        for b in all_bins:
            sub = f_df[f_df["bin_kmean"] == b]
            n = len(sub)
            low_n_flag = "LOW-N" if n < 5 else "OK"
            b_name = BIN_NAMES[b]

            rows.append({
                "model": m_name,
                "fire": f_label,
                "bin": b,
                "bin_name": b_name,
                "n": n,
                "flag": low_n_flag,
                "brier": round(float(sub["brier"].mean()), 5) if n > 0 else float("nan"),
                "iou": round(float(sub["iou"].dropna().mean()), 4) if (n > 0 and not sub["iou"].dropna().empty) else float("nan"),
                "mae_q10": round(float(sub["mae_q10"].dropna().mean()), 2) if (n > 0 and not sub["mae_q10"].dropna().empty) else float("nan"),
                "mae_q50": round(float(sub["mae_q50"].dropna().mean()), 2) if (n > 0 and not sub["mae_q50"].dropna().empty) else float("nan"),
                "mae_q90": round(float(sub["mae_q90"].dropna().mean()), 2) if (n > 0 and not sub["mae_q90"].dropna().empty) else float("nan"),
                "false_unburned_share": round(float(sub["false_unburned_share"].dropna().mean()), 4) if (n > 0 and not sub["false_unburned_share"].dropna().empty) else float("nan"),
                "ordering_violation_rate": round(float(sub["ordering_violation_rate"].mean()), 5) if n > 0 else float("nan"),
                "pred_bp_std": round(float(sub["pred_bp_std"].mean()), 5) if n > 0 else float("nan"),
                "pred_area_p01": round(float(sub["pred_area_p01"].mean()), 4) if n > 0 else float("nan"),
                "target_area_mean": round(float(sub["target_area_mean"].mean()), 4) if n > 0 else float("nan"),
            })

    return pd.DataFrame(rows)


def aggregate_per_bin_metrics(df_records: pd.DataFrame, group_col: str = "bin_kmean") -> pd.DataFrame:
    """Aggregate sample records into clean per-bin metrics table with LOW-N annotation."""
    rows = []
    groups = sorted(df_records[group_col].unique())

    for g in groups:
        sub = df_records[df_records[group_col] == g]
        n = len(sub)
        low_n_flag = "LOW-N" if n < 5 else "OK"
        b_name = BIN_NAMES[g] if isinstance(g, (int, np.integer)) and 0 <= g < len(BIN_NAMES) else str(g)

        rows.append({
            "bin": g,
            "bin_name": b_name,
            "n": n,
            "flag": low_n_flag,
            "brier": round(float(sub["brier"].mean()), 5),
            "iou": round(float(sub["iou"].dropna().mean()), 4) if not sub["iou"].dropna().empty else float("nan"),
            "mae_q10": round(float(sub["mae_q10"].dropna().mean()), 2) if not sub["mae_q10"].dropna().empty else float("nan"),
            "mae_q50": round(float(sub["mae_q50"].dropna().mean()), 2) if not sub["mae_q50"].dropna().empty else float("nan"),
            "mae_q90": round(float(sub["mae_q90"].dropna().mean()), 2) if not sub["mae_q90"].dropna().empty else float("nan"),
            "false_unburned_share": round(float(sub["false_unburned_share"].dropna().mean()), 4) if not sub["false_unburned_share"].dropna().empty else float("nan"),
            "ordering_violation_rate": round(float(sub["ordering_violation_rate"].mean()), 5),
        })

    return pd.DataFrame(rows)
