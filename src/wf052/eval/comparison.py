"""Evaluation and validation metrics comparing fire simulation engines against NIST ground truth.

Computes arrival-time MAE/median error, fraction of unburned points (ember spotting gap),
and spatial perimeter IoU / F1 scores at +3h, +6h, and +12h horizons.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio


@dataclass(frozen=True)
class HorizonMetrics:
    """Metrics at a specific simulation horizon (+3h, +6h, +12h)."""

    horizon_hours: float
    simulated_burned_cells: int
    ground_truth_cells: int
    intersection_cells: int
    union_cells: int
    iou: float
    f1_score: float


@dataclass(frozen=True)
class EvaluationMetrics:
    """Comprehensive evaluation metrics comparing simulated spread to NIST ground truth."""

    engine: str
    seed: int
    wall_clock_seconds: float
    total_nist_points_12h: int
    burned_nist_points: int
    unburned_nist_points: int
    unburned_fraction_pct: float
    arrival_error_mae_minutes: float
    arrival_error_median_minutes: float
    arrival_error_std_minutes: float
    arrival_bias_minutes: float
    horizon_3h: HorizonMetrics
    horizon_6h: HorizonMetrics
    horizon_12h: HorizonMetrics


def evaluate_simulation(
    arrival_raster: np.ndarray,
    config: dict[str, Any],
    engine_name: str,
    seed: int,
    wall_clock_seconds: float = 0.0,
    tiles_dir: Path | str = "data/tiles",
) -> EvaluationMetrics:
    """Evaluate simulated arrival times against ground-truth NIST observations."""
    tiles_path = Path(tiles_dir)
    bbox_utm = config["domain"]["bbox_utm"]
    res = float(config["domain"]["resolution"])
    h, w = arrival_raster.shape

    # Load NIST vector observations
    obs_df = gpd.read_parquet(tiles_path / "observations_nist.parquet")
    # Restrict to 12-hour simulation window (0 to 720 minutes)
    obs_12h = obs_df[(obs_df["elapsed_minutes"] >= 0) & (obs_df["elapsed_minutes"] <= 720.0)].copy()

    # Sample arrival time at each point
    errors: list[float] = []
    biases: list[float] = []
    burned_count = 0
    unburned_count = 0

    for _, row in obs_12h.iterrows():
        x = float(row["utm_x"])
        y = float(row["utm_y"])
        t_obs = float(row["elapsed_minutes"])

        col = int(np.clip((x - bbox_utm["x_min"]) / res, 0, w - 1))
        row_idx = int(np.clip((bbox_utm["y_max"] - y) / res, 0, h - 1))

        sim_t = arrival_raster[row_idx, col]
        if not np.isnan(sim_t) and sim_t <= 720.0:
            burned_count += 1
            err = abs(float(sim_t) - t_obs)
            bias = float(sim_t) - t_obs
            errors.append(err)
            biases.append(bias)
        else:
            unburned_count += 1

    total_pts = len(obs_12h)
    unburned_pct = (unburned_count / total_pts * 100.0) if total_pts > 0 else 0.0
    mae = float(np.mean(errors)) if errors else float("nan")
    median_err = float(np.median(errors)) if errors else float("nan")
    std_err = float(np.std(errors)) if errors else float("nan")
    bias_mean = float(np.mean(biases)) if biases else float("nan")

    # Load rasterized ground truth arrival times
    with rasterio.open(tiles_path / "arrival_time_ground_truth.tif") as src:
        gt_arr = src.read(1)

    # Compute IoU and F1 at +3h (180 min), +6h (360 min), and +12h (720 min)
    horizon_metrics_dict: dict[float, HorizonMetrics] = {}
    for hours in (3.0, 6.0, 12.0):
        t_lim = hours * 60.0
        sim_mask = (~np.isnan(arrival_raster)) & (arrival_raster <= t_lim)
        gt_mask = (~np.isnan(gt_arr)) & (gt_arr <= t_lim)

        inter = int(np.count_nonzero(sim_mask & gt_mask))
        union = int(np.count_nonzero(sim_mask | gt_mask))
        sim_n = int(np.count_nonzero(sim_mask))
        gt_n = int(np.count_nonzero(gt_mask))

        iou = (inter / union) if union > 0 else 0.0
        f1 = (2.0 * inter / (sim_n + gt_n)) if (sim_n + gt_n) > 0 else 0.0

        horizon_metrics_dict[hours] = HorizonMetrics(
            horizon_hours=hours,
            simulated_burned_cells=sim_n,
            ground_truth_cells=gt_n,
            intersection_cells=inter,
            union_cells=union,
            iou=iou,
            f1_score=f1,
        )

    return EvaluationMetrics(
        engine=engine_name,
        seed=seed,
        wall_clock_seconds=wall_clock_seconds,
        total_nist_points_12h=total_pts,
        burned_nist_points=burned_count,
        unburned_nist_points=unburned_count,
        unburned_fraction_pct=unburned_pct,
        arrival_error_mae_minutes=mae,
        arrival_error_median_minutes=median_err,
        arrival_error_std_minutes=std_err,
        arrival_bias_minutes=bias_mean,
        horizon_3h=horizon_metrics_dict[3.0],
        horizon_6h=horizon_metrics_dict[6.0],
        horizon_12h=horizon_metrics_dict[12.0],
    )
