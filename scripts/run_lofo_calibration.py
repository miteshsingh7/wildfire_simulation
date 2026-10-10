"""Execute Phase 1A Step 3 Leave-One-Fire-Out (LOFO) calibration, ablation, 20-seed ensemble, and freeze.

Strictly holds out Camp Fire. Evaluates:
1. Uncalibrated baseline (P_base = 0.50)
2. Single global slowdown baseline (LOFO & in-sample)
3. 6-parameter calibrated teacher fit on all 7 fires (LOFO & in-sample)
4. 6-parameter calibrated teacher fit on wind-driven fires only (Kincade, Glass, Zogg)
5. Plume-fire folds (Creek, Bobcat) reported separately
6. 20-seed ensemble P10-P90 arrival coverage
7. Parameter freeze to configs/teacher_v1.yaml with SHA-256 hash
"""

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from wf052.sim.teacher_calibrator import (
    FireDataset60m,
    TeacherParams,
    compute_composite_calibration_loss,
    evaluate_simulation_multiscale,
    load_fire_dataset_60m,
    simulate_teacher_60m,
)

FIRE_LIST: list[tuple[str, str]] = [
    ("Kincade", "wind-driven"),
    ("Bobcat", "plume"),
    ("Glass", "wind-driven"),
    ("Creek", "plume"),
    ("Zogg", "wind-driven"),
    ("Caldor", "terrain"),
    ("Tamarack", "terrain"),
]

WIND_FIRES = {"Kincade", "Glass", "Zogg"}
PLUME_FIRES = {"Creek", "Bobcat"}
TERRAIN_FIRES = {"Caldor", "Tamarack"}


def build_candidate_parameter_sets() -> tuple[list[TeacherParams], list[TeacherParams]]:
    """Construct structured candidate pools for (1) Global Slowdown and (2) 6-Parameter Calibration."""
    # 1. Global slowdown candidates (only varying p_base)
    slowdown_p_vals = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.21, 0.25, 0.30, 0.38, 0.50]
    slowdown_candidates = [TeacherParams(p_base=p) for p in slowdown_p_vals]

    # 2. Full 6-parameter candidates informed by OAT sensitivity
    # Includes local refinement around the global slowdown optimum (p_base ~ 0.21, c_2 ~ 0.15-0.22)
    # as well as higher-eccentricity wind-specialized candidates.
    full_candidates: list[TeacherParams] = []

    base_configs = [
        # Neighborhood around global slowdown optimum (p_base, c_2, c_1, a, gs, timber)
        (0.19, 0.15, 0.045, 0.06, 1.00, 1.00),
        (0.20, 0.15, 0.045, 0.06, 1.00, 1.00),
        (0.21, 0.15, 0.045, 0.06, 1.00, 1.00),
        (0.22, 0.15, 0.045, 0.06, 1.00, 1.00),
        (0.21, 0.16, 0.050, 0.07, 1.05, 0.90),
        (0.21, 0.17, 0.050, 0.07, 1.10, 0.85),
        (0.22, 0.17, 0.050, 0.07, 1.10, 0.85),
        (0.22, 0.18, 0.055, 0.07, 1.10, 0.85),
        (0.23, 0.18, 0.055, 0.07, 1.15, 0.80),
        (0.23, 0.19, 0.055, 0.08, 1.15, 0.80),
        (0.24, 0.19, 0.055, 0.08, 1.15, 0.80),
        (0.24, 0.20, 0.060, 0.08, 1.15, 0.80),
        (0.25, 0.20, 0.060, 0.08, 1.20, 0.75),
        (0.20, 0.16, 0.050, 0.08, 1.05, 0.90),
        (0.19, 0.16, 0.045, 0.08, 1.05, 0.90),
        # Moderate & higher eccentricity variants
        (0.18, 0.22, 0.045, 0.07, 1.05, 0.95),
        (0.20, 0.22, 0.050, 0.07, 1.10, 0.90),
        (0.22, 0.22, 0.055, 0.07, 1.10, 0.85),
        (0.24, 0.24, 0.055, 0.07, 1.15, 0.85),
        (0.26, 0.26, 0.060, 0.07, 1.15, 0.80),
        (0.26, 0.30, 0.060, 0.07, 1.15, 0.80),
        (0.28, 0.34, 0.060, 0.08, 1.15, 0.80),
    ]
    for pb, c2, c1, a_val, gs, tim in base_configs:
        full_candidates.append(
            TeacherParams(
                p_base=pb,
                a=a_val,
                c_1=c1,
                c_2=c2,
                p_continue=0.30,
                fuel_scale_grass_shrub=gs,
                fuel_scale_timber=tim,
                canopy_den_scale=1.0,
                wind_moisture_damping=0.0,
            )
        )

    return slowdown_candidates, full_candidates


def main() -> None:
    t_start = time.perf_counter()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing Phase 1A Step 3 LOFO Calibration on device: {device}")

    datasets: dict[str, FireDataset60m] = {}
    for fname, regime in FIRE_LIST:
        tdir = Path(f"data/tiles/fires/{fname.lower()}")
        datasets[fname] = load_fire_dataset_60m(fname, regime, tdir, device=device)

    slowdown_candidates, full_candidates = build_candidate_parameter_sets()
    uncalibrated_params = TeacherParams()

    # Combine unique candidates to evaluate once per fire and cache metrics & losses
    all_unique_params: list[TeacherParams] = [uncalibrated_params]
    for p in slowdown_candidates + full_candidates:
        if p not in all_unique_params:
            all_unique_params.append(p)

    print(f"Evaluating {len(all_unique_params)} candidate parameter vectors across 7 training fires (12h horizon)...")

    # Cache: metrics_cache[(fname, params)] -> dict[horizon, dict[metric, float]]
    # Loss cache: loss_cache[(fname, params)] -> float
    metrics_cache: dict[tuple[str, TeacherParams], dict[int, dict[str, float]]] = {}
    loss_cache: dict[tuple[str, TeacherParams], float] = {}

    horizons = (1, 3, 6, 12)

    for idx_p, p in enumerate(all_unique_params):
        t_p = time.perf_counter()
        for fname, _ in FIRE_LIST:
            ds = datasets[fname]
            sim_res = simulate_teacher_60m(ds, p, duration_hours=12, steps_per_hour=30, seed=42)
            m = evaluate_simulation_multiscale(ds, sim_res, horizons=horizons)
            loss_val = compute_composite_calibration_loss(m, p, l2_weight=0.02)
            metrics_cache[(fname, p)] = m
            loss_cache[(fname, p)] = loss_val

        mean_l = float(np.mean([loss_cache[(fn, p)] for fn, _ in FIRE_LIST]))
        print(
            f"  [{idx_p + 1:2d}/{len(all_unique_params)}] "
            f"p_base={p.p_base:.2f}, c_2={p.c_2:.2f}, c_1={p.c_1:.3f}, a={p.a:.2f}, "
            f"gs={p.fuel_scale_grass_shrub:.2f}, tim={p.fuel_scale_timber:.2f} "
            f"-> Mean 7-Fire Loss={mean_l:.4f} ({time.perf_counter() - t_p:.1f}s)"
        )

    # 1. Select In-Sample Best on All 7 Fires (for freezing to configs/teacher_v1.yaml)
    best_all7_insample = min(
        full_candidates,
        key=lambda p: float(np.mean([loss_cache[(fn, p)] for fn, _ in FIRE_LIST])),
    )
    best_slowdown_insample = min(
        slowdown_candidates,
        key=lambda p: float(np.mean([loss_cache[(fn, p)] for fn, _ in FIRE_LIST])),
    )
    best_wind_insample = min(
        full_candidates,
        key=lambda p: float(np.mean([loss_cache[(fn, p)] for fn in WIND_FIRES])),
    )

    print("\n=== Selected Parameter Sets (All-7 vs Global Slowdown vs Wind-Only) ===")
    print("  Uncalibrated Default :", uncalibrated_params.to_dict())
    print("  Global Slowdown (All7):", best_slowdown_insample.to_dict())
    print("  Calibrated (All 7)   :", best_all7_insample.to_dict())
    print("  Calibrated (WindOnly):", best_wind_insample.to_dict())

    # 2. Leave-One-Fire-Out (LOFO) Selection & Per-Fire Multi-Horizon Evaluation Table
    lofo_selected_params: dict[str, dict[str, TeacherParams]] = {}
    per_fire_rows: list[dict[str, Any]] = []

    for held_out_fire, regime in FIRE_LIST:
        train_fires_all = [fn for fn, _ in FIRE_LIST if fn != held_out_fire]
        train_fires_wind = [fn for fn in WIND_FIRES if fn != held_out_fire]

        p_slowdown_lofo = min(
            slowdown_candidates,
            key=lambda p: float(np.mean([loss_cache[(fn, p)] for fn in train_fires_all])),
        )
        p_all7_lofo = min(
            full_candidates,
            key=lambda p: float(np.mean([loss_cache[(fn, p)] for fn in train_fires_all])),
        )
        p_wind_lofo = min(
            full_candidates,
            key=lambda p: float(np.mean([loss_cache[(fn, p)] for fn in train_fires_wind])),
        )

        lofo_selected_params[held_out_fire] = {
            "uncalibrated": uncalibrated_params,
            "global_slowdown_lofo": p_slowdown_lofo,
            "calibrated_all7_lofo": p_all7_lofo,
            "calibrated_wind_only_lofo": p_wind_lofo,
            "calibrated_all7_insample": best_all7_insample,
        }

        print(
            f"  [LOFO Fold: {held_out_fire:8s}] "
            f"Slowdown p_base={p_slowdown_lofo.p_base:.2f} | "
            f"All7-LOFO (p_base={p_all7_lofo.p_base:.2f}, c_2={p_all7_lofo.c_2:.2f}) | "
            f"Wind-LOFO (p_base={p_wind_lofo.p_base:.2f}, c_2={p_wind_lofo.c_2:.2f})"
        )

        for model_name, p_vec in lofo_selected_params[held_out_fire].items():
            split_label = "in-sample" if "insample" in model_name else "held-out (LOFO)"
            m_dict = metrics_cache[(held_out_fire, p_vec)]
            for hr in horizons:
                mh = m_dict[hr]
                per_fire_rows.append({
                    "fire_name": held_out_fire,
                    "regime": regime,
                    "model": model_name,
                    "split": split_label,
                    "horizon_h": hr,
                    "iou_2km": round(mh["iou_2km"], 4),
                    "f1_2km": round(mh["f1_2km"], 4),
                    "area_ratio_sim_obs": round(mh["area_ratio"], 3),
                    "sim_area_ha": round(mh["sim_area_ha"], 1),
                    "obs_area_ha": round(mh["obs_area_ha"], 1),
                    "arrival_mae_min": round(mh["arrival_mae_min"], 1),
                    "arrival_mae_burned_min": round(mh["arrival_mae_burned_min"], 1),
                    "front_dist_err_km": round(mh["front_dist_err_km"], 2),
                    "front_dist_signed_km": round(mh["front_dist_signed_km"], 2),
                    "feds_375m_iou": round(mh["feds_375m_iou"], 4) if not np.isnan(mh["feds_375m_iou"]) else np.nan,
                    "feds_375m_f1": round(mh["feds_375m_f1"], 4) if not np.isnan(mh["feds_375m_f1"]) else np.nan,
                    "p_base": p_vec.p_base,
                    "c_2": p_vec.c_2,
                    "c_1": p_vec.c_1,
                    "a": p_vec.a,
                })

    per_fire_df = pd.DataFrame(per_fire_rows)
    per_fire_csv = Path("reports/lofo_per_fire_metrics.csv")
    per_fire_df.to_csv(per_fire_csv, index=False)

    # 3. Regime & Ablation Summary Table (All 7 vs Wind-Only vs Plume Folds Creek/Bobcat vs Terrain Folds)
    ablation_rows: list[dict[str, Any]] = []
    subsets = [
        ("All 7 Fires", {fn for fn, _ in FIRE_LIST}),
        ("Wind-Driven Folds (Kincade, Glass, Zogg)", WIND_FIRES),
        ("Plume-Fire Folds (Creek, Bobcat)", PLUME_FIRES),
        ("Terrain Folds (Caldor, Tamarack)", TERRAIN_FIRES),
    ]

    for subset_label, fire_set in subsets:
        for model_name in [
            "uncalibrated",
            "global_slowdown_lofo",
            "calibrated_all7_lofo",
            "calibrated_wind_only_lofo",
            "calibrated_all7_insample",
        ]:
            for hr in horizons:
                sub = per_fire_df[
                    (per_fire_df["fire_name"].isin(fire_set))
                    & (per_fire_df["model"] == model_name)
                    & (per_fire_df["horizon_h"] == hr)
                ]
                ablation_rows.append({
                    "evaluation_group": subset_label,
                    "model": model_name,
                    "split": sub["split"].iloc[0],
                    "horizon_h": hr,
                    "mean_iou_2km": round(float(sub["iou_2km"].mean()), 4),
                    "mean_f1_2km": round(float(sub["f1_2km"].mean()), 4),
                    "mean_area_ratio": round(float(sub["area_ratio_sim_obs"].mean()), 3),
                    "median_area_ratio": round(float(sub["area_ratio_sim_obs"].median()), 3),
                    "mean_arrival_mae_min": round(float(sub["arrival_mae_min"].mean()), 1),
                    "mean_front_dist_err_km": round(float(sub["front_dist_err_km"].mean()), 2),
                    "mean_feds_375m_f1": (
                        round(float(sub["feds_375m_f1"].dropna().mean()), 4)
                        if sub["feds_375m_f1"].notna().any()
                        else np.nan
                    ),
                })

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_csv = Path("reports/lofo_regime_ablation.csv")
    ablation_df.to_csv(ablation_csv, index=False)

    # 4. Run 20-Seed Ensemble for Each Fire using Held-Out LOFO Calibrated Parameters
    print("\nRunning 20-seed ensemble across all 7 fires (reporting P10-P90 arrival coverage)...")
    ensemble_seeds = [100 + 7 * i for i in range(20)]
    ensemble_rows: list[dict[str, Any]] = []

    for fname, regime in FIRE_LIST:
        ds = datasets[fname]
        p_lofo = lofo_selected_params[fname]["calibrated_all7_lofo"]
        arr_stack: list[np.ndarray] = []
        areas_12h: list[float] = []
        ious_12h: list[float] = []
        f1s_12h: list[float] = []

        for s_val in ensemble_seeds:
            sim_res = simulate_teacher_60m(ds, p_lofo, duration_hours=12, steps_per_hour=30, seed=s_val)
            m12 = evaluate_simulation_multiscale(ds, sim_res, horizons=(12,))[12]
            arr_stack.append(sim_res["arrival_time_min"])
            areas_12h.append(m12["sim_area_ha"])
            ious_12h.append(m12["iou_2km"])
            f1s_12h.append(m12["f1_2km"])

        arr_3d = np.stack(arr_stack, axis=0)  # [20, H, W]
        # Replace NaN (unburned in a seed) with 720.0 min (+12h horizon cap) for quantile computation
        arr_filled = np.where(np.isnan(arr_3d), 720.0, arr_3d)
        p10_arr = np.percentile(arr_filled, 10, axis=0)
        p50_arr = np.percentile(arr_filled, 50, axis=0)
        p90_arr = np.percentile(arr_filled, 90, axis=0)

        obs_arr = ds.gofer_arrival_min
        obs_valid = (~np.isnan(obs_arr)) & (obs_arr > 0.0) & (obs_arr <= 720.0)
        burned_any_seed = np.any(~np.isnan(arr_3d), axis=0) & obs_valid

        if np.any(burned_any_seed):
            o_t = obs_arr[burned_any_seed]
            p10_t = p10_arr[burned_any_seed]
            p90_t = p90_arr[burned_any_seed]
            # Exact P10-P90 coverage and hourly-quantized (GOFER dt=60m -> +/-30m half-step) P10-P90 coverage
            raw_cov = float(np.mean((o_t >= p10_t) & (o_t <= p90_t))) * 100.0
            hourly_bin_cov = float(np.mean((o_t >= (p10_t - 30.0)) & (o_t <= (p90_t + 30.0)))) * 100.0
            mean_spread_min = float(np.mean(p90_t - p10_t))
            p50_mae_min = float(np.mean(np.abs(p50_arr[burned_any_seed] - o_t)))
        else:
            raw_cov, hourly_bin_cov, mean_spread_min, p50_mae_min = 0.0, 0.0, 0.0, 720.0

        row_ens = {
            "fire_name": fname,
            "regime": regime,
            "split": "held-out (LOFO)",
            "num_seeds": 20,
            "obs_area_12h_ha": round(float(np.count_nonzero(ds.gofer_hourly_masks[12]) * 0.36), 1),
            "ens_area_12h_mean_ha": round(float(np.mean(areas_12h)), 1),
            "ens_area_12h_std_ha": round(float(np.std(areas_12h)), 1),
            "ens_iou_12h_mean": round(float(np.mean(ious_12h)), 4),
            "ens_iou_12h_std": round(float(np.std(ious_12h)), 4),
            "ens_f1_12h_mean": round(float(np.mean(f1s_12h)), 4),
            "ens_p50_arrival_mae_min": round(p50_mae_min, 1),
            "ens_p10_p90_width_min": round(mean_spread_min, 1),
            "p10_p90_raw_coverage_pct": round(raw_cov, 2),
            "p10_p90_hourly_quantized_coverage_pct": round(hourly_bin_cov, 2),
        }
        ensemble_rows.append(row_ens)
        print(
            f"  [20-Seed Ensemble: {fname:8s}] "
            f"Area={row_ens['ens_area_12h_mean_ha']:.1f}±{row_ens['ens_area_12h_std_ha']:.1f} ha | "
            f"IoU={row_ens['ens_iou_12h_mean']:.4f}±{row_ens['ens_iou_12h_std']:.4f} | "
            f"P10-P90 Width={row_ens['ens_p10_p90_width_min']:.1f}m | "
            f"Cov(raw/hourly)={row_ens['p10_p90_raw_coverage_pct']:.1f}% / {row_ens['p10_p90_hourly_quantized_coverage_pct']:.1f}%"
        )

    ens_df = pd.DataFrame(ensemble_rows)
    ens_csv = Path("reports/lofo_ensemble_20seeds.csv")
    ens_df.to_csv(ens_csv, index=False)

    # 5. Freeze Parameters into configs/teacher_v1.yaml + Compute SHA-256 Hash BEFORE Camp Fire test
    teacher_v1_cfg = {
        "version": "teacher_v1",
        "phase": "1A_step3",
        "operating_resolution_m": 60,
        "steps_per_hour": 30,
        "dt_seconds": 120.0,
        "crs": "EPSG:32610",
        "landfire_vintage": "LANDFIRE 2016 Remap (LF 2.0.0)",
        "calibrated_on_fires": [fn for fn, _ in FIRE_LIST],
        "held_out_evaluation_fire": "Camp Fire (2018)",
        "parameters": best_all7_insample.to_dict(),
        "ablation_wind_only_parameters": best_wind_insample.to_dict(),
        "baseline_global_slowdown_parameters": best_slowdown_insample.to_dict(),
        "uncalibrated_default_parameters": uncalibrated_params.to_dict(),
    }

    cfg_path = Path("configs/teacher_v1.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(teacher_v1_cfg, f, sort_keys=False)

    sha256_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    hash_path = Path("reports/teacher_v1_hash.txt")
    hash_path.write_text(f"{sha256_hash}  configs/teacher_v1.yaml\n")

    print(f"\nFrozen calibrated parameters written to {cfg_path}")
    print(f"SHA-256 Hash: {sha256_hash}")
    print(f"Total calibration & LOFO runtime: {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
