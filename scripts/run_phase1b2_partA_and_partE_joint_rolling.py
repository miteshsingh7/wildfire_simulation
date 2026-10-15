"""Phase 1B-2 Part A & Part E:
- Joint recalibration of (P_base, alpha_launch, beta_dist_m_per_ms, p_urban_wui) on
  Kincade, Glass, Zogg with LOFO under the rolling-horizon objective (re-init from
  every 3rd GOFER perimeter, forecast 1-6 h; loss weighted to 1-6 h).
- Compare Ember OFF (P_base refit under the same rolling protocol) vs Ember ON (joint fit).
- Report 12 h point-start as secondary only, including signed front error at 12 h to check overshoot.
- Run all 7 fires with the final vector, labeling in-sample vs held-out, with all denominators.
- Freeze configs/teacher_v2.yaml + SHA-256 hash (reports/teacher_v2_hash.txt).
"""

import hashlib
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from wf052.sim.ember_kernel import EmberParams
from wf052.sim.rolling_protocol import (
    evaluate_rolling_horizon_fire,
    get_active_rolling_origins,
)
from wf052.sim.teacher_calibrator import FireDataset60m, TeacherParams, load_fire_dataset_60m


EvalKey = tuple[str, bool, float, float, float, float]


def get_cached_fire_eval(
    cache: dict[EvalKey, dict[str, Any]],
    ds: FireDataset60m,
    p_base: float,
    ember_params: EmberParams,
    use_embers: bool,
) -> dict[str, Any]:
    """Evaluate (or retrieve from cache) rolling 1-6h + secondary 12h metrics for a single fire."""
    key: EvalKey = (
        ds.fire_name,
        bool(use_embers),
        round(float(p_base), 4),
        round(float(ember_params.alpha_launch if use_embers else 0.0), 6),
        round(float(ember_params.beta_dist_m_per_ms if use_embers else 0.0), 2),
        round(float(ember_params.p_urban_wui if use_embers else 0.0), 4),
    )
    if key not in cache:
        cache[key] = evaluate_rolling_horizon_fire(
            ds=ds,
            teacher_params=TeacherParams(p_base=float(p_base)),
            ember_params=ember_params,
            use_ember_kernel=use_embers,
            horizons_h=(1, 3, 6),
            seed=42,
            include_12h_secondary=True,
        )
    return cache[key]


def evaluate_candidate_on_fires(
    cache: dict[EvalKey, dict[str, Any]],
    fires: list[FireDataset60m],
    p_base: float,
    ember_params: EmberParams,
    use_embers: bool,
) -> float:
    """Compute mean rolling 1-6h objective across a list of training fires using per-fire cache."""
    losses = [
        float(get_cached_fire_eval(cache, ds, p_base, ember_params, use_embers)["rolling_loss"])
        for ds in fires
    ]
    return float(np.mean(losses))


def fit_ember_off_rolling(
    cache: dict[EvalKey, dict[str, Any]],
    train_fires: list[FireDataset60m],
) -> TeacherParams:
    """Refit P_base with Ember OFF under the 1-6h rolling-horizon protocol on supercritical domain."""
    p_grid = [0.19, 0.20, 0.21, 0.22, 0.23]
    best_p = 0.21
    best_loss = float("inf")
    dummy_e = EmberParams(alpha_launch=0.0, beta_dist_m_per_ms=100.0, p_urban_wui=0.0)
    for p_val in p_grid:
        lv = evaluate_candidate_on_fires(cache, train_fires, p_val, dummy_e, use_embers=False)
        if lv < best_loss:
            best_loss = lv
            best_p = p_val
    return TeacherParams(p_base=best_p)


def fit_joint_ember_on_rolling(
    cache: dict[EvalKey, dict[str, Any]],
    train_fires: list[FireDataset60m],
) -> tuple[TeacherParams, EmberParams]:
    """Jointly fit (P_base, alpha_launch, beta_dist_m_per_ms, p_urban_wui) under the 1-6h rolling protocol."""
    p_grid = [0.19, 0.20, 0.21, 0.22, 0.23]
    alpha_grid = [0.0015, 0.0030, 0.0045]
    beta_wui_candidates = [
        (35.0, 0.14),
        (60.0, 0.15),
        (85.0, 0.16),
        (110.0, 0.16),
    ]

    best_p = 0.20
    best_e = EmberParams(alpha_launch=0.0030, beta_dist_m_per_ms=85.0, p_urban_wui=0.16)
    best_loss = float("inf")

    for p_val in p_grid:
        for alpha in alpha_grid:
            cand_e = EmberParams(
                alpha_launch=alpha,
                beta_dist_m_per_ms=85.0,
                p_urban_wui=0.16,
            )
            lv = evaluate_candidate_on_fires(cache, train_fires, p_val, cand_e, use_embers=True)
            if lv < best_loss:
                best_loss = lv
                best_p = p_val
                best_e = cand_e

    # Stage 2: Jointly refine (beta_dist_m_per_ms, p_urban_wui) and neighbor P_base around Stage-1 winner
    base_alpha = best_e.alpha_launch
    for dp in (-0.01, 0.0, 0.01):
        p_cand = round(best_p + dp, 3)
        if not (0.19 <= p_cand <= 0.23):
            continue
        for beta, p_wui in beta_wui_candidates:
            cand_e = EmberParams(
                alpha_launch=base_alpha,
                beta_dist_m_per_ms=beta,
                p_urban_wui=p_wui,
            )
            lv = evaluate_candidate_on_fires(cache, train_fires, p_cand, cand_e, use_embers=True)
            if lv < best_loss:
                best_loss = lv
                best_p = p_cand
                best_e = cand_e

    return TeacherParams(p_base=best_p), best_e


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Phase 1B-2 Part A & Part E (Joint Rolling Recalibration) on {device}...", flush=True)
    t0 = time.perf_counter()

    setup_df = pd.read_csv("reports/phase_1a_fires_setup.csv")
    datasets: dict[str, FireDataset60m] = {}
    for _, row in setup_df.iterrows():
        fname = str(row["fire_name"])
        regime = str(row["regime"])
        t_dir = Path("data/tiles/fires") / fname.lower()
        datasets[fname] = load_fire_dataset_60m(fname, regime, t_dir, device=device)
        origins = get_active_rolling_origins(datasets[fname])
        print(f"  Loaded {fname:10s} ({regime:11s}) | Active 3h rolling origins: {origins}", flush=True)

    wind_names = ["Kincade", "Glass", "Zogg"]
    all_names = list(datasets.keys())
    eval_cache: dict[EvalKey, dict[str, Any]] = {}

    # 1. LOFO Fitting on Wind-3 Folds (Ember OFF vs Joint Ember ON)
    lofo_off_params: dict[str, TeacherParams] = {}
    lofo_on_params: dict[str, tuple[TeacherParams, EmberParams]] = {}

    print("\n--- Fitting LOFO on Wind-Driven Folds (Kincade, Glass, Zogg) ---", flush=True)
    for held_w in wind_names:
        train_w = [datasets[f] for f in wind_names if f != held_w]
        tp_off = fit_ember_off_rolling(eval_cache, train_w)
        tp_on, ep_on = fit_joint_ember_on_rolling(eval_cache, train_w)
        lofo_off_params[held_w] = tp_off
        lofo_on_params[held_w] = (tp_on, ep_on)
        print(
            f"  Hold-Out={held_w:8s} | Ember OFF P_base={tp_off.p_base:.3f} | "
            f"Joint Ember ON: P_base={tp_on.p_base:.3f}, alpha={ep_on.alpha_launch:.4f}, "
            f"beta={ep_on.beta_dist_m_per_ms:.1f}, p_wui={ep_on.p_urban_wui:.2f} "
            f"(cache_size={len(eval_cache)})",
            flush=True,
        )

    # 2. Consensus Fit on all 3 Wind-Driven Folds (used as final teacher_v2 vector & held-out on Other-4)
    print("\n--- Fitting Consensus Vector on All 3 Wind-Driven Fires ---", flush=True)
    all_wind_ds = [datasets[f] for f in wind_names]
    cons_tp_off = fit_ember_off_rolling(eval_cache, all_wind_ds)
    cons_tp_on, cons_ep_on = fit_joint_ember_on_rolling(eval_cache, all_wind_ds)
    print(
        f"  Consensus Ember OFF: P_base={cons_tp_off.p_base:.3f}\n"
        f"  Consensus Joint Ember ON (teacher_v2): P_base={cons_tp_on.p_base:.3f}, "
        f"alpha_launch={cons_ep_on.alpha_launch:.4f}, beta_dist_m_per_ms={cons_ep_on.beta_dist_m_per_ms:.1f}, "
        f"p_urban_wui={cons_ep_on.p_urban_wui:.2f}",
        flush=True,
    )

    for fname in all_names:
        if fname not in lofo_off_params:
            lofo_off_params[fname] = cons_tp_off
            lofo_on_params[fname] = (cons_tp_on, cons_ep_on)

    # 3. Evaluate all 7 fires across (a) LOFO Ember OFF, (b) LOFO Joint Ember ON,
    #    (c) Final Consensus Ember OFF, (d) Final Consensus Joint Ember ON (teacher_v2)
    per_fire_horizon_rows: list[dict[str, Any]] = []
    secondary_12h_rows: list[dict[str, Any]] = []

    zero_ember = EmberParams(alpha_launch=0.0, beta_dist_m_per_ms=100.0, p_urban_wui=0.0)

    for fname in all_names:
        ds = datasets[fname]
        is_wind = fname in wind_names

        eval_configs = [
            (
                "Ember OFF (Rolling LOFO Refit)",
                False,
                lofo_off_params[fname],
                zero_ember,
                "held-out (LOFO)" if is_wind else "held-out (unseen in fit)",
            ),
            (
                "Ember ON (Joint Rolling LOFO Fit)",
                True,
                lofo_on_params[fname][0],
                lofo_on_params[fname][1],
                "held-out (LOFO)" if is_wind else "held-out (unseen in fit)",
            ),
            (
                "Ember OFF (Final Consensus Vector)",
                False,
                cons_tp_off,
                zero_ember,
                "in-sample (Wind-3)" if is_wind else "held-out (unseen in fit)",
            ),
            (
                "Ember ON (Final teacher_v2 Joint Vector)",
                True,
                cons_tp_on,
                cons_ep_on,
                "in-sample (Wind-3)" if is_wind else "held-out (unseen in fit)",
            ),
        ]

        for mode_label, use_e, tp_curr, ep_curr, split_label in eval_configs:
            res = get_cached_fire_eval(
                cache=eval_cache,
                ds=ds,
                p_base=tp_curr.p_base,
                ember_params=ep_curr,
                use_embers=use_e,
            )
            w_df = pd.DataFrame(res["window_rows"])
            n_origins = len(w_df["t0_h"].unique())

            for dt_h in (1, 3, 6):
                sub = w_df[w_df["horizon_h"] == dt_h]
                tp_2km_sum = int(sub["tp_2km"].sum())
                un_2km_sum = int(sub["denom_obs_2km_union"].sum())
                tp_prog_sum = int(sub["tp_prog_60m"].sum())
                un_prog_sum = int(sub["union_prog_60m"].sum())
                obs_prog_sum = int(sub["obs_prog_cells_60m"].sum())
                arr_denom_sum = int(sub["denom_arrival_cells"].sum())

                per_fire_horizon_rows.append(
                    {
                        "fire_name": fname,
                        "regime": ds.regime,
                        "model_mode": mode_label,
                        "split": split_label,
                        "rolling_origins_n": n_origins,
                        "horizon_h": dt_h,
                        "p_base": tp_curr.p_base,
                        "alpha_launch": ep_curr.alpha_launch if use_e else 0.0,
                        "beta_dist_m_per_ms": ep_curr.beta_dist_m_per_ms if use_e else 0.0,
                        "p_urban_wui": ep_curr.p_urban_wui if use_e else 0.0,
                        "mean_iou_2km": float(sub["iou_2km"].mean()),
                        "pooled_2km_tp_over_union": f"{tp_2km_sum}/{un_2km_sum} ({tp_2km_sum / max(un_2km_sum, 1):.4f})",
                        "mean_f1_2km": float(sub["f1_2km"].mean()),
                        "mean_prog_iou_60m": float(sub["prog_iou_60m"].mean()),
                        "pooled_prog_tp_over_union": f"{tp_prog_sum}/{un_prog_sum} ({tp_prog_sum / max(un_prog_sum, 1):.4f})",
                        "pooled_prog_recall_with_denom": f"{tp_prog_sum}/{obs_prog_sum} ({100.0 * tp_prog_sum / max(obs_prog_sum, 1):.1f}%)",
                        "mean_area_ratio": float(sub["area_ratio"].mean()),
                        "mean_front_err_km": float(sub["front_dist_err_km"].mean()),
                        "mean_front_signed_km": float(sub["front_signed_km"].mean()),
                        "mean_arrival_mae_min": float(sub["arrival_mae_min"].mean()),
                        "total_arrival_denom_cells": arr_denom_sum,
                    }
                )

            sec = res["secondary_12h"]
            tp12 = int(sec["tp_2km_12h"])
            un12 = int(sec["union_2km_12h"])
            secondary_12h_rows.append(
                {
                    "fire_name": fname,
                    "regime": ds.regime,
                    "model_mode": mode_label,
                    "split": split_label,
                    "metric_role": "secondary_only (12h point-start from T0=0)",
                    "p_base": tp_curr.p_base,
                    "alpha_launch": ep_curr.alpha_launch if use_e else 0.0,
                    "iou_2km_12h": sec["iou_2km_12h"],
                    "iou_2km_12h_with_denom": f"{tp12}/{un12} ({sec['iou_2km_12h']:.4f})",
                    "area_ratio_12h": sec["area_ratio_12h"],
                    "front_err_12h_km": sec["front_err_12h_km"],
                    "front_signed_12h_km": sec["front_signed_12h_km"],
                }
            )

    pf_df = pd.DataFrame(per_fire_horizon_rows)
    sec_df = pd.DataFrame(secondary_12h_rows)

    out_pf = Path("reports/phase1b2_partA_rolling_per_fire.csv")
    out_sec = Path("reports/phase1b2_partA_secondary_12h.csv")
    pf_df.to_csv(out_pf, index=False)
    sec_df.to_csv(out_sec, index=False)

    # 4. Aggregate Summary Table across Fold Groups at +1h, +3h, +6h, and Mean 1-6h
    summary_rows: list[dict[str, Any]] = []
    fold_groups = [
        ("Wind-Driven Folds (Kincade, Glass, Zogg) (n=3)", wind_names),
        ("Held-Out Non-Wind Folds (Bobcat, Creek, Caldor, Tamarack) (n=4)", [f for f in all_names if f not in wind_names]),
        ("All 7 Fires (n=7)", all_names),
    ]

    for grp_label, grp_fires in fold_groups:
        sub_g = pf_df[pf_df["fire_name"].isin(grp_fires)]
        sec_g = sec_df[sec_df["fire_name"].isin(grp_fires)]
        for mode_label in pf_df["model_mode"].unique():
            sub_m = sub_g[sub_g["model_mode"] == mode_label]
            sec_m = sec_g[sec_g["model_mode"] == mode_label]
            split_str = " / ".join(sorted(sub_m["split"].unique()))

            for h_val in [1, 3, 6, "1-6h Mean"]:
                sub_h = sub_m if h_val == "1-6h Mean" else sub_m[sub_m["horizon_h"] == h_val]
                summary_rows.append(
                    {
                        "fold_group": grp_label,
                        "model_mode": mode_label,
                        "split": split_str,
                        "horizon": str(h_val),
                        "mean_iou_2km": float(sub_h["mean_iou_2km"].mean()),
                        "mean_f1_2km": float(sub_h["mean_f1_2km"].mean()),
                        "mean_prog_iou_60m": float(sub_h["mean_prog_iou_60m"].mean()),
                        "mean_area_ratio": float(sub_h["mean_area_ratio"].mean()),
                        "mean_front_err_km": float(sub_h["mean_front_err_km"].mean()),
                        "mean_front_signed_km": float(sub_h["mean_front_signed_km"].mean()),
                        "mean_arrival_mae_min": float(sub_h["mean_arrival_mae_min"].mean()),
                        "secondary_12h_iou_2km": float(sec_m["iou_2km_12h"].mean()),
                        "secondary_12h_front_signed_km": float(sec_m["front_signed_12h_km"].mean()),
                    }
                )

    sum_df = pd.DataFrame(summary_rows)
    out_sum = Path("reports/phase1b2_partA_rolling_summary.csv")
    sum_df.to_csv(out_sum, index=False)

    print("\n=== Part A Rolling Summary (1-6h Mean & Secondary 12h Signed Front Error) ===")
    print(
        sum_df[sum_df["horizon"] == "1-6h Mean"][
            [
                "fold_group",
                "model_mode",
                "split",
                "mean_iou_2km",
                "mean_prog_iou_60m",
                "mean_front_err_km",
                "mean_arrival_mae_min",
                "secondary_12h_iou_2km",
                "secondary_12h_front_signed_km",
            ]
        ].to_string(index=False)
    )

    # 5. Part E: Freeze configs/teacher_v2.yaml + SHA-256 hash
    v2_dict = {
        "model_name": "pytorchfire_teacher_v2_joint_rolling",
        "version": "2.0.0",
        "operating_resolution_m": 60,
        "crs": "EPSG:32610",
        "steps_per_hour": 30,
        "calibration_protocol": (
            "Jointly calibrated (P_base, alpha_launch, beta_dist_m_per_ms, p_urban_wui) on "
            "Kincade, Glass, Zogg under rolling-horizon 1-6h objective (re-init every 3rd GOFER perimeter). "
            "Camp Fire (development) and fresh held-out fires (Slater/Devil, Walker, Cold Springs) strictly unseen."
        ),
        "contact_parameters": cons_tp_on.to_dict(),
        "ember_parameters": cons_ep_on.to_dict(),
        "use_ember_kernel": True,
        "rolling_ember_off_baseline": cons_tp_off.to_dict(),
    }
    cfg_v2_path = Path("configs/teacher_v2.yaml")
    with open(cfg_v2_path, "w") as f:
        yaml.safe_dump(v2_dict, f, sort_keys=False)

    sha256_v2 = hashlib.sha256(cfg_v2_path.read_bytes()).hexdigest()
    hash_v2_path = Path("reports/teacher_v2_hash.txt")
    hash_v2_path.write_text(f"{sha256_v2}  configs/teacher_v2.yaml\n")
    print(f"\nFrozen {cfg_v2_path} | SHA-256: {sha256_v2}")

    # 6. Plot Part A Rolling-Horizon Comparison
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=150)
    w_sub = sum_df[
        (sum_df["fold_group"].str.startswith("Wind-Driven"))
        & (sum_df["horizon"].isin(["1", "3", "6"]))
    ]
    horizons_num = [1, 3, 6]
    for mode_l, col, ls in [
        ("Ember OFF (Rolling LOFO Refit)", "#1f77b4", "--"),
        ("Ember ON (Joint Rolling LOFO Fit)", "#d62728", "-"),
        ("Ember ON (Final teacher_v2 Joint Vector)", "#2ca02c", "-."),
    ]:
        m_rows = w_sub[w_sub["model_mode"] == mode_l]
        axes[0].plot(horizons_num, m_rows["mean_iou_2km"], marker="o", color=col, linestyle=ls, linewidth=2.0, label=mode_l)
        axes[1].plot(horizons_num, m_rows["mean_front_err_km"], marker="s", color=col, linestyle=ls, linewidth=2.0, label=mode_l)

    axes[0].set_xticks(horizons_num)
    axes[0].set_xticklabels(["+1 h", "+3 h", "+6 h"])
    axes[0].set_ylabel("Rolling ~2 km GOFER IoU (Mean across 3h Origins)")
    axes[0].set_title("A. Wind-Driven Folds (LOFO Held-Out): Rolling 1–6 h IoU", fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8.5)

    axes[1].set_xticks(horizons_num)
    axes[1].set_xticklabels(["+1 h", "+3 h", "+6 h"])
    axes[1].set_ylabel("Rolling Wind-Axis Front Distance Error (km)")
    axes[1].set_title("B. Wind-Driven Folds (LOFO Held-Out): Rolling 1–6 h Front Error", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    out_plot = Path("reports/phase_1b2_partA_rolling_ablation.png")
    plt.savefig(out_plot, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_plot} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
