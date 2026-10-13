"""Phase 1B Step 2: Fit 3-parameter EmberParams on Training Fires Only (Kincade, Glass, Zogg)
and run Ember ON vs OFF ablation across all 7 folds.

Strictly DOES NOT touch or tune on Camp Fire.
Uses perimeter-level metrics:
- Area ratio (sim/obs) at +1h, +3h, +6h, +12h
- Downwind front distance error (km) at +1h, +3h, +6h, +12h
- Far-field jump cells (>300m ahead of main contiguous front)
- ~2km GOFER IoU/F1 and 375m FEDS VIIRS F1
"""

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from wf052.sim.ember_kernel import (
    EmberParams,
    compute_far_field_jump_cells,
    simulate_teacher_with_embers_60m,
)
from wf052.sim.teacher_calibrator import (
    FireDataset60m,
    TeacherParams,
    evaluate_simulation_multiscale,
    load_fire_dataset_60m,
)


def compute_ember_perimeter_loss(
    ds: FireDataset60m,
    sim_out: dict[str, Any],
    horizons: tuple[int, ...] = (1, 3, 6, 12),
) -> tuple[float, dict[int, dict[str, float]]]:
    """Compute perimeter-level calibration loss (area ratio, front distance, jump cells, WUI overlap)."""
    m_by_h = evaluate_simulation_multiscale(ds, sim_out, horizons=horizons)
    is_wui = ds.fbfm_arr == 91

    losses: list[float] = []
    for hr in horizons:
        if hr not in m_by_h:
            continue
        m = m_by_h[hr]
        s_mask = sim_out["hourly_sim_masks"][hr]
        o_mask = ds.gofer_hourly_masks[hr]

        n_jump_sim = compute_far_field_jump_cells(s_mask, min_jump_cells=5)
        n_jump_obs = compute_far_field_jump_cells(o_mask, min_jump_cells=5)
        m["far_field_jump_cells_sim"] = float(n_jump_sim)
        m["far_field_jump_cells_obs"] = float(n_jump_obs)

        # WUI (code 91) burned cells inside observed perimeter vs simulated
        wui_obs = int(np.count_nonzero(o_mask & is_wui))
        wui_sim = int(np.count_nonzero(s_mask & is_wui))
        wui_tp = int(np.count_nonzero(s_mask & o_mask & is_wui))
        m["wui_obs_cells"] = float(wui_obs)
        m["wui_sim_cells"] = float(wui_sim)
        m["wui_tp_cells"] = float(wui_tp)

        area_loss = abs(float(np.log(np.clip(m["area_ratio"], 0.05, 20.0))))
        front_loss = m["front_dist_err_km"] / 8.0
        dice_loss = m["dice_loss_2km"]

        # Far-field jump regularization: encourage realistic non-zero spotting (10..120 jump cells)
        # without runaway speckling (>250 jump cells)
        if n_jump_sim == 0 and hr in (3, 6, 12):
            jump_pen = 0.15
        elif n_jump_sim > 200:
            jump_pen = 0.002 * (n_jump_sim - 200)
        else:
            jump_pen = 0.0

        # WUI recall bonus on training fires where GOFER perimeter crosses WUI roads/towns
        wui_loss = 0.0
        if wui_obs > 10:
            wui_rec = wui_tp / max(wui_obs, 1)
            wui_prec = wui_tp / max(wui_sim, 1) if wui_sim > 0 else 0.0
            wui_f1 = (2.0 * wui_rec * wui_prec) / max(wui_rec + wui_prec, 1e-6)
            wui_loss = 0.15 * (1.0 - wui_f1)

        losses.append(1.0 * dice_loss + 0.6 * area_loss + 0.5 * front_loss + jump_pen + wui_loss)

    return float(np.mean(losses)), m_by_h


def fit_ember_params_on_wind_fires(
    train_fires: list[FireDataset60m],
    base_teacher: TeacherParams,
) -> EmberParams:
    """Coordinate search over the 3 free EmberParams on training wind fires ONLY."""
    alpha_grid = [0.002, 0.004, 0.006, 0.008, 0.011]
    beta_grid = [70.0, 100.0, 130.0, 160.0]
    wui_grid = [0.08, 0.15, 0.25]

    cur = EmberParams(alpha_launch=0.006, beta_dist_m_per_ms=110.0, p_urban_wui=0.15)
    best_loss = float("inf")

    def eval_cand(cand: EmberParams) -> float:
        f_losses = []
        for ds in train_fires:
            sim_out = simulate_teacher_with_embers_60m(
                ds,
                base_teacher,
                ember_params=cand,
                use_ember_kernel=True,
                duration_hours=12,
                steps_per_hour=30,
                seed=42,
            )
            l_val, _ = compute_ember_perimeter_loss(ds, sim_out, horizons=(1, 3, 6, 12))
            f_losses.append(l_val)
        return float(np.mean(f_losses))

    # Coordinate search along alpha_launch, beta_dist_m_per_ms, p_urban_wui
    for alpha in alpha_grid:
        c = EmberParams(alpha_launch=alpha, beta_dist_m_per_ms=cur.beta_dist_m_per_ms, p_urban_wui=cur.p_urban_wui)
        lv = eval_cand(c)
        if lv < best_loss:
            best_loss = lv
            cur = c

    for beta in beta_grid:
        c = EmberParams(alpha_launch=cur.alpha_launch, beta_dist_m_per_ms=beta, p_urban_wui=cur.p_urban_wui)
        lv = eval_cand(c)
        if lv < best_loss:
            best_loss = lv
            cur = c

    for p_wui in wui_grid:
        c = EmberParams(alpha_launch=cur.alpha_launch, beta_dist_m_per_ms=cur.beta_dist_m_per_ms, p_urban_wui=p_wui)
        lv = eval_cand(c)
        if lv < best_loss:
            best_loss = lv
            cur = c

    return cur


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Phase 1B Step 2 Ember Kernel Calibration & Ablation on {device}...")
    t0 = time.perf_counter()

    setup_df = pd.read_csv("reports/phase_1a_fires_setup.csv")
    datasets: dict[str, FireDataset60m] = {}
    for _, row in setup_df.iterrows():
        fname = str(row["fire_name"])
        regime = str(row["regime"])
        t_dir = Path("data/tiles/fires") / fname.lower()
        datasets[fname] = load_fire_dataset_60m(fname, regime, t_dir, device=device)

    wind_names = ["Kincade", "Glass", "Zogg"]
    all_names = list(datasets.keys())

    # Base teacher chosen from Step 0(a): test both P_base=0.21 (teacher_v1) and P_base=0.22 (teacher_v1_1)
    base_teacher = TeacherParams(p_base=0.21)

    # 1. LOFO cross-validation of EmberParams across the 3 wind-driven fires:
    #    For each wind fire, fit EmberParams on the other 2 wind fires.
    #    For non-wind fires (Bobcat, Creek, Caldor, Tamarack), evaluate the consensus EmberParams
    #    fit on the 3 wind-driven fires (strictly held-out for those 4 fires!).
    print("\n--- Fitting EmberParams on Wind-Driven Training Folds (LOFO) ---")
    lofo_ember_params: dict[str, EmberParams] = {}
    for held_out_w in wind_names:
        train_w = [datasets[f] for f in wind_names if f != held_out_w]
        fit_e = fit_ember_params_on_wind_fires(train_w, base_teacher)
        lofo_ember_params[held_out_w] = fit_e
        print(
            f"  LOFO Hold-Out={held_out_w:8s} | Fit on {[d.fire_name for d in train_w]} -> "
            f"alpha_launch={fit_e.alpha_launch:.4f}, beta_dist={fit_e.beta_dist_m_per_ms:.1f} m/(m/s), "
            f"p_urban_wui={fit_e.p_urban_wui:.2f}"
        )

    # Full fit on all 3 wind-driven training fires (used as held-out evaluation on the 4 non-wind fires
    # and for freezing teacher_v2)
    consensus_ember = fit_ember_params_on_wind_fires([datasets[f] for f in wind_names], base_teacher)
    print(
        f"\nConsensus EmberParams (Fit on Kincade, Glass, Zogg; Camp Fire UNSEEN): "
        f"alpha_launch={consensus_ember.alpha_launch:.4f}, "
        f"beta_dist_m_per_ms={consensus_ember.beta_dist_m_per_ms:.1f}, "
        f"p_urban_wui={consensus_ember.p_urban_wui:.2f}"
    )

    for fname in all_names:
        if fname not in lofo_ember_params:
            lofo_ember_params[fname] = consensus_ember

    # 2. Evaluate Ablation: Ember OFF vs Ember ON (LOFO Held-Out and In-Sample) across all 7 folds
    per_fire_rows: list[dict[str, Any]] = []
    cell_ha = 0.36

    for fname in all_names:
        ds = datasets[fname]
        e_lofo = lofo_ember_params[fname]

        for mode_label, use_embers, e_params, split_tag in [
            ("Ember OFF (Contact-Only, P_base=0.21)", False, None, "held-out (LOFO)"),
            ("Ember ON (LOFO Fit)", True, e_lofo, "held-out (LOFO)"),
            ("Ember ON (Wind-3 Consensus)", True, consensus_ember, "in-sample (Wind-3) / held-out (Other-4)"),
        ]:
            sim_out = simulate_teacher_with_embers_60m(
                ds,
                base_teacher,
                ember_params=e_params,
                use_ember_kernel=use_embers,
                duration_hours=12,
                steps_per_hour=30,
                seed=42,
            )
            _, m_by_h = compute_ember_perimeter_loss(ds, sim_out, horizons=(1, 3, 6, 12))

            for hr in (1, 3, 6, 12):
                m = m_by_h[hr]
                s_mask = sim_out["hourly_sim_masks"][hr]
                o_mask = ds.gofer_hourly_masks[hr]
                sim_cells = int(np.count_nonzero(s_mask))
                obs_cells = int(np.count_nonzero(o_mask))

                per_fire_rows.append({
                    "fire_name": fname,
                    "regime": ds.regime,
                    "ablation_mode": mode_label,
                    "split": split_tag,
                    "horizon_h": hr,
                    "obs_cells_60m": obs_cells,
                    "sim_cells_60m": sim_cells,
                    "obs_area_ha": obs_cells * cell_ha,
                    "sim_area_ha": sim_cells * cell_ha,
                    "area_ratio_sim_obs": m["area_ratio"],
                    "iou_2km": m["iou_2km"],
                    "f1_2km": m["f1_2km"],
                    "feds_375m_f1": m["feds_375m_f1"],
                    "arrival_mae_min": m["arrival_mae_min"],
                    "front_sim_km": m["front_sim_km"],
                    "front_obs_km": m["front_obs_km"],
                    "front_dist_err_km": m["front_dist_err_km"],
                    "front_dist_signed_km": m["front_dist_signed_km"],
                    "far_field_jump_cells_sim": int(m["far_field_jump_cells_sim"]),
                    "far_field_jump_cells_obs": int(m["far_field_jump_cells_obs"]),
                    "wui_obs_cells": int(m["wui_obs_cells"]),
                    "wui_sim_cells": int(m["wui_sim_cells"]),
                    "wui_tp_cells": int(m["wui_tp_cells"]),
                    "total_ember_ignitions_12h": sim_out["total_ember_ignitions"],
                    "alpha_launch": e_params.alpha_launch if e_params else 0.0,
                    "beta_dist_m_per_ms": e_params.beta_dist_m_per_ms if e_params else 0.0,
                    "p_urban_wui": e_params.p_urban_wui if e_params else 0.0,
                })

    df_pf = pd.DataFrame(per_fire_rows)
    df_pf.to_csv("reports/step2_ember_ablation_per_fire.csv", index=False)

    # Aggregate summary by fold group
    groups = [
        ("Wind-Driven Folds (Kincade, Glass, Zogg)", wind_names),
        ("Plume-Fire Folds (Creek, Bobcat)", ["Creek", "Bobcat"]),
        ("Terrain Folds (Caldor, Tamarack)", ["Caldor", "Tamarack"]),
        ("All 7 Training Folds", all_names),
    ]
    agg_rows: list[dict[str, Any]] = []
    for g_name, f_list in groups:
        for mode_label in [
            "Ember OFF (Contact-Only, P_base=0.21)",
            "Ember ON (LOFO Fit)",
            "Ember ON (Wind-3 Consensus)",
        ]:
            for hr in (1, 3, 6, 12):
                sub = df_pf[
                    (df_pf["fire_name"].isin(f_list))
                    & (df_pf["ablation_mode"] == mode_label)
                    & (df_pf["horizon_h"] == hr)
                ]
                agg_rows.append({
                    "fold_group": f"{g_name} (n={len(f_list)})",
                    "ablation_mode": mode_label,
                    "split": str(sub["split"].iloc[0]),
                    "horizon_h": hr,
                    "mean_iou_2km": float(sub["iou_2km"].mean()),
                    "mean_f1_2km": float(sub["f1_2km"].mean()),
                    "mean_feds_375m_f1": float(sub["feds_375m_f1"].mean()) if hr == 12 else np.nan,
                    "mean_area_ratio": float(sub["area_ratio_sim_obs"].mean()),
                    "median_area_ratio": float(sub["area_ratio_sim_obs"].median()),
                    "mean_front_dist_err_km": float(sub["front_dist_err_km"].mean()),
                    "mean_front_signed_km": float(sub["front_dist_signed_km"].mean()),
                    "mean_far_field_jump_cells": float(sub["far_field_jump_cells_sim"].mean()),
                    "total_wui_tp_over_obs": f"{int(sub['wui_tp_cells'].sum())}/{int(sub['wui_obs_cells'].sum())}",
                })

    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv("reports/step2_ember_ablation_summary.csv", index=False)

    print(f"\nCompleted Step 2 Ember Calibration & Ablation in {time.perf_counter() - t0:.1f}s")
    print("\n=== STEP 2: Ember ON vs OFF Ablation Summary (+6h and +12h) ===")
    print(df_agg[df_agg["horizon_h"].isin([6, 12])].to_string(index=False))

    # Plot Ember ON vs OFF ablation figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    w_sub = df_agg[df_agg["fold_group"].str.startswith("Wind-Driven")]
    for mode_label, color, ls in [
        ("Ember OFF (Contact-Only, P_base=0.21)", "tab:gray", "--"),
        ("Ember ON (LOFO Fit)", "tab:red", "-"),
        ("Ember ON (Wind-3 Consensus)", "tab:orange", ":"),
    ]:
        m_df = w_sub[w_sub["ablation_mode"] == mode_label].sort_values("horizon_h")
        axes[0].plot(
            m_df["horizon_h"],
            m_df["mean_iou_2km"],
            marker="o",
            linestyle=ls,
            linewidth=2.2,
            color=color,
            label=mode_label,
        )
        axes[1].plot(
            m_df["horizon_h"],
            m_df["mean_front_dist_err_km"],
            marker="s",
            linestyle=ls,
            linewidth=2.2,
            color=color,
            label=mode_label,
        )

    axes[0].set_title("Wind-Driven Folds (n=3): ~2 km GOFER IoU (Ember ON vs OFF)", fontweight="bold")
    axes[0].set_xlabel("Horizon (Hours)")
    axes[0].set_ylabel("Mean ~2 km GOFER IoU")
    axes[0].set_xticks([1, 3, 6, 12])
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8.5)

    axes[1].set_title("Wind-Driven Folds (n=3): Wind-Axis Front Distance Error (km)", fontweight="bold")
    axes[1].set_xlabel("Horizon (Hours)")
    axes[1].set_ylabel("Mean Front Distance Error (km)")
    axes[1].set_xticks([1, 3, 6, 12])
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    out_png = Path("reports/phase_1b_ember_ablation.png")
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Saved ember ablation plot to {out_png}")


if __name__ == "__main__":
    main()
