"""Phase 1B-2 Part C: Baselines for Assimilation & Camp Fire Ember ON vs OFF (Point-Start + Assimilation).

Implements:
1. Fits the constant front speed v_const [m/min] on the 3 wind-driven training fires
   (Kincade, Glass, Zogg) across active 3h rolling windows (and logs all-7 speed for reference).
2. Evaluates on Camp Fire (2018) [split = development] at T0 in {+1 h, +2 h, +3 h} forecasting
   the strictly-future 3-hour window (T0, T0 + 3 h] ONLY (drops all cumulative [0, T0 + 3 h] metrics).
3. Compares:
   - Baseline: Nearest Assimilated Point + Constant Front Speed (fit on training fires)
   - Contact-Only Teacher (Ember OFF, Rolling Fit) — Point-Start & Assimilation Mode
   - Joint Teacher v2 (Ember ON, Joint Rolling Fit) — Point-Start & Assimilation Mode
4. Also evaluates Camp Fire 12 h Point-Start mode with Ember OFF vs Ember ON across
   NIST observation types (vegetative, spot_fire, structural, other, ALL) and GeoMAC perimeters.
"""

from pathlib import Path
import sys
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import torch
import yaml

sys.path.insert(0, "scripts")
from evaluate_heldout_camp_fire import build_camp_fire_dataset_60m, evaluate_camp_fire_12h
from run_phase1b_step1_assimilation import build_nist_observed_extent_mask

from wf052.sim.ember_kernel import EmberParams, compute_far_field_jump_cells, simulate_teacher_with_embers_60m
from wf052.sim.rolling_protocol import fit_constant_front_speed_on_training_fires
from wf052.sim.teacher_calibrator import (
    FireDataset60m,
    TeacherParams,
    load_fire_dataset_60m,
)


def evaluate_strictly_future_window(
    split_label: str,
    model_name: str,
    mode_label: str,
    t0_hour: int,
    gdf_nist: gpd.GeoDataFrame,
    arr_min: np.ndarray,
    bbox_utm: dict[str, float],
    res_m: float = 60.0,
) -> list[dict[str, Any]]:
    """Evaluate strictly-future NIST points in (T0, T0 + 3h] ONLY (no cumulative rows)."""
    h, w = arr_min.shape
    t0_min = float(t0_hour * 60.0)
    t_end_min = float((t0_hour + 3) * 60.0)

    valid = gdf_nist[
        (gdf_nist["elapsed_minutes"] > t0_min) & (gdf_nist["elapsed_minutes"] <= t_end_min)
    ].copy()
    win_str = f"({t0_hour}h, +{t0_hour+3}h] ({int(t0_min)}..{int(t_end_min)} min)"

    cols = np.clip(((valid["utm_x"].values - bbox_utm["x_min"]) / res_m).astype(int), 0, w - 1)
    rows = np.clip(((bbox_utm["y_max"] - valid["utm_y"].values) / res_m).astype(int), 0, h - 1)
    valid["sim_arrival_min"] = arr_min[rows, cols]
    valid["reached"] = (~np.isnan(valid["sim_arrival_min"])) & (valid["sim_arrival_min"] <= t_end_min)
    valid["signed_bias"] = valid["sim_arrival_min"] - valid["elapsed_minutes"]
    valid["abs_error"] = np.abs(valid["signed_bias"])

    out_rows: list[dict[str, Any]] = []
    for cat in ["vegetative", "spot_fire", "structural", "other", "ALL"]:
        sub = valid if cat == "ALL" else valid[valid["fire_type"] == cat]
        n_tot = len(sub)
        if n_tot == 0:
            continue
        reached_df = sub[sub["reached"]]
        n_reach = len(reached_df)
        n_unreach = n_tot - n_reach
        rec = n_reach / n_tot

        if n_reach > 0:
            b_mean = float(reached_df["signed_bias"].mean())
            b_med = float(reached_df["signed_bias"].median())
            mae = float(reached_df["abs_error"].mean())
            med_ae = float(reached_df["abs_error"].median())
        else:
            b_mean = float("nan")
            b_med = float("nan")
            mae = float("nan")
            med_ae = float("nan")

        out_rows.append(
            {
                "split": split_label,
                "model": model_name,
                "mode": mode_label,
                "T0_init": f"+{t0_hour} h ({int(t0_min)} min)",
                "strictly_future_window": win_str,
                "fire_type": cat,
                "total_points": n_tot,
                "reached_points": n_reach,
                "unreached_points": n_unreach,
                "fraction_reached_with_denom": f"{n_reach}/{n_tot} ({100.0 * rec:.1f}%)",
                "reached_pct": 100.0 * rec,
                "signed_bias_mean_min": b_mean,
                "signed_bias_median_min": b_med,
                "mae_min": mae,
                "median_ae_min": med_ae,
            }
        )
    return out_rows


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Running Phase 1B-2 Part C (Assimilation Baselines & Camp Fire Ember ON/OFF) on {device}...")

    # 1. Load frozen teacher_v2.yaml
    cfg_v2_path = Path("configs/teacher_v2.yaml")
    if cfg_v2_path.exists():
        with open(cfg_v2_path, encoding="utf-8") as f:
            cfg_v2 = yaml.safe_load(f)
        c_key = "contact_parameters" if "contact_parameters" in cfg_v2 else "teacher_parameters"
        p_base_on = float(cfg_v2[c_key]["p_base"])
        e_cfg = cfg_v2["ember_parameters"]
        ember_on = EmberParams(
            alpha_launch=float(e_cfg["alpha_launch"]),
            beta_dist_m_per_ms=float(e_cfg["beta_dist_m_per_ms"]),
            p_urban_wui=float(e_cfg["p_urban_wui"]),
        )
        p_base_off = float(cfg_v2.get("rolling_ember_off_baseline", {}).get("p_base", p_base_on))
    else:
        p_base_on = 0.20
        p_base_off = 0.21
        ember_on = EmberParams(alpha_launch=0.0020, beta_dist_m_per_ms=110.0, p_urban_wui=0.16)

    t_on = TeacherParams(p_base=p_base_on)
    t_off = TeacherParams(p_base=p_base_off)

    # 2. Load training fires and fit constant front speed v_const [m/min]
    setup_df = pd.read_csv("reports/phase_1a_fires_setup.csv")
    wind_names = {"Kincade", "Glass", "Zogg"}
    wind_train_ds: list[FireDataset60m] = []
    all_train_ds: list[FireDataset60m] = []
    for _, row in setup_df.iterrows():
        fname = str(row["fire_name"])
        regime = str(row["regime"])
        t_dir = Path("data/tiles/fires") / fname.lower()
        ds = load_fire_dataset_60m(fname, regime, t_dir, device=device)
        all_train_ds.append(ds)
        if fname in wind_names:
            wind_train_ds.append(ds)

    v_const_wind_m_min = fit_constant_front_speed_on_training_fires(wind_train_ds, horizon_h=3)
    v_const_all7_m_min = fit_constant_front_speed_on_training_fires(all_train_ds, horizon_h=3)
    print(
        f"Fitted constant front speed on training fires (3h rolling windows): "
        f"wind-driven={v_const_wind_m_min:.2f} m/min ({v_const_wind_m_min * 0.06:.2f} km/h), "
        f"all-7={v_const_all7_m_min:.2f} m/min ({v_const_all7_m_min * 0.06:.2f} km/h)"
    )

    # 3. Load Camp Fire (development)
    ds_camp, cfg_p0, transform_60m = build_camp_fire_dataset_60m(device)
    bbox_utm = cfg_p0["domain"]["bbox_utm"]
    gdf_nist = gpd.read_parquet("data/tiles/observations_nist.parquet")

    # 4. Run Point-Start (12h) for Ember OFF and Ember ON on Camp Fire
    sim_pt_off = simulate_teacher_with_embers_60m(
        ds=ds_camp,
        params=t_off,
        ember_params=EmberParams(),
        use_ember_kernel=False,
        duration_hours=12,
        steps_per_hour=30,
        seed=42,
    )
    sim_pt_on = simulate_teacher_with_embers_60m(
        ds=ds_camp,
        params=t_on,
        ember_params=ember_on,
        use_ember_kernel=True,
        duration_hours=12,
        steps_per_hour=30,
        seed=42,
    )

    rows_partc: list[dict[str, Any]] = []
    cell_ha = 0.36

    for t0_hr in (1, 2, 3):
        t0_min = float(t0_hr * 60.0)
        init_mask, n_init_pts = build_nist_observed_extent_mask(
            gdf_nist, t0_min, bbox_utm, transform_60m, ds_camp.fbfm_arr, res_m=60.0
        )
        init_ha = float(np.count_nonzero(init_mask) * cell_ha)

        # (a) Nearest assimilated point + constant front speed baseline (fit on wind-driven training fires)
        dist_m = distance_transform_edt(~init_mask) * 60.0
        arr_const_wind = np.where(init_mask, t0_min, t0_min + dist_m / max(v_const_wind_m_min, 1e-4))
        rows_partc.extend(
            evaluate_strictly_future_window(
                split_label="development (Camp Fire 2018; speed fit on Kincade/Glass/Zogg)",
                model_name=f"Baseline: Nearest Assimilated Point + Constant Speed ({v_const_wind_m_min:.2f} m/min)",
                mode_label=f"Assimilation @ +{t0_hr}h (init={n_init_pts} pts / {init_ha:.0f} ha)",
                t0_hour=t0_hr,
                gdf_nist=gdf_nist,
                arr_min=arr_const_wind,
                bbox_utm=bbox_utm,
            )
        )

        # (b) Ember OFF — Assimilation Mode [T0, T0+3h]
        sim_assim_off = simulate_teacher_with_embers_60m(
            ds=ds_camp,
            params=t_off,
            ember_params=EmberParams(),
            use_ember_kernel=False,
            duration_hours=3,
            steps_per_hour=30,
            seed=42,
            init_burned_mask=init_mask,
            start_hour_index=t0_hr,
            start_elapsed_min=t0_min,
        )
        arr_assim_off = sim_assim_off["arrival_time_min"]
        end_ha_off = float(np.count_nonzero(~np.isnan(arr_assim_off)) * cell_ha)
        rows_partc.extend(
            evaluate_strictly_future_window(
                split_label="development (Camp Fire 2018)",
                model_name=f"Teacher Ember OFF (P_base={p_base_off:.2f})",
                mode_label=f"Assimilation @ +{t0_hr}h (init={init_ha:.0f} ha -> {end_ha_off:.0f} ha)",
                t0_hour=t0_hr,
                gdf_nist=gdf_nist,
                arr_min=arr_assim_off,
                bbox_utm=bbox_utm,
            )
        )

        # (c) Joint Teacher v2 (Ember ON) — Assimilation Mode [T0, T0+3h]
        sim_assim_on = simulate_teacher_with_embers_60m(
            ds=ds_camp,
            params=t_on,
            ember_params=ember_on,
            use_ember_kernel=True,
            duration_hours=3,
            steps_per_hour=30,
            seed=42,
            init_burned_mask=init_mask,
            start_hour_index=t0_hr,
            start_elapsed_min=t0_min,
        )
        arr_assim_on = sim_assim_on["arrival_time_min"]
        end_ha_on = float(np.count_nonzero(~np.isnan(arr_assim_on)) * cell_ha)
        rows_partc.extend(
            evaluate_strictly_future_window(
                split_label="development (Camp Fire 2018)",
                model_name=(
                    f"Teacher v2 Ember ON (P_base={p_base_on:.2f}, alpha={ember_on.alpha_launch:.4f}, "
                    f"beta={ember_on.beta_dist_m_per_ms:.0f}, p_wui={ember_on.p_urban_wui:.2f})"
                ),
                mode_label=f"Assimilation @ +{t0_hr}h (init={init_ha:.0f} ha -> {end_ha_on:.0f} ha)",
                t0_hour=t0_hr,
                gdf_nist=gdf_nist,
                arr_min=arr_assim_on,
                bbox_utm=bbox_utm,
            )
        )

        # (d) Ember OFF — Point-Start Mode evaluated on the strictly-future window (T0, T0+3h]
        rows_partc.extend(
            evaluate_strictly_future_window(
                split_label="development (Camp Fire 2018)",
                model_name=f"Teacher Ember OFF (P_base={p_base_off:.2f})",
                mode_label="Point-Start from t=0",
                t0_hour=t0_hr,
                gdf_nist=gdf_nist,
                arr_min=sim_pt_off["arrival_time_min"],
                bbox_utm=bbox_utm,
            )
        )

        # (e) Joint Teacher v2 (Ember ON) — Point-Start Mode evaluated on strictly-future window (T0, T0+3h]
        rows_partc.extend(
            evaluate_strictly_future_window(
                split_label="development (Camp Fire 2018)",
                model_name=(
                    f"Teacher v2 Ember ON (P_base={p_base_on:.2f}, alpha={ember_on.alpha_launch:.4f}, "
                    f"beta={ember_on.beta_dist_m_per_ms:.0f}, p_wui={ember_on.p_urban_wui:.2f})"
                ),
                mode_label="Point-Start from t=0",
                t0_hour=t0_hr,
                gdf_nist=gdf_nist,
                arr_min=sim_pt_on["arrival_time_min"],
                bbox_utm=bbox_utm,
            )
        )

    df_partc = pd.DataFrame(rows_partc)
    df_partc.to_csv("reports/phase1b2_partC_assimilation_vs_baseline.csv", index=False)

    # 5. Also evaluate 12h Point-Start on Camp Fire (Ember OFF vs Ember ON) across all NIST points & perimeters
    res_12h_off = evaluate_camp_fire_12h(ds_camp, t_off, cfg_p0, transform_60m)
    # Evaluate Ember ON 12h point-start using the exact same GeoMAC + NIST protocol
    arr_on_12 = sim_pt_on["arrival_time_min"]
    h_grid, w_grid = arr_on_12.shape
    valid_nist_12 = gdf_nist[(gdf_nist["elapsed_minutes"] >= 0.0) & (gdf_nist["elapsed_minutes"] <= 720.0)].copy()
    cols_12 = np.clip(((valid_nist_12["utm_x"].values - bbox_utm["x_min"]) / 60.0).astype(int), 0, w_grid - 1)
    rows_12 = np.clip(((bbox_utm["y_max"] - valid_nist_12["utm_y"].values) / 60.0).astype(int), 0, h_grid - 1)
    valid_nist_12["sim_arrival_min"] = arr_on_12[rows_12, cols_12]
    valid_nist_12["reached"] = ~np.isnan(valid_nist_12["sim_arrival_min"])
    valid_nist_12["signed_bias"] = valid_nist_12["sim_arrival_min"] - valid_nist_12["elapsed_minutes"]
    valid_nist_12["abs_error"] = np.abs(valid_nist_12["signed_bias"])

    pt12_rows: list[dict[str, Any]] = []
    for model_label, sim_out in [
        (f"Teacher Ember OFF (P_base={p_base_off:.2f})", sim_pt_off),
        (
            f"Teacher v2 Ember ON (P_base={p_base_on:.2f}, alpha={ember_on.alpha_launch:.4f}, "
            f"beta={ember_on.beta_dist_m_per_ms:.0f}, p_wui={ember_on.p_urban_wui:.2f})",
            sim_pt_on,
        ),
    ]:
        arr_m = sim_out["arrival_time_min"]
        v_df = valid_nist_12.copy()
        v_df["sim_arrival_min"] = arr_m[rows_12, cols_12]
        v_df["reached"] = ~np.isnan(v_df["sim_arrival_min"])
        v_df["signed_bias"] = v_df["sim_arrival_min"] - v_df["elapsed_minutes"]
        v_df["abs_error"] = np.abs(v_df["signed_bias"])
        far_jumps_6h = compute_far_field_jump_cells(sim_out["hourly_sim_masks"][6], min_jump_cells=5)
        far_jumps_12h = compute_far_field_jump_cells(sim_out["hourly_sim_masks"][12], min_jump_cells=5)

        for cat in ["vegetative", "spot_fire", "structural", "other", "ALL"]:
            sub = v_df if cat == "ALL" else v_df[v_df["fire_type"] == cat]
            n_tot = len(sub)
            r_sub = sub[sub["reached"]]
            n_r = len(r_sub)
            rec = n_r / max(n_tot, 1)
            pt12_rows.append(
                {
                    "split": "development (Camp Fire 2018, 12h point-start)",
                    "model": model_label,
                    "fire_type": cat,
                    "total_points": n_tot,
                    "reached_points": n_r,
                    "unreached_points": n_tot - n_r,
                    "fraction_reached_with_denom": f"{n_r}/{n_tot} ({100.0 * rec:.1f}%)",
                    "reached_pct": 100.0 * rec,
                    "signed_bias_mean_min": float(r_sub["signed_bias"].mean()) if n_r > 0 else float("nan"),
                    "mae_min": float(r_sub["abs_error"].mean()) if n_r > 0 else float("nan"),
                    "median_ae_min": float(r_sub["abs_error"].median()) if n_r > 0 else float("nan"),
                    "sim_area_6h_ha": float(np.count_nonzero(sim_out["hourly_sim_masks"][6]) * 0.36),
                    "sim_area_12h_ha": float(np.count_nonzero(sim_out["hourly_sim_masks"][12]) * 0.36),
                    "far_jump_cells_6h": far_jumps_6h,
                    "far_jump_cells_12h": far_jumps_12h,
                    "total_ember_ignitions": sim_out["total_ember_ignitions"],
                }
            )
    _ = res_12h_off
    df_pt12 = pd.DataFrame(pt12_rows)
    df_pt12.to_csv("reports/phase1b2_partC_camp_pointstart_ember_on_vs_off.csv", index=False)

    # 6. Plot strictly-future Assimilation vs Constant-Speed Baseline vs Point-Start
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sub_all = df_partc[df_partc["fire_type"] == "ALL"].copy()

    t0_labels = ["+1 h (60 min)", "+2 h (120 min)", "+3 h (180 min)"]
    x = np.arange(len(t0_labels))
    width = 0.18

    series_specs = [
        ("Baseline: Nearest Assimilated", "Assimilation", "Constant-Speed Baseline (Assim)", "#7f7f7f"),
        ("Teacher Ember OFF", "Assimilation", "Teacher Ember OFF (Assim)", "#1f77b4"),
        ("Teacher v2 Ember ON", "Assimilation", "Teacher v2 Ember ON (Assim)", "#d62728"),
        ("Teacher Ember OFF", "Point-Start", "Teacher Ember OFF (Point-Start)", "#aec7e8"),
        ("Teacher v2 Ember ON", "Point-Start", "Teacher v2 Ember ON (Point-Start)", "#ff9896"),
    ]

    for idx_s, (m_pref, mode_pref, leg_label, color) in enumerate(series_specs):
        rec_vals = []
        mae_vals = []
        for t_lbl in t0_labels:
            row_m = sub_all[
                (sub_all["T0_init"] == t_lbl)
                & (sub_all["model"].str.startswith(m_pref))
                & (sub_all["mode"].str.startswith(mode_pref))
            ]
            if len(row_m) > 0:
                rec_vals.append(float(row_m["reached_pct"].iloc[0]))
                mae_vals.append(float(row_m["mae_min"].iloc[0]))
            else:
                rec_vals.append(0.0)
                mae_vals.append(0.0)
        offset = (idx_s - 2.0) * width
        axes[0].bar(x + offset, rec_vals, width, label=leg_label, color=color, edgecolor="black", linewidth=0.5)
        axes[1].bar(x + offset, mae_vals, width, label=leg_label, color=color, edgecolor="black", linewidth=0.5)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["T0=+1h -> (1h, 4h]\n(N=806 pts)", "T0=+2h -> (2h, 5h]\n(N=776 pts)", "T0=+3h -> (3h, 6h]\n(N=593 pts)"])
    axes[0].set_ylabel("Strictly-Future NIST Points Reached (%)")
    axes[0].set_title("A. Strictly-Future Recall (%) on Camp Fire [Development]")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["T0=+1h -> (1h, 4h]\n(N=806 pts)", "T0=+2h -> (2h, 5h]\n(N=776 pts)", "T0=+3h -> (3h, 6h]\n(N=593 pts)"])
    axes[1].set_ylabel("Arrival-Time MAE on Reached Points (min)")
    axes[1].set_title("B. Strictly-Future Arrival MAE (min) [Lower is Better]")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/phase_1b2_partC_assimilation_vs_baseline.png", dpi=160)
    plt.close(fig)

    print("\n=== PART C: Strictly-Future Next 3h Window (T0, T0+3h] Summary (ALL NIST Types) ===")
    print(
        sub_all[
            [
                "split",
                "T0_init",
                "strictly_future_window",
                "model",
                "mode",
                "fraction_reached_with_denom",
                "mae_min",
                "median_ae_min",
                "signed_bias_mean_min",
            ]
        ].to_string(index=False)
    )
    print("\n=== PART C: Camp Fire 12h Point-Start (Ember OFF vs Ember ON) ===")
    print(
        df_pt12[
            [
                "split",
                "model",
                "fire_type",
                "fraction_reached_with_denom",
                "mae_min",
                "median_ae_min",
                "sim_area_6h_ha",
                "sim_area_12h_ha",
                "far_jump_cells_6h",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
