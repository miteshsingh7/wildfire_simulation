"""Phase 1B Step 1: Operational Assimilation Test Mode (init-from-observed-extent) on Camp Fire.

For T0 in {+1 h (60 min), +2 h (120 min), +3 h (180 min)}:
- Builds the observed burned extent at T0 from NIST observations with elapsed_minutes <= T0.
- Runs a 3-hour forward forecast [T0, T0 + 3 h] on the 60m grid.
- Evaluates arrival-time error (MAE, median AE, signed bias) and Recall (with exact denominators
  N_reached / N_total) on NIST points by fire type (vegetative, spot_fire, structural, other, ALL),
  both on the strictly future forecast window (T0, T0 + 3 h] and cumulative [0, T0 + 3 h].
"""

import sys
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")
from evaluate_heldout_camp_fire import build_camp_fire_dataset_60m

from wf052.sim.ember_kernel import EmberParams, simulate_teacher_with_embers_60m
from wf052.sim.teacher_calibrator import FireDataset60m, TeacherParams


def build_nist_observed_extent_mask(
    gdf_nist: gpd.GeoDataFrame,
    t0_min: float,
    bbox_utm: dict[str, float],
    transform_60m: Any,
    fbfm_60m: np.ndarray,
    res_m: float = 60.0,
) -> tuple[np.ndarray, int]:
    """Construct 60m observed burned mask at T0 from NIST points with elapsed_minutes <= T0.

    Uses local 3x3 cell (180m x 180m) patches around every NIST point observed at <= T0
    plus 1-cell morphological closing to connect adjacent front observations without
    pre-filling unburned areas between spot fires.
    """
    from scipy.ndimage import binary_closing

    _ = transform_60m
    h, w = fbfm_60m.shape
    sub_t0 = gdf_nist[(gdf_nist["elapsed_minutes"] >= 0.0) & (gdf_nist["elapsed_minutes"] <= t0_min)].copy()
    n_obs_t0 = len(sub_t0)

    mask = np.zeros((h, w), dtype=bool)

    # Always include Pulga point of origin (633391, 4407338)
    ign_col = int(np.clip((633391.0 - bbox_utm["x_min"]) / res_m, 0, w - 1))
    ign_row = int(np.clip((bbox_utm["y_max"] - 4407338.0) / res_m, 0, h - 1))
    mask[max(0, ign_row - 1) : min(h, ign_row + 2), max(0, ign_col - 1) : min(w, ign_col + 2)] = True

    if n_obs_t0 == 0:
        return mask, 0

    xs = sub_t0["utm_x"].values
    ys = sub_t0["utm_y"].values
    cols = np.clip(((xs - bbox_utm["x_min"]) / res_m).astype(int), 0, w - 1)
    rows = np.clip(((bbox_utm["y_max"] - ys) / res_m).astype(int), 0, h - 1)
    for r_i, c_i in zip(rows, cols, strict=True):
        r_sl = slice(max(0, r_i - 1), min(h, r_i + 2))
        c_sl = slice(max(0, c_i - 1), min(w, c_i + 2))
        mask[r_sl, c_sl] = True

    # Connect immediately adjacent patches (1-cell gap = 60m) and exclude open water (code 98)
    mask = binary_closing(mask, iterations=1) & (fbfm_60m != 98)
    return mask, n_obs_t0


def evaluate_assimilation_window(
    split_label: str,
    model_name: str,
    t0_hour: int,
    window_mode: str,
    gdf_nist: gpd.GeoDataFrame,
    arr_min: np.ndarray,
    bbox_utm: dict[str, float],
    res_m: float = 60.0,
) -> list[dict[str, Any]]:
    """Evaluate arrival-time error and recall by NIST type for a 3-hour forecast from T0."""
    h, w = arr_min.shape
    t0_min = float(t0_hour * 60.0)
    t_end_min = float((t0_hour + 3) * 60.0)

    if window_mode == "forecast_next_3h":
        valid = gdf_nist[
            (gdf_nist["elapsed_minutes"] > t0_min) & (gdf_nist["elapsed_minutes"] <= t_end_min)
        ].copy()
        win_str = f"({t0_hour}h, +{t0_hour+3}h] ({int(t0_min)}..{int(t_end_min)} min)"
    else:
        valid = gdf_nist[
            (gdf_nist["elapsed_minutes"] >= 0.0) & (gdf_nist["elapsed_minutes"] <= t_end_min)
        ].copy()
        win_str = f"[0h, +{t0_hour+3}h] (0..{int(t_end_min)} min)"

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
            b_mean = np.nan
            b_med = np.nan
            mae = np.nan
            med_ae = np.nan

        out_rows.append({
            "split": split_label,
            "model": model_name,
            "T0_init": f"+{t0_hour} h ({int(t0_min)} min)",
            "eval_window": win_str,
            "window_mode": window_mode,
            "fire_type": cat,
            "total_points": n_tot,
            "reached_points": n_reach,
            "unreached_points": n_unreach,
            "recall_with_denom": f"{n_reach}/{n_tot} ({100.0*rec:.1f}%)",
            "recall_pct": 100.0 * rec,
            "signed_bias_mean_min": b_mean,
            "signed_bias_median_min": b_med,
            "abs_error_mae_min": mae,
            "abs_error_median_min": med_ae,
        })
    return out_rows


def run_step1_assimilation(
    ds_camp: FireDataset60m,
    bbox_utm: dict[str, float],
    transform_60m: Any,
    models: dict[str, tuple[TeacherParams, EmberParams | None, bool]],
) -> pd.DataFrame:
    """Run assimilation mode at T0 in {+1h, +2h, +3h} and point-start baseline."""
    gdf_nist = gpd.read_parquet("data/tiles/observations_nist.parquet")
    all_rows: list[dict[str, Any]] = []
    cell_ha = 0.36

    for model_name, (t_params, e_params, use_embers) in models.items():
        # Also run point-start 12h so we can compare point-start vs assimilation on the exact same windows
        sim_pt = simulate_teacher_with_embers_60m(
            ds_camp,
            t_params,
            ember_params=e_params,
            use_ember_kernel=use_embers,
            duration_hours=6,
            steps_per_hour=30,
            seed=42,
        )
        arr_pt = sim_pt["arrival_time_min"]

        for t0_hr in (1, 2, 3):
            t0_min = float(t0_hr * 60.0)
            init_mask, n_init_pts = build_nist_observed_extent_mask(
                gdf_nist, t0_min, bbox_utm, transform_60m, ds_camp.fbfm_arr, res_m=60.0
            )
            init_ha = float(np.count_nonzero(init_mask) * cell_ha)

            sim_assim = simulate_teacher_with_embers_60m(
                ds_camp,
                t_params,
                ember_params=e_params,
                use_ember_kernel=use_embers,
                duration_hours=3,
                steps_per_hour=30,
                seed=42,
                init_burned_mask=init_mask,
                start_hour_index=t0_hr,
                start_elapsed_min=t0_min,
            )
            arr_assim = sim_assim["arrival_time_min"]
            end_ha = float(np.count_nonzero(~np.isnan(arr_assim)) * cell_ha)

            for w_mode in ("forecast_next_3h", "cumulative_to_T0_plus_3h"):
                rows_a = evaluate_assimilation_window(
                    split_label="development (Camp Fire 2018, assimilation mode)",
                    model_name=f"{model_name} [Assimilation @ +{t0_hr}h, init={n_init_pts} pts / {init_ha:.0f} ha -> {end_ha:.0f} ha]",
                    t0_hour=t0_hr,
                    window_mode=w_mode,
                    gdf_nist=gdf_nist,
                    arr_min=arr_assim,
                    bbox_utm=bbox_utm,
                )
                all_rows.extend(rows_a)

                rows_p = evaluate_assimilation_window(
                    split_label="development (Camp Fire 2018, point-start baseline)",
                    model_name=f"{model_name} [Point-Start from t=0]",
                    t0_hour=t0_hr,
                    window_mode=w_mode,
                    gdf_nist=gdf_nist,
                    arr_min=arr_pt,
                    bbox_utm=bbox_utm,
                )
                all_rows.extend(rows_p)

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv("reports/step1_assimilation_camp_fire.csv", index=False)
    return df_out


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ds_camp, cfg_p0, transform_60m = build_camp_fire_dataset_60m(device)
    bbox_utm = cfg_p0["domain"]["bbox_utm"]

    models = {
        "teacher_v1 (P_base=0.21, Contact-Only)": (TeacherParams(p_base=0.21), None, False),
        "teacher_v1_1 (P_base=0.22, Contact-Only)": (TeacherParams(p_base=0.22), None, False),
    }
    df_assim = run_step1_assimilation(ds_camp, bbox_utm, transform_60m, models)

    sub_next3 = df_assim[df_assim["window_mode"] == "forecast_next_3h"]
    print("=== STEP 1: Assimilation Forecast (Strictly Future Next 3h Window (T0, T0+3h]) ===")
    print(
        sub_next3[
            [
                "split",
                "model",
                "T0_init",
                "fire_type",
                "recall_with_denom",
                "signed_bias_mean_min",
                "abs_error_mae_min",
                "abs_error_median_min",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
