"""Evaluate frozen configs/teacher_v1.yaml on Held-Out 2018 Camp Fire at 60m (Phase 1A Step 3).

Strictly verifies SHA-256 hash of configs/teacher_v1.yaml before touching Camp Fire data.
Computes:
1. GeoMAC IR perimeter overlap at +11.4 h (684 min, OBJECTID=1052) and +12.0 h (720 min),
   plus Final Contained footprint (OBJECTID=1398).
2. Burned area progression at +3h, +6h, +11.4h, +12h.
3. NIST TN 2135 Appendix F observation points split by type:
   - vegetative (primary contact-spread score)
   - spot_fire (reported, not tuned on)
   - structural (reported, not tuned on)
   - other & ALL
4. 20-seed ensemble P10-P90 uncertainty coverage on held-out Camp Fire.
5. Early-time town arrival checks (Concow, Paradise East Edge, Magalia).
6. Optional resolution penalty check: Kincade (2019) at 30 m vs 60 m.
"""

import hashlib
import time
from collections import Counter
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from wf052.sim.pytorchfire_engine import FBFM40_TO_P_VEG
from wf052.sim.teacher_calibrator import (
    FireDataset60m,
    TeacherParams,
    aggregate_to_coarse_fraction,
    load_fire_dataset_60m,
    simulate_teacher_60m,
)


def block_mode(arr: np.ndarray, factor: int) -> np.ndarray:
    """Downsample categorical 2D raster by block mode (dominant class)."""
    h, w = arr.shape
    new_h = h // factor
    new_w = w // factor
    cropped = arr[: new_h * factor, : new_w * factor]
    reshaped = (
        cropped.reshape(new_h, factor, new_w, factor)
        .transpose(0, 2, 1, 3)
        .reshape(new_h, new_w, factor * factor)
    )
    out = np.zeros((new_h, new_w), dtype=arr.dtype)
    for r in range(new_h):
        for c in range(new_w):
            out[r, c] = Counter(reshaped[r, c]).most_common(1)[0][0]
    return out


def block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Downsample continuous 2D raster by block mean."""
    h, w = arr.shape
    new_h = h // factor
    new_w = w // factor
    cropped = arr[: new_h * factor, : new_w * factor]
    reshaped = (
        cropped.reshape(new_h, factor, new_w, factor)
        .transpose(0, 2, 1, 3)
        .reshape(new_h, new_w, factor * factor)
    )
    return reshaped.mean(axis=2).astype(np.float32)


def verify_frozen_config(config_path: Path, hash_path: Path) -> dict[str, Any]:
    """Verify SHA-256 of configs/teacher_v1.yaml matches reports/teacher_v1_hash.txt."""
    raw_bytes = config_path.read_bytes()
    actual_hash = hashlib.sha256(raw_bytes).hexdigest()
    expected_hash = hash_path.read_text().strip().split()[0]
    if not expected_hash or actual_hash != expected_hash:
        raise RuntimeError(
            f"Integrity check failed! {config_path} SHA-256={actual_hash} != expected {expected_hash}"
        )
    print(f"Verified frozen config {config_path} | SHA-256: {actual_hash}")
    return yaml.safe_load(raw_bytes)


def build_camp_fire_dataset_60m(device: torch.device) -> tuple[FireDataset60m, dict[str, Any], Any]:
    """Construct 60m Camp Fire dataset from 30m tiles using physical block aggregation."""
    with open("configs/phase0.yaml") as f:
        cfg = yaml.safe_load(f)

    tiles_dir = Path("data/tiles")
    with rasterio.open(tiles_dir / "fbfm40.tif") as src:
        fbfm_30m = src.read(1)
    with rasterio.open(tiles_dir / "canopy_cover.tif") as src:
        cc_30m = src.read(1).astype(np.float32)
    with rasterio.open(tiles_dir / "elevation.tif") as src:
        elev_30m = src.read(1).astype(np.float32)

    factor = 2
    res_m = 60.0
    fbfm_60m = block_mode(fbfm_30m, factor)
    cc_60m = block_mean(cc_30m, factor)
    elev_60m = block_mean(elev_30m, factor)
    h, w = fbfm_60m.shape  # 564 x 562

    p_veg_arr = np.full((h, w), 0.20, dtype=np.float32)
    for code, p_val in FBFM40_TO_P_VEG.items():
        p_veg_arr[fbfm_60m == code] = p_val
    p_veg_arr[fbfm_60m <= 0] = -1.0

    is_burnable_np = p_veg_arr > -0.5
    is_gs_np = (fbfm_60m >= 101) & (fbfm_60m <= 147)
    is_timber_np = (fbfm_60m >= 161) & (fbfm_60m <= 204)
    p_den_arr = np.clip((cc_60m / 100.0) * 0.4 - 0.2, -0.2, 0.2).astype(np.float32)

    slope_arr = np.zeros((h, w, 3, 3), dtype=np.float32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            dist = np.sqrt(dr**2 + dc**2) * res_m
            elev_shifted = np.roll(np.roll(elev_60m, -dr, axis=0), -dc, axis=1)
            dz = elev_shifted - elev_60m
            slope_arr[:, :, dr + 1, dc + 1] = np.degrees(np.arctan(dz / dist))

    weather_df = pd.read_parquet(tiles_dir / "hrrr_weather_hourly.parquet")

    bbox_utm = cfg["domain"]["bbox_utm"]
    ign_x = float(cfg["event"]["ignition"]["utm_x"])
    ign_y = float(cfg["event"]["ignition"]["utm_y"])
    ign_col = int(np.clip((ign_x - bbox_utm["x_min"]) / res_m, 0, w - 1))
    ign_row = int(np.clip((bbox_utm["y_max"] - ign_y) / res_m, 0, h - 1))

    init_mask = np.zeros((13, h, w), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if 0 <= ign_row + dr < h and 0 <= ign_col + dc < w:
                init_mask[0, ign_row + dr, ign_col + dc] = True

    transform_60m = from_bounds(
        bbox_utm["x_min"],
        bbox_utm["y_min"],
        bbox_utm["x_max"],
        bbox_utm["y_max"],
        w,
        h,
    )

    ds = FireDataset60m(
        fire_name="CampFire_2018_HeldOut",
        regime="wind-driven",
        grid_size=h,
        res_m=res_m,
        fbfm_arr=fbfm_60m,
        p_veg_base=torch.from_numpy(p_veg_arr).to(device),
        is_grass_shrub=torch.from_numpy(is_gs_np).to(device),
        is_timber=torch.from_numpy(is_timber_np).to(device),
        is_burnable=torch.from_numpy(is_burnable_np).to(device),
        p_den_base=torch.from_numpy(p_den_arr).to(device),
        slope_deg=torch.from_numpy(slope_arr).to(device),
        weather_df=weather_df,
        gofer_hourly_masks=init_mask,
        gofer_arrival_min=np.full((h, w), np.nan, dtype=np.float32),
        gofer_areas_ha=np.zeros(13, dtype=np.float32),
        feds_12h_mask_375m=np.zeros((h // 6, w // 6), dtype=bool),
        feds_24h_mask_375m=np.zeros((h // 6, w // 6), dtype=bool),
        wind_axis_proj_km=np.zeros((h, w), dtype=np.float32),
    )
    return ds, cfg, transform_60m


def evaluate_geomac_perimeters_60m(
    model_name: str,
    split_label: str,
    arr_min: np.ndarray,
    transform_60m: Any,
    res_m: float = 60.0,
) -> list[dict[str, Any]]:
    """Evaluate 60m arrival raster against GeoMAC +11.4h (ID 1052) and Final Contained (ID 1398)."""
    h, w = arr_min.shape
    cell_ha = (res_m * res_m) / 10_000.0  # 0.36 ha

    geomac_path = Path("data/raw/geomac_camp_fire_2018_all_perimeters.geojson")
    gdf_geomac = gpd.read_file(geomac_path).to_crs("EPSG:32610")

    cases = [
        {
            "reference_perimeter": "+11.4 h GeoMAC IR Overflight (17:54 PST, ID 1052)",
            "objectid": 1052,
            "datetime_utc": "2018-11-09 01:54:00 UTC",
            "sim_horizon_min": 684.0,
        },
        {
            "reference_perimeter": "+11.4 h GeoMAC IR Overflight vs +12.0 h Sim",
            "objectid": 1052,
            "datetime_utc": "2018-11-09 01:54:00 UTC",
            "sim_horizon_min": 720.0,
        },
        {
            "reference_perimeter": "Final Contained Footprint (ID 1398)",
            "objectid": 1398,
            "datetime_utc": "2018-11-25 18:22:00 UTC",
            "sim_horizon_min": 720.0,
        },
    ]

    rows_out: list[dict[str, Any]] = []
    for c in cases:
        sub = gdf_geomac[gdf_geomac["OBJECTID"] == c["objectid"]]
        geom = sub.geometry.iloc[0]
        obs_mask = (
            rasterize(
                [(geom, 1)],
                out_shape=(h, w),
                transform=transform_60m,
                fill=0,
                dtype=np.uint8,
            )
            == 1
        )
        t_lim = float(c["sim_horizon_min"])
        sim_mask = (~np.isnan(arr_min)) & (arr_min <= t_lim)

        tp = int(np.count_nonzero(sim_mask & obs_mask))
        fp = int(np.count_nonzero(sim_mask & (~obs_mask)))
        fn = int(np.count_nonzero((~sim_mask) & obs_mask))
        tn = int(np.count_nonzero((~sim_mask) & (~obs_mask)))

        iou = tp / max(tp + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2.0 * prec * rec) / max(prec + rec, 1e-9)

        obs_area_ha = float(np.count_nonzero(obs_mask) * cell_ha)
        sim_area_ha = float(np.count_nonzero(sim_mask) * cell_ha)

        rows_out.append({
            "split": split_label,
            "model": model_name,
            "reference_perimeter": c["reference_perimeter"],
            "sim_horizon_min": t_lim,
            "obs_area_ha": obs_area_ha,
            "sim_area_ha": sim_area_ha,
            "area_ratio_sim_obs": sim_area_ha / max(obs_area_ha, 1.0),
            "TP_cells": tp,
            "FP_cells": fp,
            "FN_cells": fn,
            "TN_cells": tn,
            "IoU": iou,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
        })
    return rows_out


def evaluate_nist_split_60m(
    model_name: str,
    split_label: str,
    gdf_nist: gpd.GeoDataFrame,
    arr_min: np.ndarray,
    bbox_utm: dict[str, float],
    res_m: float = 60.0,
    max_horizon_minutes: float = 720.0,
) -> list[dict[str, Any]]:
    """Evaluate 60m arrival raster against NIST observations partitioned by fire type."""
    h, w = arr_min.shape
    valid = gdf_nist[
        (gdf_nist["elapsed_minutes"] >= 0.0) & (gdf_nist["elapsed_minutes"] <= max_horizon_minutes)
    ].copy()

    cols = np.clip(((valid["utm_x"].values - bbox_utm["x_min"]) / res_m).astype(int), 0, w - 1)
    rows = np.clip(((bbox_utm["y_max"] - valid["utm_y"].values) / res_m).astype(int), 0, h - 1)
    valid["sim_arrival_min"] = arr_min[rows, cols]
    valid["burned_by_sim"] = (~np.isnan(valid["sim_arrival_min"])) & (
        valid["sim_arrival_min"] <= max_horizon_minutes
    )
    valid["signed_bias"] = valid["sim_arrival_min"] - valid["elapsed_minutes"]
    valid["abs_error"] = np.abs(valid["signed_bias"])

    categories = [
        ("vegetative", "contact_spread_target"),
        ("spot_fire", "reported_not_tuned"),
        ("structural", "reported_not_tuned"),
        ("other", "reported_not_tuned"),
        ("ALL", "aggregate"),
    ]
    out_rows: list[dict[str, Any]] = []
    for cat, role in categories:
        sub = valid if cat == "ALL" else valid[valid["fire_type"] == cat]
        total_pts = len(sub)
        if total_pts == 0:
            continue
        burned = sub[sub["burned_by_sim"]]
        n_burned = len(burned)
        n_unburned = total_pts - n_burned
        unburned_pct = (n_unburned / total_pts) * 100.0

        if n_burned > 0:
            bias_mean = float(burned["signed_bias"].mean())
            bias_med = float(burned["signed_bias"].median())
            mae = float(burned["abs_error"].mean())
            med_ae = float(burned["abs_error"].median())
        else:
            bias_mean = np.nan
            bias_med = np.nan
            mae = np.nan
            med_ae = np.nan

        out_rows.append({
            "split": split_label,
            "model": model_name,
            "fire_type": cat,
            "evaluation_role": role,
            "total_points": total_pts,
            "burned_points": n_burned,
            "unburned_points": n_unburned,
            "never_burned_pct": unburned_pct,
            "signed_bias_mean_min": bias_mean,
            "signed_bias_median_min": bias_med,
            "abs_error_mae_min": mae,
            "abs_error_median_min": med_ae,
        })
    return out_rows


def run_resolution_penalty_check_kincade(
    device: torch.device,
    frozen_params: TeacherParams,
) -> pd.DataFrame:
    """Optional check: run Kincade (2019) at 30m (60 steps/hr) vs 60m (30 steps/hr)."""
    print("\n=== Running Optional Resolution Penalty Check on Kincade (30m vs 60m) ===")
    kincade_dir = Path("data/tiles/fires/kincade")
    ds_60m = load_fire_dataset_60m("Kincade", "wind-driven", kincade_dir, device=device)

    # 1. Run at 60m (720x720, 30 steps/hr)
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    res_60m = simulate_teacher_60m(ds_60m, frozen_params, duration_hours=24, steps_per_hour=30, seed=42)
    if device.type == "mps":
        torch.mps.synchronize()
    sec_60m = time.perf_counter() - t0

    # 2. Construct 30m Kincade dataset by 2x nearest upsampling of 60m grid (1440x1440, 60 steps/hr)
    fbfm_30m = np.repeat(np.repeat(ds_60m.fbfm_arr, 2, axis=0), 2, axis=1)
    p_veg_30m = ds_60m.p_veg_base.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
    is_gs_30m = ds_60m.is_grass_shrub.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
    is_tim_30m = ds_60m.is_timber.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
    is_burn_30m = ds_60m.is_burnable.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
    p_den_30m = ds_60m.p_den_base.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
    slope_30m = ds_60m.slope_deg.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
    gofer_masks_30m = np.repeat(np.repeat(ds_60m.gofer_hourly_masks, 2, axis=1), 2, axis=2)

    ds_30m = FireDataset60m(
        fire_name="Kincade_30m",
        regime="wind-driven",
        grid_size=1440,
        res_m=30.0,
        fbfm_arr=fbfm_30m,
        p_veg_base=p_veg_30m,
        is_grass_shrub=is_gs_30m,
        is_timber=is_tim_30m,
        is_burnable=is_burn_30m,
        p_den_base=p_den_30m,
        slope_deg=slope_30m,
        weather_df=ds_60m.weather_df,
        gofer_hourly_masks=gofer_masks_30m,
        gofer_arrival_min=np.repeat(np.repeat(ds_60m.gofer_arrival_min, 2, axis=0), 2, axis=1),
        gofer_areas_ha=ds_60m.gofer_areas_ha,
        feds_12h_mask_375m=ds_60m.feds_12h_mask_375m,
        feds_24h_mask_375m=ds_60m.feds_24h_mask_375m,
        wind_axis_proj_km=np.repeat(np.repeat(ds_60m.wind_axis_proj_km, 2, axis=0), 2, axis=1),
    )

    if device.type == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()
    res_30m = simulate_teacher_60m(ds_30m, frozen_params, duration_hours=24, steps_per_hour=60, seed=42)
    if device.type == "mps":
        torch.mps.synchronize()
    sec_30m = time.perf_counter() - t1

    rows = []
    for res_label, res_m, grid_str, steps_hr, sec, sim_out, obs_masks, block_2km in [
        ("60 m (Operating)", 60.0, "720x720", 30, sec_60m, res_60m, ds_60m.gofer_hourly_masks, 33),
        ("30 m (Fine Check)", 30.0, "1440x1440", 60, sec_30m, res_30m, gofer_masks_30m, 66),
    ]:
        cell_ha = (res_m * res_m) / 10_000.0
        for hr in (12, 24):
            s_mask = sim_out["hourly_sim_masks"][hr]
            o_mask = obs_masks[hr]
            sim_ha = float(np.count_nonzero(s_mask) * cell_ha)
            obs_ha = float(np.count_nonzero(o_mask) * cell_ha)
            s_frac = aggregate_to_coarse_fraction(s_mask, block_2km) >= 0.25
            o_frac = aggregate_to_coarse_fraction(o_mask, block_2km) >= 0.25
            tp = float(np.count_nonzero(s_frac & o_frac))
            fp = float(np.count_nonzero(s_frac & (~o_frac)))
            fn = float(np.count_nonzero((~s_frac) & o_frac))
            iou = tp / max(tp + fp + fn, 1.0)
            f1 = (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0)
            rows.append({
                "fire_name": "Kincade (2019)",
                "resolution": res_label,
                "grid": grid_str,
                "steps_per_hour": steps_hr,
                "horizon_h": hr,
                "wall_clock_24h_sec": sec,
                "sim_area_ha": sim_ha,
                "obs_area_ha": obs_ha,
                "area_ratio": sim_ha / max(obs_ha, 1.0),
                "gofer_2km_iou": iou,
                "gofer_2km_f1": f1,
            })
    return pd.DataFrame(rows)


def main() -> None:
    cfg_teacher = verify_frozen_config(
        Path("configs/teacher_v1.yaml"),
        Path("reports/teacher_v1_hash.txt"),
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Held-Out 2018 Camp Fire 60m evaluation on {device}...")

    ds_camp, cfg_p0, transform_60m = build_camp_fire_dataset_60m(device)
    bbox_utm = cfg_p0["domain"]["bbox_utm"]
    res_m = 60.0
    cell_ha = (res_m * res_m) / 10_000.0  # 0.36 ha

    models_to_eval = {
        "teacher_v1 (Frozen All-7)": TeacherParams(**cfg_teacher["parameters"]),
        "wind_only_v1 (Frozen Wind-3)": TeacherParams(**cfg_teacher["ablation_wind_only_parameters"]),
        "uncalibrated_default (p_base=0.50)": TeacherParams(**cfg_teacher["uncalibrated_default_parameters"]),
        "phase0_baseline (p_base=0.58)": TeacherParams(p_base=0.58),
    }

    gdf_nist = gpd.read_parquet("data/tiles/observations_nist.parquet")

    perim_rows: list[dict[str, Any]] = []
    area_rows: list[dict[str, Any]] = []
    nist_rows: list[dict[str, Any]] = []
    arrival_rasters: dict[str, np.ndarray] = {}

    for model_name, params in models_to_eval.items():
        sim_out = simulate_teacher_60m(
            ds_camp,
            params,
            duration_hours=12,
            steps_per_hour=int(cfg_teacher["steps_per_hour"]),
            seed=42,
        )
        arr_min = sim_out["arrival_time_min"]
        arrival_rasters[model_name] = arr_min

        # Save canonical frozen teacher_v1 arrival raster
        if model_name.startswith("teacher_v1"):
            out_tif = Path("data/tiles/arrival_time_teacher_v1_60m_seed42.tif")
            h, w = arr_min.shape
            with rasterio.open(
                out_tif,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=1,
                dtype=np.float32,
                crs="EPSG:32610",
                transform=transform_60m,
                nodata=np.nan,
            ) as dst:
                dst.write(arr_min, 1)
            print(f"Saved frozen teacher 60m arrival raster to {out_tif}")

        perim_rows.extend(
            evaluate_geomac_perimeters_60m(
                model_name=model_name,
                split_label="held-out (Camp Fire 2018)",
                arr_min=arr_min,
                transform_60m=transform_60m,
                res_m=res_m,
            )
        )

        nist_rows.extend(
            evaluate_nist_split_60m(
                model_name=model_name,
                split_label="held-out (Camp Fire 2018)",
                gdf_nist=gdf_nist,
                arr_min=arr_min,
                bbox_utm=bbox_utm,
                res_m=res_m,
            )
        )

        # Area progression at +3h, +6h, +11.4h, +12h
        for label_h, t_min, obs_ha, obs_src in [
            ("+3.0 h", 180.0, np.nan, "No observation"),
            ("+6.0 h", 360.0, np.nan, "No observation"),
            ("+11.4 h", 684.0, 22156.2, "GeoMAC IR Flight (ID 1052)"),
            ("+12.0 h", 720.0, 22156.2, "GeoMAC IR Flight (+11.4h ref)"),
        ]:
            b_cells = int(np.count_nonzero((~np.isnan(arr_min)) & (arr_min <= t_min)))
            b_ha = b_cells * cell_ha
            area_rows.append({
                "split": "held-out (Camp Fire 2018)",
                "model": model_name,
                "horizon": label_h,
                "horizon_min": t_min,
                "sim_burned_cells_60m": b_cells,
                "sim_burned_area_ha": b_ha,
                "obs_area_in_domain_ha": obs_ha,
                "obs_source": obs_src,
                "sim_obs_ratio": b_ha / obs_ha if not np.isnan(obs_ha) else np.nan,
            })

    df_perim = pd.DataFrame(perim_rows)
    df_perim.to_csv("reports/camp_fire_heldout_60m_perimeters.csv", index=False)
    print("\n=== HELD-OUT CAMP FIRE (60m): GeoMAC Perimeter Overlap ===")
    print(df_perim.to_string(index=False))

    df_area = pd.DataFrame(area_rows)
    df_area.to_csv("reports/camp_fire_heldout_60m_areas.csv", index=False)
    print("\n=== HELD-OUT CAMP FIRE (60m): Area Progression ===")
    print(df_area.to_string(index=False))

    df_nist = pd.DataFrame(nist_rows)
    df_nist.to_csv("reports/camp_fire_heldout_60m_nist_split.csv", index=False)
    print("\n=== HELD-OUT CAMP FIRE (60m): NIST Observation Split by Type ===")
    print(df_nist.to_string(index=False))

    # 20-Seed Ensemble on Held-Out Camp Fire for teacher_v1
    seeds_20 = [100 + 17 * i for i in range(20)]
    ens_arrivals = []
    ens_11_4h_areas = []
    ens_11_4h_ious = []

    geomac_path = Path("data/raw/geomac_camp_fire_2018_all_perimeters.geojson")
    gdf_geomac = gpd.read_file(geomac_path).to_crs("EPSG:32610")
    geom_1052 = gdf_geomac[gdf_geomac["OBJECTID"] == 1052].geometry.iloc[0]
    obs_1052_mask = (
        rasterize(
            [(geom_1052, 1)],
            out_shape=(ds_camp.p_veg_base.shape[0], ds_camp.p_veg_base.shape[1]),
            transform=transform_60m,
            fill=0,
            dtype=np.uint8,
        )
        == 1
    )

    teacher_v1_params = TeacherParams(**cfg_teacher["parameters"])
    for s in seeds_20:
        s_out = simulate_teacher_60m(ds_camp, teacher_v1_params, duration_hours=12, steps_per_hour=30, seed=s)
        a_min = s_out["arrival_time_min"]
        ens_arrivals.append(a_min)
        m_684 = (~np.isnan(a_min)) & (a_min <= 684.0)
        ens_11_4h_areas.append(float(np.count_nonzero(m_684) * cell_ha))
        tp = float(np.count_nonzero(m_684 & obs_1052_mask))
        fp = float(np.count_nonzero(m_684 & (~obs_1052_mask)))
        fn = float(np.count_nonzero((~m_684) & obs_1052_mask))
        ens_11_4h_ious.append(tp / max(tp + fp + fn, 1.0))

    stack_arr = np.stack(ens_arrivals, axis=0)  # [20, H, W]
    stack_filled = np.where(np.isnan(stack_arr), 720.0, stack_arr)
    p10_map = np.percentile(stack_filled, 10, axis=0)
    p50_map = np.percentile(stack_filled, 50, axis=0)
    p90_map = np.percentile(stack_filled, 90, axis=0)

    # Check NIST point P10-P90 coverage and town arrival times
    valid_nist = gdf_nist[
        (gdf_nist["elapsed_minutes"] >= 0.0) & (gdf_nist["elapsed_minutes"] <= 720.0)
    ].copy()
    h, w = p50_map.shape
    cols = np.clip(((valid_nist["utm_x"].values - bbox_utm["x_min"]) / res_m).astype(int), 0, w - 1)
    rows = np.clip(((bbox_utm["y_max"] - valid_nist["utm_y"].values) / res_m).astype(int), 0, h - 1)
    valid_nist["p10"] = p10_map[rows, cols]
    valid_nist["p50"] = p50_map[rows, cols]
    valid_nist["p90"] = p90_map[rows, cols]
    valid_nist["in_p10_p90"] = (valid_nist["elapsed_minutes"] >= valid_nist["p10"] - 30.0) & (
        valid_nist["elapsed_minutes"] <= valid_nist["p90"] + 30.0
    )

    veg_nist = valid_nist[valid_nist["fire_type"] == "vegetative"]
    ens_summary_df = pd.DataFrame([{
        "split": "held-out (Camp Fire 2018)",
        "model": "teacher_v1 (20-Seed Ensemble)",
        "area_11_4h_mean_ha": float(np.mean(ens_11_4h_areas)),
        "area_11_4h_std_ha": float(np.std(ens_11_4h_areas)),
        "area_11_4h_p10_ha": float(np.percentile(ens_11_4h_areas, 10)),
        "area_11_4h_p90_ha": float(np.percentile(ens_11_4h_areas, 90)),
        "iou_11_4h_mean": float(np.mean(ens_11_4h_ious)),
        "iou_11_4h_std": float(np.std(ens_11_4h_ious)),
        "nist_veg_p10_p90_coverage_pct": float(veg_nist["in_p10_p90"].mean() * 100.0),
        "nist_all_p10_p90_coverage_pct": float(valid_nist["in_p10_p90"].mean() * 100.0),
    }])
    ens_summary_df.to_csv("reports/camp_fire_heldout_60m_ensemble20.csv", index=False)
    print("\n=== HELD-OUT CAMP FIRE (60m): 20-Seed Ensemble Summary ===")
    print(ens_summary_df.to_string(index=False))

    # Town arrival check at 60m
    towns = [
        {"landmark": "Concow (Community Center)", "utm_x": 626550.0, "utm_y": 4401950.0, "obs_first_min": 60.0, "obs_main_min": 120.0},
        {"landmark": "Paradise East Edge (Pentz Rd)", "utm_x": 622250.0, "utm_y": 4402850.0, "obs_first_min": 76.0, "obs_main_min": 150.0},
        {"landmark": "Magalia (Center / Coutolenc)", "utm_x": 619950.0, "utm_y": 4407550.0, "obs_first_min": 173.0, "obs_main_min": 300.0},
    ]
    town_rows = []
    for t in towns:
        tc = int(np.clip((t["utm_x"] - bbox_utm["x_min"]) / res_m, 0, w - 1))
        tr = int(np.clip((bbox_utm["y_max"] - t["utm_y"]) / res_m, 0, h - 1))
        # Take min in 3x3 neighborhood around landmark
        r_sl = slice(max(0, tr - 1), min(h, tr + 2))
        c_sl = slice(max(0, tc - 1), min(w, tc + 2))
        t_v1 = float(np.nanmin(arrival_rasters["teacher_v1 (Frozen All-7)"][r_sl, c_sl]))
        t_w1 = float(np.nanmin(arrival_rasters["wind_only_v1 (Frozen Wind-3)"][r_sl, c_sl]))
        t_unc = float(np.nanmin(arrival_rasters["uncalibrated_default (p_base=0.50)"][r_sl, c_sl]))
        town_rows.append({
            "split": "held-out (Camp Fire 2018)",
            "landmark": t["landmark"],
            "obs_first_ember_min": t["obs_first_min"],
            "obs_main_front_min": t["obs_main_min"],
            "teacher_v1_arrival_min": t_v1,
            "teacher_v1_p10_min": float(np.min(p10_map[r_sl, c_sl])),
            "teacher_v1_p90_min": float(np.min(p90_map[r_sl, c_sl])),
            "wind_only_v1_arrival_min": t_w1,
            "uncalibrated_arrival_min": t_unc,
        })
    df_towns = pd.DataFrame(town_rows)
    df_towns.to_csv("reports/camp_fire_heldout_60m_towns.csv", index=False)
    print("\n=== HELD-OUT CAMP FIRE (60m): Landmark Arrival Check ===")
    print(df_towns.to_string(index=False))

    # Optional Resolution Penalty Check: Kincade 30m vs 60m
    df_res_pen = run_resolution_penalty_check_kincade(device, teacher_v1_params)
    df_res_pen.to_csv("reports/resolution_penalty_30m_vs_60m.csv", index=False)
    print("\n=== OPTIONAL RESOLUTION PENALTY CHECK (Kincade 30m vs 60m) ===")
    print(df_res_pen.to_string(index=False))


if __name__ == "__main__":
    main()
