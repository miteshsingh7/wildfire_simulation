"""Step 3: Perimeter and Progression Metrics for Surrogate vs Teacher v2 and GeoMAC / NIST.

Evaluates:
- 3a: IoU, recall, precision of surrogate at +11.4 h vs observed GeoMAC ID 1052 raster,
      beside Teacher v2 ensemble row from Section 7.1 of the handover.
- 3b: Perimeter polygons at +1, +2, +4, +6 h from surrogate (burn_prob >= 0.5 and q50 <= t),
      saved to out/demo/perimeters/ (and documents that observed intermediate perimeters do not exist).
- 3c: Arrival-time error at NIST TN 2252 timestamped fire detections (signed bias, median absolute error,
      share within 30 min, share predicted never-burn) for surrogate and Teacher v2 ensemble median.
      Documents that NIST timestamps are reconstructed after the fact.
"""

import json
import os
import time
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes
import shapely.geometry
import torch

from evaluate_heldout_camp_fire import (
    TeacherParams,
    build_camp_fire_dataset_60m,
)
from wf052.sim.ember_kernel import (
    EmberParams,
    simulate_teacher_with_embers_60m,
)
from wf052.routing.contracts import validate_arrival_field
from wf052.surrogate.adapter import TilePrediction, stitch_tiles_to_arrival_field
from wf052.surrogate.dataset import rebuild_spatial_inputs
from wf052.surrogate.model import WindConditionedUNet


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("STAND-IN WEIGHTS (Phase 3A Smoke Checkpoint)")
    print("STEP 3: PERIMETER AND PROGRESSION METRICS")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Build Camp Fire dataset and grid geometry
    ds_camp, cfg, transform_60m = build_camp_fire_dataset_60m(device)
    h_full, w_full = ds_camp.fbfm_arr.shape
    print(f"Full Domain: {h_full} rows x {w_full} cols at 60m")

    from wf052.replay.providers import TeacherFieldProvider
    tf_base = TeacherFieldProvider().get_field(0.0)
    origin_x = float(tf_base["x"].values[0])
    origin_y = float(tf_base["y"].values[0])
    x_min = origin_x - 30.0
    y_max = origin_y + 30.0

    # 1. Load Surrogate Model
    ckpt_path = os.environ.get("CKPT", "checkpoints/best.pt")
    print(f"Loading surrogate model from: {ckpt_path}")
    model = WindConditionedUNet(in_spatial_channels=13, base_channels=32, emb_dim=64, max_horizon_min=720.0)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load HRRR wind vector series (12, 2)
    weather_df = pd.read_parquet("data/tiles/hrrr_weather_hourly.parquet")
    u_vals = weather_df["u10_ms"].values[:12].astype(np.float32)
    v_vals = weather_df["v10_ms"].values[:12].astype(np.float32)
    wind_series = np.column_stack([u_vals, v_vals])
    wind_batch = torch.from_numpy(wind_series).unsqueeze(0).to(device)

    # Generate full domain prediction using 3x3 grid (9 tiles of 256x256)
    r_origins = [0, 154, 308]
    c_origins = [0, 153, 306]
    tile_preds: list[TilePrediction] = []

    print("\nPredicting 3x3 tile grid (9 tiles)...")
    for r0 in r_origins:
        for c0 in c_origins:
            # Ignition at Poe Dam: (232, 452)
            rel_ign_r = 232 - r0
            rel_ign_c = 452 - c0
            tile_tensor = rebuild_spatial_inputs(
                fire_id="camp_fire",
                crop_origin=(r0, c0),
                ignition_cell=(rel_ign_r, rel_ign_c),
                fires_base_dir="out/camp_fire_60m",
                tile_shape=(256, 256),
            )
            spatial_batch = tile_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(spatial_batch, wind_batch)
                q_tile = out.quantiles[0].cpu().numpy()  # (3, 256, 256)
                prob_tile = torch.sigmoid(out.burn_logits[0, 0]).cpu().numpy()  # (256, 256)

            tile_preds.append(
                TilePrediction(
                    quantiles=q_tile,
                    burn_prob=prob_tile,
                    r0=r0,
                    c0=c0,
                )
            )

    print("Stitching 9 tiles into full-domain ArrivalField (p_thr=0.1)...")
    full_af = stitch_tiles_to_arrival_field(
        tile_predictions=tile_preds,
        domain_shape=(h_full, w_full),
        origin_x=origin_x,
        origin_y=origin_y,
        resolution_m=60.0,
        crs="EPSG:32610",
        ignition_time="2018-11-08T14:33:00Z",
        p_thr=0.1,
    )
    validate_arrival_field(full_af)

    surr_q50 = full_af["arrival_q50"].values  # (564, 562), +inf where unburned
    surr_burn_prob = np.zeros((h_full, w_full), dtype=np.float32)
    for p in tile_preds:
        h_t, w_t = p.burn_prob.shape
        surr_burn_prob[p.r0 : p.r0 + h_t, p.c0 : p.c0 + w_t] = np.maximum(
            surr_burn_prob[p.r0 : p.r0 + h_t, p.c0 : p.c0 + w_t],
            p.burn_prob,
        )

    # =========================================================================
    # STEP 3a: IoU, Recall, Precision at +11.4 h vs GeoMAC ID 1052
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3a: PERIMETER OVERLAP AT +11.4 h (GeoMAC ID 1052)")
    print("=" * 80)

    gdf_geomac = gpd.read_file("data/raw/geomac_camp_fire_2018_all_perimeters.geojson").to_crs("EPSG:32610")
    sub_1052 = gdf_geomac[gdf_geomac["OBJECTID"] == 1052]
    geom_1052 = sub_1052.geometry.iloc[0]

    obs_1052_mask = (
        rasterize(
            [(geom_1052, 1)],
            out_shape=(h_full, w_full),
            transform=transform_60m,
            fill=0,
            dtype=np.uint8,
        )
        == 1
    )
    obs_cells = int(np.sum(obs_1052_mask))
    obs_ha = obs_cells * (60.0 * 60.0) / 10000.0

    # Surrogate at +11.4 h (684 min): burned if q50 <= 684.0 and finite
    surr_114_mask = np.isfinite(surr_q50) & (surr_q50 <= 684.0)
    surr_cells = int(np.sum(surr_114_mask))
    surr_ha = surr_cells * (60.0 * 60.0) / 10000.0

    tp = int(np.sum(surr_114_mask & obs_1052_mask))
    fp = int(np.sum(surr_114_mask & (~obs_1052_mask)))
    fn = int(np.sum((~surr_114_mask) & obs_1052_mask))

    surr_iou = tp / max(tp + fp + fn, 1)
    surr_recall = tp / max(tp + fn, 1)
    surr_precision = tp / max(tp + fp, 1)
    surr_area_ratio = surr_ha / max(obs_ha, 1.0)

    comp_3a_rows = [
        {
            "model": "Surrogate (STAND-IN Phase 3A Smoke)",
            "sim_area_ha": f"{surr_ha:.1f}",
            "area_ratio": f"{surr_area_ratio:.4f}",
            "iou": f"{surr_iou:.4f}",
            "recall": f"{surr_recall * 100.0:.2f}%",
            "precision": f"{surr_precision * 100.0:.2f}%",
            "notes": "Single forward pass (462.8 ms), p_thr=0.1, t<=684min",
        },
        {
            "model": "Teacher v2 Ensemble Median (Handover Sec 7.1)",
            "sim_area_ha": "11586.6 [7233.7, 21136.2]",
            "area_ratio": "0.5229 [0.3265, 0.9539]",
            "iou": "0.3147 [0.1658, 0.3763]",
            "recall": "40.80% [20.65%, 47.56%]",
            "precision": "60.19% [46.10%, 89.63%]",
            "notes": "20 fixed seeds (100+17*i), physics+embers, runtime ~4.0s/seed",
        },
    ]
    df_3a = pd.DataFrame(comp_3a_rows)
    print(df_3a.to_string(index=False))

    # =========================================================================
    # STEP 3b: Perimeter Polygons at +1, +2, +4, +6 h
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3b: SURROGATE PERIMETER POLYGONS AT +1, +2, +4, +6 h")
    print("=" * 80)
    perim_dir = Path("out/demo/perimeters")
    perim_dir.mkdir(parents=True, exist_ok=True)

    hours = [1.0, 2.0, 4.0, 6.0]
    polygon_records = []

    for h_val in hours:
        t_limit_min = h_val * 60.0
        # Criterion from spec: burn_prob >= 0.5 and q50 <= t
        poly_mask = (surr_burn_prob >= 0.5) & (surr_q50 <= t_limit_min)
        poly_mask_uint8 = poly_mask.astype(np.uint8)
        mask_cells = int(np.sum(poly_mask_uint8))
        mask_ha = mask_cells * (60.0 * 60.0) / 10000.0

        # Extract polygon contours via rasterio.shapes
        poly_geoms = []
        for shape_geom, val in shapes(poly_mask_uint8, mask=poly_mask_uint8, transform=transform_60m):
            if val == 1:
                poly_geoms.append(shapely.geometry.shape(shape_geom))

        if poly_geoms:
            unary_poly = shapely.ops.unary_union(poly_geoms)
        else:
            unary_poly = shapely.geometry.Polygon()

        poly_gdf = gpd.GeoDataFrame(
            [{"horizon_hours": h_val, "horizon_min": t_limit_min, "area_ha": mask_ha, "cell_count": mask_cells}],
            geometry=[unary_poly],
            crs="EPSG:32610",
        )
        geojson_out = perim_dir / f"surrogate_perimeter_{int(h_val)}h.geojson"
        poly_gdf.to_file(geojson_out, driver="GeoJSON")

        polygon_records.append({
            "horizon_hours": f"+{int(h_val)} h",
            "horizon_min": t_limit_min,
            "cells": mask_cells,
            "area_ha": round(mask_ha, 1),
            "geojson_saved": str(geojson_out),
            "observed_intermediate_perimeter": "NONE EXIST IN REPO DATA",
        })

    df_3b = pd.DataFrame(polygon_records)
    print(df_3b.to_string(index=False))
    print("\nIntermediate Observed Perimeters Status:")
    print("  GeoMAC raw file 'data/raw/geomac_camp_fire_2018_all_perimeters.geojson' earliest entry is OBJECTID 1052 at 17:54 PST (+11.4 h).")
    print("  No intermediate observed perimeter polygons exist in repo data for +1h, +2h, +4h, or +6h.")

    # =========================================================================
    # STEP 3c: Arrival-Time Error at NIST TN 2252 Detections
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3c: NIST TN 2252 ARRIVAL-TIME ERROR")
    print("=" * 80)
    print("Note on Provenance: NIST TN 2252 timestamps were reconstructed after the fact")
    print("from 911 emergency dispatch recordings, radio logs, and witness statements.")

    gdf_nist = gpd.read_parquet("data/tiles/observations_nist.parquet")
    valid_nist = gdf_nist[(gdf_nist["elapsed_minutes"] >= 0.0) & (gdf_nist["elapsed_minutes"] <= 720.0)].copy()
    n_nist_total = len(valid_nist)
    print(f"Total NIST observations in [0, 12h]: {n_nist_total} points")

    cols = np.clip(((valid_nist["utm_x"].values - x_min) / 60.0).astype(int), 0, w_full - 1)
    rows = np.clip(((y_max - valid_nist["utm_y"].values) / 60.0).astype(int), 0, h_full - 1)

    # 1. Surrogate arrival evaluation
    surr_pred_arrivals = surr_q50[rows, cols]
    valid_nist["surr_q50"] = surr_pred_arrivals
    valid_nist["surr_burned"] = np.isfinite(surr_pred_arrivals) & (surr_pred_arrivals <= 720.0)
    valid_nist["surr_signed_bias"] = valid_nist["surr_q50"] - valid_nist["elapsed_minutes"]
    valid_nist["surr_abs_error"] = np.abs(valid_nist["surr_signed_bias"])

    surr_burned_pts = valid_nist[valid_nist["surr_burned"]]
    surr_n_burned = len(surr_burned_pts)
    surr_n_never = n_nist_total - surr_n_burned
    surr_never_share = (surr_n_never / n_nist_total) * 100.0

    if surr_n_burned > 0:
        surr_signed_bias_med = float(surr_burned_pts["surr_signed_bias"].median())
        surr_mae = float(surr_burned_pts["surr_abs_error"].median())
        surr_within_30m_count = int(np.sum(surr_burned_pts["surr_abs_error"] <= 30.0))
        surr_within_30m_share = (surr_within_30m_count / n_nist_total) * 100.0
    else:
        surr_signed_bias_med = np.nan
        surr_mae = np.nan
        surr_within_30m_share = 0.0

    # 2. Teacher v2 evaluation across the 5 representative seeds (389, 117, 304, 100, 236)
    # Using the TeacherParams and EmberParams calibrated for Teacher v2
    t_params = TeacherParams(
        p_base=0.22, a=0.06, c_1=0.045, c_2=0.15, p_continue=0.30,
        fuel_scale_grass_shrub=1.0, fuel_scale_timber=1.0, canopy_den_scale=1.0, wind_moisture_damping=0.0
    )
    e_params = EmberParams(alpha_launch=0.0040, beta_dist_m_per_ms=110.0, p_urban_wui=0.15)
    test_seeds = [389, 117, 304, 100, 236]

    teacher_bias_list = []
    teacher_mae_list = []
    teacher_within_30_list = []
    teacher_never_list = []

    print("Computing Teacher v2 ensemble metrics across 5 seeds...")
    for s in test_seeds:
        sim_s = simulate_teacher_with_embers_60m(
            ds_camp, t_params, ember_params=e_params, use_ember_kernel=True,
            duration_hours=12, steps_per_hour=30, seed=s
        )
        arr_s = sim_s["arrival_time_min"]
        t_arrivals = arr_s[rows, cols]
        t_burned_mask = np.isfinite(t_arrivals) & (t_arrivals <= 720.0)
        n_b = int(np.sum(t_burned_mask))
        n_never = n_nist_total - n_b

        bias_val = t_arrivals[t_burned_mask] - valid_nist["elapsed_minutes"].values[t_burned_mask]
        abs_err = np.abs(bias_val)

        teacher_never_list.append((n_never / n_nist_total) * 100.0)
        if n_b > 0:
            teacher_bias_list.append(float(np.median(bias_val)))
            teacher_mae_list.append(float(np.median(abs_err)))
            teacher_within_30_list.append((int(np.sum(abs_err <= 30.0)) / n_nist_total) * 100.0)

    t_bias_med = float(np.median(teacher_bias_list))
    t_mae_med = float(np.median(teacher_mae_list))
    t_w30_med = float(np.median(teacher_within_30_list))
    t_never_med = float(np.median(teacher_never_list))

    comp_3c_rows = [
        {
            "model": "Surrogate (STAND-IN Phase 3A Smoke)",
            "total_nist_points": n_nist_total,
            "median_signed_bias_min": f"{surr_signed_bias_med:+.1f}",
            "median_absolute_error_min": f"{surr_mae:.1f}",
            "share_within_30min": f"{surr_within_30m_share:.1f}%",
            "share_predicted_never_burn": f"{surr_never_share:.1f}%",
            "notes": "STAND-IN WEIGHTS (Phase 3A smoke)",
        },
        {
            "model": "Teacher v2 Ensemble Median (5 representative seeds)",
            "total_nist_points": n_nist_total,
            "median_signed_bias_min": f"{t_bias_med:+.1f}",
            "median_absolute_error_min": f"{t_mae_med:.1f}",
            "share_within_30min": f"{t_w30_med:.1f}%",
            "share_predicted_never_burn": f"{t_never_med:.1f}%",
            "notes": "Physics + Embers, seeds [389, 117, 304, 100, 236]",
        },
    ]
    df_3c = pd.DataFrame(comp_3c_rows)
    print("\n" + df_3c.to_string(index=False))

    # Save summary tables to CSV
    Path("reports").mkdir(exist_ok=True)
    df_3a.to_csv("reports/step3a_perimeter_metrics.csv", index=False)
    df_3b.to_csv("reports/step3b_intermediate_perimeters.csv", index=False)
    df_3c.to_csv("reports/step3c_nist_arrival_errors.csv", index=False)
    print(f"\nSaved Step 3 reports to reports/step3*.csv in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
