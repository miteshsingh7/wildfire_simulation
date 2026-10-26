"""Step 1: Teacher Attribution on Camp Fire across 3 arms (20 seeds each).

Arms:
(a) v1: p_base 0.21, embers OFF
(b) p_base 0.22, embers OFF (override use_ember_kernel=False)
(c) v2: p_base 0.22, embers ON

For each arm reports:
- Median and P10-P90 of area ha, area ratio, IoU, recall, precision at +11.4 h (684 min).
- Quantifies gain attribution: p_base alone (b vs a) and embers (c vs b).
- Compares seed 42 (used in replay providers A, A2, B-on-A2) against arm (c) distribution.
"""

import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from rasterio.features import rasterize

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from evaluate_heldout_camp_fire import build_camp_fire_dataset_60m
from wf052.sim.ember_kernel import EmberParams, simulate_teacher_with_embers_60m
from wf052.sim.teacher_calibrator import TeacherParams


def run_arm(
    ds_camp: Any,
    t_params: TeacherParams,
    e_params: EmberParams | None,
    use_embers: bool,
    seeds: list[int],
    obs_mask: np.ndarray,
    obs_cells: int,
    obs_ha: float,
    cell_ha: float = 0.36,
    arm_name: str = "arm",
) -> pd.DataFrame:
    records = []
    for idx, seed in enumerate(seeds):
        t0 = time.time()
        sim_out = simulate_teacher_with_embers_60m(
            ds_camp,
            t_params,
            ember_params=e_params,
            use_ember_kernel=use_embers,
            duration_hours=12,
            steps_per_hour=30,
            seed=seed,
        )
        elapsed = time.time() - t0
        arr = sim_out["arrival_time_min"]

        sim_mask = np.isfinite(arr) & (arr <= 684.0)
        tp = int(np.count_nonzero(sim_mask & obs_mask))
        fp = int(np.count_nonzero(sim_mask & (~obs_mask)))
        fn = int(np.count_nonzero((~sim_mask) & obs_mask))
        sim_cells = tp + fp
        sim_ha = float(sim_cells * cell_ha)
        area_ratio = sim_ha / obs_ha
        iou = tp / max(tp + fp + fn, 1)
        prec = tp / max(sim_cells, 1)
        rec = tp / max(obs_cells, 1)

        records.append({
            "arm": arm_name,
            "seed_index": idx,
            "seed": seed,
            "sim_cells": sim_cells,
            "sim_area_ha": sim_ha,
            "area_ratio": area_ratio,
            "tp_cells": tp,
            "fp_cells": fp,
            "fn_cells": fn,
            "iou": iou,
            "recall": rec,
            "precision": prec,
            "runtime_sec": elapsed,
        })
        print(f"[{arm_name}] Seed {seed:3d} ({idx+1:2d}/20): Sim {sim_ha:8.1f} ha | AreaRatio {area_ratio:.4f} | IoU {iou:.4f} | Rec {rec*100:5.2f}% | Prec {prec*100:5.2f}% ({elapsed:.2f}s)")
    return pd.DataFrame(records)


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Teacher Attribution on {device}...")

    ds_camp, bbox, transform = build_camp_fire_dataset_60m(device)
    h, w = ds_camp.fbfm_arr.shape
    cell_ha = 0.36

    gdf = gpd.read_file("data/raw/geomac_camp_fire_2018_all_perimeters.geojson").to_crs("EPSG:32610")
    sub = gdf[gdf["OBJECTID"] == 1052]
    geom = sub.geometry.iloc[0]
    obs_mask = rasterize([(geom, 1)], out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8) == 1
    obs_cells = int(np.count_nonzero(obs_mask))
    obs_ha = float(obs_cells * cell_ha)

    seeds = [100 + 17 * i for i in range(20)]

    # Arm (a): v1: p_base 0.21, embers OFF
    t_params_a = TeacherParams(
        p_base=0.21, a=0.06, c_1=0.045, c_2=0.15, p_continue=0.30,
        fuel_scale_grass_shrub=1.0, fuel_scale_timber=1.0, canopy_den_scale=1.0, wind_moisture_damping=0.0
    )
    print("\n--- Running Arm (a): v1 (p_base=0.21, embers OFF) ---")
    df_a = run_arm(ds_camp, t_params_a, None, False, seeds, obs_mask, obs_cells, obs_ha, cell_ha, "arm_a_v1_p21_noembers")

    # Arm (b): p_base 0.22, embers OFF
    t_params_b = TeacherParams(
        p_base=0.22, a=0.06, c_1=0.045, c_2=0.15, p_continue=0.30,
        fuel_scale_grass_shrub=1.0, fuel_scale_timber=1.0, canopy_den_scale=1.0, wind_moisture_damping=0.0
    )
    print("\n--- Running Arm (b): p_base=0.22, embers OFF ---")
    df_b = run_arm(ds_camp, t_params_b, None, False, seeds, obs_mask, obs_cells, obs_ha, cell_ha, "arm_b_p22_noembers")

    # Arm (c): v2: p_base 0.22, embers ON
    # Load from reports/teacher_v2_camp_fire_20seeds.csv if exists, else run
    p_c = Path("reports/teacher_v2_camp_fire_20seeds.csv")
    if p_c.exists():
        print(f"\n--- Loading Arm (c): v2 (p_base=0.22, embers ON) from {p_c} ---")
        df_c = pd.read_csv(p_c)
        df_c["arm"] = "arm_c_v2_p22_embers"
    else:
        e_params = EmberParams(alpha_launch=0.0040, beta_dist_m_per_ms=110.0, p_urban_wui=0.15)
        print("\n--- Running Arm (c): v2 (p_base=0.22, embers ON) ---")
        df_c = run_arm(ds_camp, t_params_b, e_params, True, seeds, obs_mask, obs_cells, obs_ha, cell_ha, "arm_c_v2_p22_embers")

    # Combine
    df_all = pd.concat([df_a, df_b, df_c], ignore_index=True)
    df_all.to_csv("reports/teacher_attribution_3arms_20seeds.csv", index=False)

    print("\n" + "=" * 80)
    print("TEACHER ATTRIBUTION 3-ARM COMPARISON SUMMARY (Camp Fire @ +11.4h)")
    print("=" * 80)

    summary_rows = []
    for arm_label, df_arm in [
        ("Arm (a): v1 (p_base=0.21, Embers OFF)", df_a),
        ("Arm (b): p_base=0.22, Embers OFF", df_b),
        ("Arm (c): v2 (p_base=0.22, Embers ON)", df_c),
    ]:
        row = {"arm": arm_label}
        for col in ["sim_area_ha", "area_ratio", "iou", "recall", "precision"]:
            vals = df_arm[col].values
            p10 = np.percentile(vals, 10)
            p50 = np.percentile(vals, 50)
            p90 = np.percentile(vals, 90)
            row[f"{col}_median"] = p50
            row[f"{col}_p10"] = p10
            row[f"{col}_p90"] = p90
        summary_rows.append(row)

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv("reports/teacher_attribution_summary.csv", index=False)

    for r in summary_rows:
        print(f"\n{r['arm']}:")
        print(f"  Area (ha) : Median {r['sim_area_ha_median']:8.1f} ha | P10-P90 [{r['sim_area_ha_p10']:8.1f}, {r['sim_area_ha_p90']:8.1f}] ha")
        print(f"  Area Ratio: Median {r['area_ratio_median']:8.4f}    | P10-P90 [{r['area_ratio_p10']:8.4f}, {r['area_ratio_p90']:8.4f}]")
        print(f"  IoU       : Median {r['iou_median']:8.4f}    | P10-P90 [{r['iou_p10']:8.4f}, {r['iou_p90']:8.4f}]")
        print(f"  Recall    : Median {r['recall_median']*100:6.2f}%   | P10-P90 [{r['recall_p10']*100:6.2f}%, {r['recall_p90']*100:6.2f}%]")
        print(f"  Precision : Median {r['precision_median']*100:6.2f}%   | P10-P90 [{r['precision_p10']*100:6.2f}%, {r['precision_p90']*100:6.2f}%]")

    # Attribution decomposition
    med_area_a = df_a["sim_area_ha"].median()
    med_area_b = df_b["sim_area_ha"].median()
    med_area_c = df_c["sim_area_ha"].median()
    total_area_gain = med_area_c - med_area_a
    p_base_area_gain = med_area_b - med_area_a
    ember_area_gain = med_area_c - med_area_b

    med_iou_a = df_a["iou"].median()
    med_iou_b = df_b["iou"].median()
    med_iou_c = df_c["iou"].median()
    total_iou_gain = med_iou_c - med_iou_a
    p_base_iou_gain = med_iou_b - med_iou_a
    ember_iou_gain = med_iou_c - med_iou_b

    med_rec_a = df_a["recall"].median()
    med_rec_b = df_b["recall"].median()
    med_rec_c = df_c["recall"].median()
    total_rec_gain = med_rec_c - med_rec_a
    p_base_rec_gain = med_rec_b - med_rec_a
    ember_rec_gain = med_rec_c - med_rec_b

    print("\n" + "=" * 80)
    print("GAIN ATTRIBUTION: p_base ALONE (b vs a) vs EMBERS (c vs b)")
    print("=" * 80)
    print(f"Area Gain (Median): Total {total_area_gain:+8.1f} ha")
    print(f"  - from p_base (0.21 -> 0.22): {p_base_area_gain:+8.1f} ha ({p_base_area_gain/max(total_area_gain, 1e-6)*100:5.1f}%)")
    print(f"  - from Embers (OFF -> ON)  : {ember_area_gain:+8.1f} ha ({ember_area_gain/max(total_area_gain, 1e-6)*100:5.1f}%)")

    print(f"\nIoU Gain (Median): Total {total_iou_gain:+.4f}")
    print(f"  - from p_base (0.21 -> 0.22): {p_base_iou_gain:+.4f} ({p_base_iou_gain/max(total_iou_gain, 1e-6)*100:5.1f}%)")
    print(f"  - from Embers (OFF -> ON)  : {ember_iou_gain:+.4f} ({ember_iou_gain/max(total_iou_gain, 1e-6)*100:5.1f}%)")

    print(f"\nRecall Gain (Median): Total {total_rec_gain*100:+5.2f}%")
    print(f"  - from p_base (0.21 -> 0.22): {p_base_rec_gain*100:+5.2f}% ({p_base_rec_gain/max(total_rec_gain, 1e-6)*100:5.1f}%)")
    print(f"  - from Embers (OFF -> ON)  : {ember_rec_gain*100:+5.2f}% ({ember_rec_gain/max(total_rec_gain, 1e-6)*100:5.1f}%)")

    # Seed 42 distribution check
    import rasterio
    with rasterio.open("data/tiles/arrival_time_teacher_v2_60m_seed42.tif") as src:
        arr_seed42 = src.read(1)
    sim_cells_42 = int(np.count_nonzero(np.isfinite(arr_seed42) & (arr_seed42 <= 684.0)))
    sim_ha_42 = sim_cells_42 * cell_ha
    all_areas_c = df_c["sim_area_ha"].values
    pct_seed42 = float(np.mean(all_areas_c < sim_ha_42) * 100.0)
    rank_seed42 = int(np.sum(all_areas_c < sim_ha_42)) + 1

    print("\n" + "=" * 80)
    print("FROZEN SEED 42 IN ARM (c) DISTRIBUTION")
    print("=" * 80)
    print(f"Replay provider frozen raster (arrival_time_teacher_v2_60m_seed42.tif):")
    print(f"  - Simulated Area: {sim_ha_42:.1f} ha ({sim_cells_42} cells)")
    print(f"  - Rank: {rank_seed42} of 21 (including seed 42)")
    print(f"  - Percentile in Arm (c) 20 seeds: {pct_seed42:.1f}th percentile")


if __name__ == "__main__":
    main()
