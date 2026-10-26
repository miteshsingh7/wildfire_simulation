"""Evaluate Teacher v2 on Camp Fire across 20 seeds (100, 117, 134, ..., 100 + 17*i).

Outputs reports/teacher_v2_camp_fire_20seeds.csv and reports median and P10-P90 across seeds for:
- Simulated area (ha) at +11.4 h (684 min)
- Area ratio vs observed (22,157.64 ha, 61,549 cells)
- IoU
- Recall
- Precision (TP / Sim)
"""

import sys
import time
from pathlib import Path

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


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Teacher v2 20-seed Monte Carlo on {device}...")

    ds_camp, bbox, transform = build_camp_fire_dataset_60m(device)
    h, w = ds_camp.fbfm_arr.shape
    cell_ha = 0.36

    # Load GeoMAC ID 1052 ground-truth perimeter at +11.4 h (17:54 PST)
    gdf = gpd.read_file("data/raw/geomac_camp_fire_2018_all_perimeters.geojson").to_crs("EPSG:32610")
    sub = gdf[gdf["OBJECTID"] == 1052]
    geom = sub.geometry.iloc[0]
    obs_mask = rasterize([(geom, 1)], out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8) == 1
    obs_cells = int(np.count_nonzero(obs_mask))
    obs_ha = float(obs_cells * cell_ha)
    print(f"Observed GeoMAC ID 1052 (+11.4h): {obs_cells} cells ({obs_ha:.2f} ha)")

    # Teacher v2 parameters (configs/teacher_v2.yaml)
    t_params = TeacherParams(
        p_base=0.22,
        a=0.06,
        c_1=0.045,
        c_2=0.15,
        p_continue=0.30,
        fuel_scale_grass_shrub=1.0,
        fuel_scale_timber=1.0,
        canopy_den_scale=1.0,
        wind_moisture_damping=0.0,
    )
    e_params = EmberParams(
        alpha_launch=0.0040,
        beta_dist_m_per_ms=110.0,
        p_urban_wui=0.15,
    )

    seeds = [100 + 17 * i for i in range(20)]
    print(f"Seeds: {seeds}")

    records = []
    t_start_all = time.time()

    for idx, seed in enumerate(seeds):
        t0 = time.time()
        sim_out = simulate_teacher_with_embers_60m(
            ds_camp,
            t_params,
            ember_params=e_params,
            use_ember_kernel=True,
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
            "fire_name": "Camp Fire (2018)",
            "role": "DEVELOPMENT",
            "seed_index": idx,
            "seed": seed,
            "horizon_min": 684.0,
            "horizon_h": 11.4,
            "obs_cells": obs_cells,
            "obs_area_ha": obs_ha,
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
        print(
            f"Seed {seed:3d} ({idx+1:2d}/20): Sim {sim_ha:8.1f} ha | AreaRatio {area_ratio:.4f} | "
            f"IoU {iou:.4f} | Recall {rec*100:5.2f}% | Precision {prec*100:5.2f}% ({elapsed:.2f}s)"
        )

    df = pd.DataFrame(records)
    out_csv = Path("reports/teacher_v2_camp_fire_20seeds.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved 20-seed results to {out_csv} in {time.time() - t_start_all:.1f}s")

    print("\n=== SUMMARY STATISTICS ACROSS 20 SEEDS (Camp Fire @ +11.4h) ===")
    for col in ["sim_area_ha", "area_ratio", "iou", "recall", "precision"]:
        vals = df[col].values
        p10 = np.percentile(vals, 10)
        p50 = np.percentile(vals, 50)
        p90 = np.percentile(vals, 90)
        mean = np.mean(vals)
        std = np.std(vals)
        if col in ["recall", "precision"]:
            print(
                f"{col:15s}: Median {p50*100:5.2f}% | P10-P90 [{p10*100:5.2f}%, {p90*100:5.2f}%] | "
                f"Mean {mean*100:5.2f}% ± {std*100:5.2f}%"
            )
        elif col == "sim_area_ha":
            print(
                f"{col:15s}: Median {p50:8.1f} ha | P10-P90 [{p10:8.1f}, {p90:8.1f}] ha | "
                f"Mean {mean:8.1f} ± {std:8.1f} ha"
            )
        else:
            print(
                f"{col:15s}: Median {p50:8.4f} | P10-P90 [{p10:8.4f}, {p90:8.4f}] | "
                f"Mean {mean:8.4f} ± {std:8.4f}"
            )


if __name__ == "__main__":
    main()
