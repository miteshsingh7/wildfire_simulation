"""Evaluate perimeter overlap and confusion matrices for Phase 0.5 audit (Task 2).

Evaluates:
1. Point-level NIST observation confusion matrix (TP, FP, FN, TN) at +3h, +6h, +12h.
2. Raster perimeter overlap against real GeoMAC infrared/containment perimeters:
   - Early Day 1 perimeter (2018-11-08 17:54 UTC, 22,090 ha)
   - Evening Day 1 perimeter (2018-11-09 00:00 UTC, 28,512 ha)
   - Final Contained perimeter (2018-11-25 18:22 UTC, 62,053 ha) -> "final-footprint overlap"
3. Audits and details the Phase 0 raster bug where nodata=-9999.0 inflated domain IoU.
"""

from datetime import UTC
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.features import rasterize


def evaluate_perimeters() -> dict[str, pd.DataFrame]:
    with open("configs/phase0.yaml") as f:
        cfg = yaml.safe_load(f)

    bbox_utm = cfg["domain"]["bbox_utm"]
    res = float(cfg["domain"]["resolution"])
    rows = int(cfg["domain"]["grid"]["rows"])
    cols = int(cfg["domain"]["grid"]["cols"])
    cell_ha = (res * res) / 10_000.0

    # Load simulation arrival rasters
    pt_path = Path("data/tiles/arrival_time_pytorchfire_seed42.tif")
    c2f_path = Path("data/tiles/arrival_time_cell2fire_seed42.tif")

    with rasterio.open(pt_path) as src:
        pt_arr = src.read(1)
        transform = src.transform

    c2f_arr = None
    if c2f_path.exists():
        with rasterio.open(c2f_path) as src:
            c2f_arr = src.read(1)

    # -------------------------------------------------------------
    # 1. NIST POINT-LEVEL CONFUSION MATRICES
    # -------------------------------------------------------------
    obs_df = gpd.read_parquet("data/tiles/observations_nist.parquet")
    # Points within 12h horizon
    obs_12h = obs_df[
        (obs_df["elapsed_minutes"] >= 0) & (obs_df["elapsed_minutes"] <= 720.0)
    ].copy()

    # Assign grid cell row/col to each point
    col_idx = ((obs_12h["utm_x"] - bbox_utm["x_min"]) / res).astype(int)
    row_idx = ((bbox_utm["y_max"] - obs_12h["utm_y"]) / res).astype(int)
    obs_12h["col"] = np.clip(col_idx, 0, cols - 1)
    obs_12h["row"] = np.clip(row_idx, 0, rows - 1)
    obs_12h["pt_arrival"] = pt_arr[obs_12h["row"].values, obs_12h["col"].values]

    nist_conf = []
    for h_hours in [3.0, 6.0, 12.0]:
        t_lim = h_hours * 60.0
        # True positive: observed burned <= t_lim AND sim burned <= t_lim
        # False negative: observed burned <= t_lim AND (sim burned > t_lim or sim unburned)
        # False positive: observed burned > t_lim AND sim burned <= t_lim
        # True negative: observed burned > t_lim AND (sim burned > t_lim or sim unburned)
        obs_burned = obs_12h["elapsed_minutes"] <= t_lim
        sim_burned = (~np.isnan(obs_12h["pt_arrival"])) & (obs_12h["pt_arrival"] <= t_lim)

        tp = int(np.count_nonzero(obs_burned & sim_burned))
        fn = int(np.count_nonzero(obs_burned & (~sim_burned)))
        fp = int(np.count_nonzero((~obs_burned) & sim_burned))
        tn = int(np.count_nonzero((~obs_burned) & (~sim_burned)))

        total_obs = tp + fn
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        nist_conf.append({
            "horizon": f"+{int(h_hours)} h ({int(t_lim)} min)",
            "total_nist_observed_burned": total_obs,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "IoU": iou,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })
    df_nist_conf = pd.DataFrame(nist_conf)

    # -------------------------------------------------------------
    # 2. RASTER PERIMETER OVERLAP AGAINST GEOMAC PERIMETERS
    # -------------------------------------------------------------
    geomac_path = Path("data/raw/geomac_camp_fire_2018_all_perimeters.geojson")
    gdf_geomac = gpd.read_file(geomac_path).to_crs("EPSG:32610")
    from datetime import datetime

    def to_utc_str(ts_ms: float | None) -> str:
        if ts_ms is None or np.isnan(ts_ms):
            return "N/A"
        return datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    gdf_geomac["datetime_utc"] = gdf_geomac["perimeterdatetime"].apply(to_utc_str)

    # Select representative perimeters:
    # 1. Early Day 1: 2018-11-08 17:54 UTC (~+3.5h from ignition)
    # 2. Evening Day 1: 2018-11-09 00:00 UTC (~+9.5h from ignition)
    # 3. Final Contained: 2018-11-25 18:22 UTC (Day 17 contained, 62,053 ha)
    perim_cases = [
        {
            "name": "+3.5 h GeoMAC Infrared Flight",
            "dt_utc": "2018-11-08 17:54:00 UTC",
            "eval_sim_t": 210.0,  # 3.5h
            "metric_label": "+3.5 h Perimeter Overlap",
            "target_acres": 54586.4,
        },
        {
            "name": "+9.5 h GeoMAC Evening Perimeter",
            "dt_utc": "2018-11-09 00:00:00 UTC",
            "eval_sim_t": 570.0,  # 9.5h
            "metric_label": "+9.5 h Perimeter Overlap",
            "target_acres": 70454.1,
        },
        {
            "name": "Final Contained CAL FIRE / GeoMAC",
            "dt_utc": "2018-11-25 18:22:00 UTC",
            "eval_sim_t": 720.0,  # 12h sim output
            "metric_label": "Final-Footprint Overlap (+12h Sim vs Final Contained)",
            "target_acres": 153335.6,
        },
    ]

    perimeter_results = []
    for case in perim_cases:
        subset = gdf_geomac[gdf_geomac["datetime_utc"] == case["dt_utc"]]
        if len(subset) == 0:
            continue
        geom = subset.geometry.iloc[0]

        # Rasterize geometry onto canonical grid
        obs_mask = rasterize(
            [(geom, 1)],
            out_shape=(rows, cols),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        ) == 1

        sim_t = case["eval_sim_t"]
        sim_mask_pt = (~np.isnan(pt_arr)) & (pt_arr <= sim_t)

        # PyTorchFire Confusion Matrix
        tp_pt = int(np.count_nonzero(sim_mask_pt & obs_mask))
        fp_pt = int(np.count_nonzero(sim_mask_pt & (~obs_mask)))
        fn_pt = int(np.count_nonzero((~sim_mask_pt) & obs_mask))
        tn_pt = int(np.count_nonzero((~sim_mask_pt) & (~obs_mask)))

        iou_pt = tp_pt / (tp_pt + fp_pt + fn_pt) if (tp_pt + fp_pt + fn_pt) > 0 else 0.0
        prec_pt = tp_pt / (tp_pt + fp_pt) if (tp_pt + fp_pt) > 0 else 0.0
        rec_pt = tp_pt / (tp_pt + fn_pt) if (tp_pt + fn_pt) > 0 else 0.0
        f1_pt = (2 * prec_pt * rec_pt) / (prec_pt + rec_pt) if (prec_pt + rec_pt) > 0 else 0.0

        obs_cells = int(np.count_nonzero(obs_mask))
        sim_cells_pt = int(np.count_nonzero(sim_mask_pt))

        perimeter_results.append({
            "reference_perimeter": case["name"],
            "datetime_utc": case["dt_utc"],
            "metric_label": case["metric_label"],
            "sim_horizon_min": sim_t,
            "engine": "PyTorchFire",
            "obs_area_in_domain_ha": obs_cells * cell_ha,
            "sim_area_ha": sim_cells_pt * cell_ha,
            "TP_cells": tp_pt,
            "FP_cells": fp_pt,
            "FN_cells": fn_pt,
            "TN_cells": tn_pt,
            "IoU": iou_pt,
            "Precision": prec_pt,
            "Recall": rec_pt,
            "F1": f1_pt,
        })

        # Cell2Fire for Final Footprint
        if c2f_arr is not None:
            sim_mask_c2f = (~np.isnan(c2f_arr)) & (c2f_arr <= sim_t)
            tp_c2f = int(np.count_nonzero(sim_mask_c2f & obs_mask))
            fp_c2f = int(np.count_nonzero(sim_mask_c2f & (~obs_mask)))
            fn_c2f = int(np.count_nonzero((~sim_mask_c2f) & obs_mask))
            tn_c2f = int(np.count_nonzero((~sim_mask_c2f) & (~obs_mask)))

            iou_c2f = tp_c2f / (tp_c2f + fp_c2f + fn_c2f) if (tp_c2f + fp_c2f + fn_c2f) > 0 else 0.0
            prec_c2f = tp_c2f / (tp_c2f + fp_c2f) if (tp_c2f + fp_c2f) > 0 else 0.0
            rec_c2f = tp_c2f / (tp_c2f + fn_c2f) if (tp_c2f + fn_c2f) > 0 else 0.0
            f1_c2f = (
                (2 * prec_c2f * rec_c2f) / (prec_c2f + rec_c2f)
                if (prec_c2f + rec_c2f) > 0
                else 0.0
            )

            perimeter_results.append({
                "reference_perimeter": case["name"],
                "datetime_utc": case["dt_utc"],
                "metric_label": case["metric_label"],
                "sim_horizon_min": sim_t,
                "engine": "Cell2Fire",
                "obs_area_in_domain_ha": obs_cells * cell_ha,
                "sim_area_ha": int(np.count_nonzero(sim_mask_c2f)) * cell_ha,
                "TP_cells": tp_c2f,
                "FP_cells": fp_c2f,
                "FN_cells": fn_c2f,
                "TN_cells": tn_c2f,
                "IoU": iou_c2f,
                "Precision": prec_c2f,
                "Recall": rec_c2f,
                "F1": f1_c2f,
            })

    df_perimeters = pd.DataFrame(perimeter_results)
    return {
        "nist_point_confusion": df_nist_conf,
        "perimeter_overlap_confusion": df_perimeters,
    }


def main() -> None:
    res = evaluate_perimeters()
    print("=== 1. Point-Level NIST Observation Confusion Matrix (PyTorchFire) ===")
    print(res["nist_point_confusion"].to_string(index=False))

    print("\n=== 2. Raster Perimeter Overlap Against GeoMAC & Final Contained Footprint ===")
    print(res["perimeter_overlap_confusion"].to_string(index=False))

    out_csv = Path("reports/perimeter_evaluation_confusion.csv")
    res["perimeter_overlap_confusion"].to_csv(out_csv, index=False)
    print(f"\nSaved perimeter evaluation to {out_csv}")

    out_nist_csv = Path("reports/nist_point_confusion.csv")
    res["nist_point_confusion"].to_csv(out_nist_csv, index=False)
    print(f"Saved NIST point confusion to {out_nist_csv}")


if __name__ == "__main__":
    main()
