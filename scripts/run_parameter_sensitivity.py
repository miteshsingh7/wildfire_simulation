"""One-at-a-Time (OAT) parameter sensitivity analysis on 2 training fires (Kincade & Bobcat).

Evaluates every PyTorchFire parameter around its default (including P_base=0.50) at +6h, +12h, and +24h,
and computes sensitivity elasticities to select the parameters to fit in Step 3.
"""

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from wf052.sim.teacher_calibrator import (
    TeacherParams,
    evaluate_simulation_multiscale,
    load_fire_dataset_60m,
    simulate_teacher_60m,
)

PARAMETER_SWEEPS: dict[str, dict[str, Any]] = {
    "p_base": {
        "default": 0.50,
        "values": [0.06, 0.12, 0.25, 0.50, 0.70],
        "description": "Base propagation probability (p_h / P_base)",
    },
    "a": {
        "default": 0.06,
        "values": [0.00, 0.03, 0.06, 0.10, 0.15],
        "description": "Topographic slope scaling factor exp(a * slope_deg)",
    },
    "c_1": {
        "default": 0.045,
        "values": [0.01, 0.025, 0.045, 0.08, 0.12],
        "description": "Isotropic wind velocity scaling factor exp(c_1 * V)",
    },
    "c_2": {
        "default": 0.15,
        "values": [0.02, 0.08, 0.15, 0.30, 0.50],
        "description": "Wind directional eccentricity / flank suppression exp(c_2 * V * (cos(theta) - 1))",
    },
    "p_continue": {
        "default": 0.30,
        "values": [0.05, 0.15, 0.30, 0.50, 0.70],
        "description": "Probability a burning cell continues burning at next step",
    },
    "fuel_scale_grass_shrub": {
        "default": 1.00,
        "values": [0.30, 0.60, 1.00, 1.40],
        "description": "Multiplier on p_veg for Grass & Shrub fuel models (FBFM40 101-147)",
    },
    "fuel_scale_timber": {
        "default": 1.00,
        "values": [0.30, 0.60, 1.00, 1.40],
        "description": "Multiplier on p_veg for Timber & Slash fuel models (FBFM40 161-204)",
    },
    "canopy_den_scale": {
        "default": 1.00,
        "values": [0.00, 0.50, 1.00, 2.00],
        "description": "Multiplier on canopy cover density modifier p_den",
    },
    "wind_moisture_damping": {
        "default": 0.00,
        "values": [0.00, 0.50, 1.50, 3.00],
        "description": "Time-varying RH wind/moisture damping coefficient gamma_rh",
    },
}


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running OAT Parameter Sensitivity on device: {device}")

    fires = [
        ("Kincade", "wind-driven", Path("data/tiles/fires/kincade")),
        ("Bobcat", "plume", Path("data/tiles/fires/bobcat")),
    ]

    datasets = {
        fname: load_fire_dataset_60m(fname, regime, tdir, device=device)
        for fname, regime, tdir in fires
    }

    default_params = TeacherParams()
    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for fname, ds in datasets.items():
        print(f"\n=== Sensitivity Sweep on {fname} ({ds.regime}) ===")
        for param_name, p_info in PARAMETER_SWEEPS.items():
            for val in p_info["values"]:
                cur_params = replace(default_params, **{param_name: float(val)})
                sim_res = simulate_teacher_60m(ds, cur_params, duration_hours=24, steps_per_hour=30, seed=42)
                m = evaluate_simulation_multiscale(ds, sim_res, horizons=(6, 12, 24))

                row = {
                    "fire_name": fname,
                    "regime": ds.regime,
                    "parameter": param_name,
                    "default_val": p_info["default"],
                    "test_val": float(val),
                    "is_default": abs(float(val) - float(p_info["default"])) < 1e-6,
                    "obs_area_6h_ha": round(m[6]["obs_area_ha"], 1),
                    "sim_area_6h_ha": round(m[6]["sim_area_ha"], 1),
                    "area_ratio_6h": round(m[6]["area_ratio"], 2),
                    "arr_mae_6h_min": round(m[6]["arrival_mae_min"], 1),
                    "iou_6h_2km": round(m[6]["iou_2km"], 4),
                    "obs_area_12h_ha": round(m[12]["obs_area_ha"], 1),
                    "sim_area_12h_ha": round(m[12]["sim_area_ha"], 1),
                    "area_ratio_12h": round(m[12]["area_ratio"], 2),
                    "arr_mae_12h_min": round(m[12]["arrival_mae_min"], 1),
                    "iou_12h_2km": round(m[12]["iou_2km"], 4),
                    "obs_area_24h_ha": round(m[24]["obs_area_ha"], 1),
                    "sim_area_24h_ha": round(m[24]["sim_area_ha"], 1),
                    "area_ratio_24h": round(m[24]["area_ratio"], 2),
                    "arr_mae_24h_min": round(m[24]["arrival_mae_min"], 1),
                    "arr_mae_burned_24h_min": round(m[24]["arrival_mae_burned_min"], 1),
                    "iou_24h_2km": round(m[24]["iou_2km"], 4),
                }
                rows.append(row)
                print(
                    f"  [{fname}] {param_name}={val:<6.3f} | "
                    f"+6h: {row['sim_area_6h_ha']:8.1f} ha (MAE {row['arr_mae_6h_min']:5.1f}m) | "
                    f"+12h: {row['sim_area_12h_ha']:8.1f} ha (MAE {row['arr_mae_12h_min']:5.1f}m) | "
                    f"+24h: {row['sim_area_24h_ha']:8.1f} ha (MAE {row['arr_mae_24h_min']:5.1f}m, IoU {row['iou_24h_2km']:.3f})"
                )

    df = pd.DataFrame(rows)
    out_csv = Path("reports/parameter_sensitivity.csv")
    df.to_csv(out_csv, index=False)

    # Summarize sensitivity range (max - min) / default for each parameter
    summary_rows = []
    for param_name, p_info in PARAMETER_SWEEPS.items():
        sub = df[df["parameter"] == param_name]
        for fname in ["Kincade", "Bobcat"]:
            f_sub = sub[sub["fire_name"] == fname]
            def_row = f_sub[f_sub["is_default"]].iloc[0]
            def_area_24 = float(def_row["sim_area_24h_ha"])
            min_area_24 = float(f_sub["sim_area_24h_ha"].min())
            max_area_24 = float(f_sub["sim_area_24h_ha"].max())
            rel_swing_pct = ((max_area_24 - min_area_24) / max(def_area_24, 1.0)) * 100.0

            min_mae_24 = float(f_sub["arr_mae_24h_min"].min())
            max_mae_24 = float(f_sub["arr_mae_24h_min"].max())
            mae_swing_min = max_mae_24 - min_mae_24

            summary_rows.append({
                "parameter": param_name,
                "default": p_info["default"],
                "fire_name": fname,
                "min_area_24h_ha": round(min_area_24, 1),
                "max_area_24h_ha": round(max_area_24, 1),
                "rel_area_swing_24h_pct": round(rel_swing_pct, 1),
                "min_mae_24h_min": round(min_mae_24, 1),
                "max_mae_24h_min": round(max_mae_24, 1),
                "mae_swing_24h_min": round(mae_swing_min, 1),
                "description": p_info["description"],
            })

    sum_df = pd.DataFrame(summary_rows)
    sum_csv = Path("reports/parameter_sensitivity_summary.csv")
    sum_df.to_csv(sum_csv, index=False)
    print(f"\nCompleted sensitivity analysis in {time.perf_counter() - t_start:.2f}s.")
    print(f"Saved {out_csv} and {sum_csv}")
    print("\n=== Sensitivity Summary (24h Area Swing % & Arrival MAE Swing) ===")
    print(sum_df.to_string(index=False))


if __name__ == "__main__":
    main()
