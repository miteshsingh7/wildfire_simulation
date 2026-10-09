"""10-seed benchmarking and validation runner for PyTorchFire vs Cell2Fire.

Executes 10 stochastic simulation runs for each engine on the 2018 Camp Fire domain,
measures runtime, memory (peak RSS), stochastic seed variance, evaluates arrival-time
MAE/median error and perimeter IoU/F1 against NIST ground truth, and exports
structured metrics for the Phase 0 report.
"""

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml

from wf052.eval.comparison import evaluate_simulation
from wf052.sim.cell2fire_engine import run_cell2fire_simulation
from wf052.sim.pytorchfire_engine import run_pytorchfire_simulation


def get_peak_rss_mb() -> float:
    """Return peak resident memory size in Megabytes."""
    rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS, ru_maxrss is in bytes; on Linux, it is in kilobytes
    if os.uname().sysname == "Darwin":
        return float(rusage) / (1024.0 * 1024.0)
    return float(rusage) / 1024.0


def save_raster_geotiff(
    arr: np.ndarray,
    out_path: Path,
    reference_tif: Path = Path("data/tiles/fbfm40.tif"),
) -> None:
    """Save 2D array as GeoTIFF inheriting spatial reference from reference raster."""
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()
    profile.update(
        dtype=rasterio.float32,
        count=1,
        nodata=np.nan,
        compress="lzw",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def run_benchmark_suite(
    config_path: str = "configs/phase0.yaml",
    engines: list[str] | None = None,
    seeds: list[int] | None = None,
    save_canonical_rasters: bool = True,
) -> dict[str, Any]:
    """Execute multi-seed benchmarks for requested engines."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if engines is None:
        engines = ["PyTorchFire", "Cell2Fire"]
    if seeds is None:
        seeds = cfg["engines"].get("seeds", [42, 101, 202, 303, 404, 505, 606, 707, 808, 909])

    tiles_dir = Path(cfg["data"]["tiles_dir"])
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    results_by_engine: dict[str, list[dict[str, Any]]] = {}

    for engine in engines:
        print(f"\n{'=' * 70}\nBenchmarking Engine: {engine} ({len(seeds)} seeds)\n{'=' * 70}")
        engine_runs: list[dict[str, Any]] = []

        for idx, seed in enumerate(seeds, 1):
            print(f"[{engine}] Run {idx}/{len(seeds)} (Seed: {seed})...")

            if engine.lower() == "pytorchfire":
                res = run_pytorchfire_simulation(config_path=config_path, seed=seed)
                wall_sec = res["wall_clock_seconds"]
                arrival_arr = res["arrival_time_raster"]
                burned_cells = res["final_burned_cells"]
                burned_ha = res["burned_area_ha"]
            elif engine.lower() == "cell2fire":
                res_c2f = run_cell2fire_simulation(config_path=config_path, seed=seed)
                wall_sec = res_c2f.wall_clock_seconds
                arrival_arr = res_c2f.arrival_time_raster
                burned_cells = res_c2f.final_burned_cells
                burned_ha = res_c2f.burned_area_ha
            else:
                raise ValueError(f"Unknown engine: {engine}")

            peak_rss = get_peak_rss_mb()

            # Evaluate against NIST observations
            eval_metrics = evaluate_simulation(
                arrival_raster=arrival_arr,
                config=cfg,
                engine_name=engine,
                seed=seed,
                wall_clock_seconds=wall_sec,
                tiles_dir=tiles_dir,
            )

            # Save canonical raster for seed 42
            if save_canonical_rasters and seed == 42:
                raster_out = tiles_dir / f"arrival_time_{engine.lower()}_seed42.tif"
                save_raster_geotiff(arrival_arr, raster_out)
                print(f"Saved canonical arrival raster: {raster_out}")

            run_dict = {
                "engine": engine,
                "seed": seed,
                "wall_clock_seconds": wall_sec,
                "peak_rss_mb": peak_rss,
                "final_burned_cells": burned_cells,
                "burned_area_ha": burned_ha,
                "total_nist_points": eval_metrics.total_nist_points_12h,
                "burned_nist_points": eval_metrics.burned_nist_points,
                "unburned_fraction_pct": eval_metrics.unburned_fraction_pct,
                "arrival_error_mae_min": eval_metrics.arrival_error_mae_minutes,
                "arrival_error_median_min": eval_metrics.arrival_error_median_minutes,
                "arrival_error_std_min": eval_metrics.arrival_error_std_minutes,
                "arrival_bias_min": eval_metrics.arrival_bias_minutes,
                "iou_3h": eval_metrics.horizon_3h.iou,
                "f1_3h": eval_metrics.horizon_3h.f1_score,
                "iou_6h": eval_metrics.horizon_6h.iou,
                "f1_6h": eval_metrics.horizon_6h.f1_score,
                "iou_12h": eval_metrics.horizon_12h.iou,
                "f1_12h": eval_metrics.horizon_12h.f1_score,
            }
            engine_runs.append(run_dict)
            print(
                f"  -> Burned: {burned_cells} cells ({burned_ha:.1f} ha) | "
                f"Runtime: {wall_sec:.2f}s | "
                f"MAE: {eval_metrics.arrival_error_mae_minutes:.1f}m | "
                f"Unburned NIST: {eval_metrics.unburned_fraction_pct:.1f}% | "
                f"IoU@12h: {eval_metrics.horizon_12h.iou:.4f}"
            )

        results_by_engine[engine] = engine_runs

    # Compute summary aggregates (mean +/- std)
    summary: dict[str, Any] = {}
    for engine, runs in results_by_engine.items():
        keys = [
            "wall_clock_seconds",
            "peak_rss_mb",
            "final_burned_cells",
            "burned_area_ha",
            "unburned_fraction_pct",
            "arrival_error_mae_min",
            "arrival_error_median_min",
            "iou_3h",
            "f1_3h",
            "iou_6h",
            "f1_6h",
            "iou_12h",
            "f1_12h",
        ]
        agg: dict[str, Any] = {"runs_count": len(runs)}
        for k in keys:
            vals = [r[k] for r in runs if not np.isnan(r[k])]
            if vals:
                agg[f"{k}_mean"] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals))
            else:
                agg[f"{k}_mean"] = float("nan")
                agg[f"{k}_std"] = float("nan")
        summary[engine] = agg

    output_payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "runs": results_by_engine,
    }

    out_json = reports_dir / "benchmark_metrics.json"
    with open(out_json, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nAll benchmark metrics exported to {out_json}")

    return output_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 10-seed benchmarks for PyTorchFire and Cell2Fire"
    )
    parser.add_argument("--engines", nargs="+", default=["PyTorchFire", "Cell2Fire"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    run_benchmark_suite(engines=args.engines, seeds=args.seeds)
