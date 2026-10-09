"""Parse NIST TN 2135 Appendix F observations with Type of Fire classification.

Classifies observation records into:
- vegetative ('V')
- spot_fire ('S')
- structural ('R' residential, 'C' commercial)
- other ('O', etc.)

Evaluates PyTorchFire simulated arrival times against ground truth for each category,
computing signed bias, absolute error (mean and median), and fraction never burned.
"""

import re
from collections import Counter
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from pypdf import PdfReader
from pyproj import Transformer
from shapely.geometry import Point


def parse_nist_with_fire_types(pdf_path: Path) -> gpd.GeoDataFrame:
    """Extract all observation records including Type of Fire categorization."""
    reader = PdfReader(str(pdf_path))
    end_idx = 396
    for idx in range(286, min(415, len(reader.pages))):
        text = reader.pages[idx].extract_text()
        if "Appendix G" in text:
            end_idx = idx
            break

    records: list[dict[str, Any]] = []
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
    base_hour, base_min = 6, 25

    # Regex matching table lines with Date, Time, Latitude, Longitude
    # Table columns: Source# Date Time TimeSource ObsWindow Latitude Longitude LocationSource InfoSource Notes Type Residual SSI
    line_pattern = re.compile(
        r"(\d{1,2}/\d{1,2})\s+(\d{2}:\d{2})\s+.*?"
        r"(\d{2}\.\d{4,7})\s+(-121\.\d{4,7})"
    )

    for page_idx in range(286, end_idx):
        text = reader.pages[page_idx].extract_text()
        lines = text.splitlines()
        for line in lines:
            m = line_pattern.search(line)
            if not m:
                continue

            date_str = m.group(1)
            time_str = m.group(2)
            lat = float(m.group(3))
            lon = float(m.group(4))

            # Extract Type of Fire code from the tail of the line
            # Looking for isolated letter code V, S, R, C, O
            tail = line[m.end():]
            type_match = re.search(r"\b([VSRCO])\b(?:\s+|$)", tail)
            if type_match:
                raw_type = type_match.group(1).upper()
            else:
                # Check entire line after coordinates
                alt_match = re.search(r"\s+([VSRCO])\s*$", line)
                raw_type = alt_match.group(1).upper() if alt_match else "O"

            # Normalize to standardized categories
            if raw_type == "V":
                category = "vegetative"
            elif raw_type == "S":
                category = "spot_fire"
            elif raw_type in ("R", "C"):
                category = "structural"
            else:
                category = "other"

            # Parse elapsed minutes
            th, tm = map(int, time_str.split(":"))
            if date_str == "11/8":
                elapsed_min = (th - base_hour) * 60 + (tm - base_min)
            else:
                elapsed_min = (th + 24 - base_hour) * 60 + (tm - base_min)

            utm_x, utm_y = transformer.transform(lon, lat)
            records.append({
                "date": date_str,
                "time": time_str,
                "elapsed_minutes": float(elapsed_min),
                "latitude": lat,
                "longitude": lon,
                "utm_x": utm_x,
                "utm_y": utm_y,
                "fire_type_code": raw_type,
                "fire_type": category,
                "geometry": Point(lon, lat),
                "page": page_idx + 1,
            })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf


def evaluate_split_metrics(
    gdf: gpd.GeoDataFrame,
    arrival_raster: np.ndarray,
    config: dict[str, Any],
    max_horizon_minutes: float = 720.0,
) -> pd.DataFrame:
    """Compute arrival-time metrics partitioned by fire observation type."""
    bbox_utm = config["domain"]["bbox_utm"]
    res = float(config["domain"]["resolution"])
    h, w = arrival_raster.shape

    # Filter to 12-hour simulation window
    valid = gdf[(gdf["elapsed_minutes"] >= 0) & (gdf["elapsed_minutes"] <= max_horizon_minutes)].copy()

    sim_arrivals: list[float] = []
    for _, row in valid.iterrows():
        x = float(row["utm_x"])
        y = float(row["utm_y"])
        col = int(np.clip((x - bbox_utm["x_min"]) / res, 0, w - 1))
        row_idx = int(np.clip((bbox_utm["y_max"] - y) / res, 0, h - 1))
        t_sim = arrival_raster[row_idx, col]
        sim_arrivals.append(float(t_sim))

    valid["sim_arrival_min"] = sim_arrivals
    valid["burned_by_sim"] = (~np.isnan(valid["sim_arrival_min"])) & (valid["sim_arrival_min"] <= max_horizon_minutes)
    valid["signed_bias"] = valid["sim_arrival_min"] - valid["elapsed_minutes"]
    valid["abs_error"] = np.abs(valid["signed_bias"])

    # Aggregate by fire_type and overall
    categories = ["vegetative", "spot_fire", "structural", "other"]
    summary_rows = []

    for cat in categories + ["ALL"]:
        sub = valid if cat == "ALL" else valid[valid["fire_type"] == cat]
        total_pts = len(sub)
        if total_pts == 0:
            continue

        burned_pts = sub[sub["burned_by_sim"]]
        n_burned = len(burned_pts)
        n_unburned = total_pts - n_burned
        unburned_pct = (n_unburned / total_pts) * 100.0

        if n_burned > 0:
            signed_bias_mean = float(burned_pts["signed_bias"].mean())
            signed_bias_median = float(burned_pts["signed_bias"].median())
            abs_err_mean = float(burned_pts["abs_error"].mean())
            abs_err_median = float(burned_pts["abs_error"].median())
        else:
            signed_bias_mean = np.nan
            signed_bias_median = np.nan
            abs_err_mean = np.nan
            abs_err_median = np.nan

        summary_rows.append({
            "fire_type": cat,
            "total_points": total_pts,
            "burned_points": n_burned,
            "unburned_points": n_unburned,
            "unburned_pct": unburned_pct,
            "signed_bias_mean_min": signed_bias_mean,
            "signed_bias_median_min": signed_bias_median,
            "abs_error_mae_min": abs_err_mean,
            "abs_error_median_min": abs_err_median,
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def main() -> None:
    pdf_path = Path("data/raw/NIST.TN.2135.pdf")
    out_parquet = Path("data/tiles/observations_nist.parquet")
    config_path = Path("configs/phase0.yaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("Parsing NIST Appendix F with Type of Fire extraction...")
    gdf = parse_nist_with_fire_types(pdf_path)
    print(f"Total extracted observation points: {len(gdf)}")
    counts = Counter(gdf["fire_type"])
    print(f"Breakdown: {dict(counts)}")

    # Update observations_nist.parquet
    gdf.to_parquet(out_parquet)
    print(f"Updated {out_parquet}")

    # Load canonical PyTorchFire arrival raster
    raster_path = Path("data/tiles/arrival_time_pytorchfire_seed42.tif")
    if raster_path.exists():
        with rasterio.open(raster_path) as src:
            arrival_arr = src.read(1)
        summary_df = evaluate_split_metrics(gdf, arrival_arr, cfg)
        print("\n=== NIST Ground-Truth Arrival Time Metric Split (PyTorchFire Seed 42) ===")
        print(summary_df.to_string(index=False))

        out_csv = Path("reports/metric_split_nist.csv")
        summary_df.to_csv(out_csv, index=False)
        print(f"\nSaved metric split to {out_csv}")


if __name__ == "__main__":
    main()
