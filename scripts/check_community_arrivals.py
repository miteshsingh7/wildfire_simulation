"""Examine PyTorchFire arrival time across Paradise town boundaries and key locations."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer

KEY_LOCATIONS = [
    {
        "name": "Concow (Rim Rd / Concow Rd)",
        "lat": 39.807822,
        "lon": -121.482963,
        "description": "Second ignition / early head fire crossing Rim Rd into Concow",
    },
    {
        "name": "Paradise East Edge (Pentz Rd / Bille Rd)",
        "lat": 39.769252,
        "lon": -121.579192,
        "description": "First impact on Paradise residential perimeter at Pentz Rd / Eden Roc",
    },
    {
        "name": "Paradise Core (Skyway / Wagstaff Rd)",
        "lat": 39.777646,
        "lon": -121.606300,
        "description": "Main commercial and arterial corridor in Paradise",
    },
    {
        "name": "Magalia (Skyway / Coutolenc Rd)",
        "lat": 39.811500,
        "lon": -121.597800,
        "description": "Northern town community of Magalia above Paradise",
    },
]

def main() -> None:
    with open("configs/phase0.yaml") as f:
        cfg = yaml.safe_load(f)

    bbox_utm = cfg["domain"]["bbox_utm"]
    res = float(cfg["domain"]["resolution"])

    obs_df = gpd.read_parquet("data/tiles/observations_nist.parquet")
    pt_path = Path("data/tiles/arrival_time_pytorchfire_seed42.tif")
    c2f_path = Path("data/tiles/arrival_time_cell2fire_seed42.tif")

    with rasterio.open(pt_path) as src:
        pt_arr = src.read(1)
        h, w = pt_arr.shape

    c2f_arr = None
    if c2f_path.exists():
        with rasterio.open(c2f_path) as src:
            c2f_arr = src.read(1)

    with rasterio.open("data/tiles/fbfm40.tif") as src:
        fbfm_arr = src.read(1)

    t = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)

    rows_out = []
    for loc in KEY_LOCATIONS:
        lat = loc["lat"]
        lon = loc["lon"]
        utm_x, utm_y = t.transform(lon, lat)
        col = int(np.clip((utm_x - bbox_utm["x_min"]) / res, 0, w - 1))
        row = int(np.clip((bbox_utm["y_max"] - utm_y) / res, 0, h - 1))

        # Query nearby NIST observations within 1.5 km
        obs_df["dist_m"] = np.sqrt((obs_df["utm_x"] - utm_x) ** 2 + (obs_df["utm_y"] - utm_y) ** 2)
        nearby = obs_df[obs_df["dist_m"] <= 1500.0]

        if len(nearby) > 0:
            earliest_idx = nearby["elapsed_minutes"].idxmin()
            earliest_row = nearby.loc[earliest_idx]
            earliest_obs = float(earliest_row["elapsed_minutes"])
            obs_type = str(earliest_row["fire_type"])
            obs_clock = f"{earliest_row['time']} PST"
        else:
            earliest_obs = np.nan
            obs_type = "N/A"
            obs_clock = "N/A"

        # Check fuel code at pixel
        fuel_code = int(fbfm_arr[row, col])

        # PyTorchFire arrival time: check exact, then expand window (30m, 150m, 300m, 500m, 1000m)
        exact_pt = pt_arr[row, col]

        # Check concentric radii
        min_arrivals = {}
        for radius_cells in [1, 5, 10, 17, 33]: # ~30m, 150m, 300m, 500m, 1000m
            r0, r1 = max(0, row - radius_cells), min(h, row + radius_cells + 1)
            c0, c1 = max(0, col - radius_cells), min(w, col + radius_cells + 1)
            sub = pt_arr[r0:r1, c0:c1]
            valid = sub[~np.isnan(sub)]
            min_arrivals[radius_cells * 30] = float(np.min(valid)) if len(valid) > 0 else np.nan

        # Best representative arrival at community boundary / nearest burnable front
        pt_arrival = exact_pt if not np.isnan(exact_pt) else min_arrivals[510]
        error_min = pt_arrival - earliest_obs if (not np.isnan(pt_arrival) and not np.isnan(earliest_obs)) else np.nan

        c2f_val = c2f_arr[row, col] if c2f_arr is not None else np.nan

        rows_out.append({
            "location": loc["name"],
            "fbfm_code": fuel_code,
            "observed_earliest_min": earliest_obs,
            "observed_clock_pst": obs_clock,
            "earliest_obs_type": obs_type,
            "pytorchfire_arrival_exact": exact_pt,
            "pytorchfire_arrival_500m": min_arrivals[510],
            "pytorchfire_arrival_1000m": min_arrivals[990],
            "pytorchfire_best_arrival": pt_arrival,
            "pytorchfire_error_min": error_min,
            "cell2fire_arrival": c2f_val,
        })

    df = pd.DataFrame(rows_out)
    print("=== Detailed Community Arrival Times ===")
    print(df.to_string(index=False))
    df.to_csv("reports/early_time_check_detailed.csv", index=False)

if __name__ == "__main__":
    main()
