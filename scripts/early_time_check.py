"""Early-time check comparing simulated vs observed fire arrival at key locations (Task 5).

Evaluates:
- Concow (Rim Rd / Concow Reservoir)
- Paradise East Edge (Pentz Rd / Bille Rd)
- Paradise Core (Skyway / Wagstaff Rd)
- Magalia (Skyway / Coutolenc Rd)

Reports observed arrival time (minutes after ignition and clock time PST),
fire observation type (vegetative vs spot fire), simulated arrival time (exact and
500m community boundary), and timing error.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer

KEY_LOCATIONS = [
    {
        "location": "Concow (Rim Rd / Concow Rd)",
        "lat": 39.807822,
        "lon": -121.482963,
        "description": "Head fire crossing Rim Rd into Concow / Concow Reservoir",
    },
    {
        "location": "Paradise East Edge (Pentz Rd / Bille Rd)",
        "lat": 39.769252,
        "lon": -121.579192,
        "description": "First impact on Paradise residential perimeter at Pentz Rd",
    },
    {
        "location": "Paradise Core (Skyway / Wagstaff Rd)",
        "lat": 39.777646,
        "lon": -121.606300,
        "description": "Central commercial arterial corridor in Paradise",
    },
    {
        "location": "Magalia (Skyway / Coutolenc Rd)",
        "lat": 39.811500,
        "lon": -121.597800,
        "description": "Upper ridge community north of Paradise",
    },
]


def run_early_time_check() -> pd.DataFrame:
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

    results = []
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

        fuel_code = int(fbfm_arr[row, col])

        # Exact pixel arrival
        exact_pt = pt_arr[row, col]

        # 500m radius arrival (17 cells)
        r0, r1 = max(0, row - 17), min(h, row + 18)
        c0, c1 = max(0, col - 17), min(w, col + 18)
        sub_pt = pt_arr[r0:r1, c0:c1]
        valid_pt = sub_pt[~np.isnan(sub_pt)]
        boundary_arrival_500m = float(np.min(valid_pt)) if len(valid_pt) > 0 else np.nan

        # Representative arrival
        rep_pt_arrival = exact_pt if not np.isnan(exact_pt) else boundary_arrival_500m
        pt_error = (
            rep_pt_arrival - earliest_obs
            if (not np.isnan(rep_pt_arrival) and not np.isnan(earliest_obs))
            else np.nan
        )

        # Cell2Fire
        c2f_val = c2f_arr[row, col] if c2f_arr is not None else np.nan
        sub_c2f = c2f_arr[r0:r1, c0:c1] if c2f_arr is not None else None
        valid_c2f = sub_c2f[~np.isnan(sub_c2f)] if sub_c2f is not None else []
        c2f_boundary = float(np.min(valid_c2f)) if len(valid_c2f) > 0 else np.nan

        results.append({
            "location": loc["location"],
            "fbfm_code": fuel_code,
            "observed_earliest_min": earliest_obs,
            "observed_clock_pst": obs_clock,
            "earliest_obs_type": obs_type,
            "pytorchfire_arrival_exact_min": exact_pt,
            "pytorchfire_arrival_500m_min": boundary_arrival_500m,
            "pytorchfire_rep_arrival_min": rep_pt_arrival,
            "pytorchfire_timing_error_min": pt_error,
            "cell2fire_arrival_exact_min": c2f_val,
            "cell2fire_arrival_500m_min": c2f_boundary,
        })

    df = pd.DataFrame(results)
    return df


def main() -> None:
    print("=== Early-Time Fire Arrival Verification: Key Communities ===")
    df = run_early_time_check()
    print(df.to_string(index=False))
    out_csv = Path("reports/early_time_check.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved early-time comparison to {out_csv}")


if __name__ == "__main__":
    main()
