"""Extract FEDS 375m 12-hourly VIIRS observations from Zenodo 15261836 for the training fires."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from scripts.prepare_training_fires_data import TRAINING_FIRES_CONFIG


def _stream_filter_feds_year(year: int, fire_bboxes: dict[str, dict[str, Any]]) -> dict[str, list[tuple[float, float, str]]]:
    """Stream Zenodo FEDS pixels CSV for a year and filter rows inside fire bounding boxes."""
    url = f"https://zenodo.org/api/records/15261836/files/feds_western_us_{year}_pixels.csv/content"
    results: dict[str, list[tuple[float, float, str]]] = {fn: [] for fn in fire_bboxes}
    print(f"Streaming FEDS {year} from {url}...")

    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        header = None
        lat_idx, lon_idx, dt_idx = 1, 2, 8
        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if header is None:
                header = raw_line.split(",")
                lat_idx = header.index("Lat")
                lon_idx = header.index("Lon")
                dt_idx = header.index("datetime")
                continue

            parts = raw_line.split(",")
            if len(parts) <= max(lat_idx, lon_idx, dt_idx):
                continue
            try:
                lat = float(parts[lat_idx])
                lon = float(parts[lon_idx])
            except ValueError:
                continue

            for fname, binfo in fire_bboxes.items():
                b = binfo["bbox_wgs84"]
                if b["lon_min"] <= lon <= b["lon_max"] and b["lat_min"] <= lat <= b["lat_max"]:
                    results[fname].append((lon, lat, parts[dt_idx]))

    for fname, pts in results.items():
        print(f"  [FEDS {year}] {fname}: extracted {len(pts)} VIIRS 375m active fire pixels")
    return results


def main() -> None:
    setup_df = pd.read_csv("reports/phase_1a_fires_setup.csv")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)

    by_year: dict[int, dict[str, dict[str, Any]]] = {}
    for _, row in setup_df.iterrows():
        fname = str(row["fire_name"])
        yr = int(row["year"])
        b_vals = [float(x.strip()) for x in str(row["bbox_wgs84"]).strip("[]").split(",")]
        u_vals = [float(x.strip()) for x in str(row["bbox_utm_epsg32610"]).strip("[]").split(",")]
        by_year.setdefault(yr, {})[fname] = {
            "bbox_wgs84": {"lon_min": b_vals[0], "lat_min": b_vals[1], "lon_max": b_vals[2], "lat_max": b_vals[3]},
            "bbox_utm": {"x_min": u_vals[0], "y_min": u_vals[1], "x_max": u_vals[2], "y_max": u_vals[3]},
            "cal_start_utc": str(row["cal_start_utc"]),
            "grid_size": TRAINING_FIRES_CONFIG[fname]["grid_size"],
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {yr: pool.submit(_stream_filter_feds_year, yr, fdict) for yr, fdict in by_year.items()}
        yr_results = {yr: fut.result() for yr, fut in futs.items()}

    # Rasterize onto 375m (~360m = 6x6 block of 60m cells) grid for each fire at +12h and +24h
    for yr, fdict in by_year.items():
        for fname, binfo in fdict.items():
            pts = yr_results[yr][fname]
            fire_tiles_dir = Path(f"data/tiles/fires/{fname.lower()}")
            grid_size = binfo["grid_size"]
            coarse_size = grid_size // 6  # 6 x 60m = 360m (~375m VIIRS scale)
            bbox_utm = binfo["bbox_utm"]
            t0 = pd.to_datetime(binfo["cal_start_utc"])

            mask_12h = np.zeros((coarse_size, coarse_size), dtype=bool)
            mask_24h = np.zeros((coarse_size, coarse_size), dtype=bool)

            if pts:
                df_pts = pd.DataFrame(pts, columns=["lon", "lat", "datetime"])
                df_pts["dt"] = pd.to_datetime(df_pts["datetime"])
                df_pts = df_pts[(df_pts["dt"] >= t0 - pd.Timedelta(hours=12)) & (df_pts["dt"] <= t0 + pd.Timedelta(hours=24))]
                if len(df_pts) > 0:
                    xs, ys = transformer.transform(df_pts["lon"].values, df_pts["lat"].values)
                    cols = np.clip(((xs - bbox_utm["x_min"]) / 360.0).astype(int), 0, coarse_size - 1)
                    rows = np.clip(((bbox_utm["y_max"] - ys) / 360.0).astype(int), 0, coarse_size - 1)
                    hrs = (df_pts["dt"] - t0).dt.total_seconds().values / 3600.0
                    for r_i, c_i, h_i in zip(rows, cols, hrs, strict=True):
                        # Dilate by 1 cell (375m footprint kernel)
                        r_slice = slice(max(0, r_i - 1), min(coarse_size, r_i + 2))
                        c_slice = slice(max(0, c_i - 1), min(coarse_size, c_i + 2))
                        if h_i <= 12.0:
                            mask_12h[r_slice, c_slice] = True
                        if h_i <= 24.0:
                            mask_24h[r_slice, c_slice] = True

            np.savez_compressed(
                fire_tiles_dir / "feds_375m.npz",
                mask_12h=mask_12h,
                mask_24h=mask_24h,
                num_viirs_points=len(pts),
            )
            print(
                f"Saved {fire_tiles_dir / 'feds_375m.npz'}: "
                f"12h cells={int(mask_12h.sum())}, 24h cells={int(mask_24h.sum())}"
            )


if __name__ == "__main__":
    main()
