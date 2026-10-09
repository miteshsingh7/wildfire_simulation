"""NIST Camp Fire observation data extraction and arrival-time rasterization.

Extracts all ground-truth fire observation data points from NIST Technical Note 2135
Appendix F (Maranghides et al., 2021), projects them to UTM 10N (EPSG:32610),
persists them as GeoParquet, and rasterizes observed fire arrival times to the
canonical 30m simulation grid.
"""

import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import requests
import yaml
from pypdf import PdfReader
from pyproj import Transformer
from rasterio.transform import from_bounds
from shapely.geometry import Point

NIST_TN2135_URL = "https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.2135.pdf"


def download_or_load_nist_report(raw_dir: Path) -> Path:
    """Ensure NIST TN 2135 PDF is downloaded locally in raw directory."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = raw_dir / "NIST.TN.2135.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 10_000_000:
        return pdf_path

    # Check /tmp cache first
    tmp_path = Path("/tmp/NIST.TN.2135.pdf")
    if tmp_path.exists() and tmp_path.stat().st_size > 10_000_000:
        import shutil

        shutil.copy(tmp_path, pdf_path)
        return pdf_path

    resp = requests.get(NIST_TN2135_URL, timeout=120)
    resp.raise_for_status()
    with open(pdf_path, "wb") as f:
        f.write(resp.content)
    return pdf_path


def parse_nist_appendix_f(pdf_path: Path) -> gpd.GeoDataFrame:
    """Parse fire observation records from Appendix F of NIST TN 2135."""
    reader = PdfReader(str(pdf_path))
    end_idx = 396
    for idx in range(286, min(415, len(reader.pages))):
        text = reader.pages[idx].extract_text()
        if "Appendix G" in text:
            end_idx = idx
            break

    # Line pattern matching date, time, latitude, longitude
    # Format e.g. 11/8 06:25 ... 39.814610 -121.434004
    pattern = re.compile(
        r"(\d{1,2}/\d{1,2})\s+(\d{2}:\d{2})\s+.*?"
        r"(\d{2}\.\d{4,7})\s+(-121\.\d{4,7})\s+([A-Za-z0-9_-]+)?\s+([A-Za-z0-9_-]+)?"
    )

    records: list[dict[str, Any]] = []
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)

    # Base ignition time: 11/8 06:25 PST
    base_hour, base_min = 6, 25

    for page_idx in range(286, end_idx):
        text = reader.pages[page_idx].extract_text()
        for line in text.splitlines():
            m = pattern.search(line)
            if m:
                date_str, time_str = m.group(1), m.group(2)
                lat_str, lon_str = m.group(3), m.group(4)
                lat = float(lat_str)
                lon = float(lon_str)

                # Parse hours and minutes
                th, tm = map(int, time_str.split(":"))
                # Elapsed minutes from ignition on Nov 8
                if date_str == "11/8":
                    elapsed_min = (th - base_hour) * 60 + (tm - base_min)
                else:
                    # Nov 9
                    elapsed_min = (th + 24 - base_hour) * 60 + (tm - base_min)

                utm_x, utm_y = transformer.transform(lon, lat)
                records.append(
                    {
                        "date": date_str,
                        "time": time_str,
                        "elapsed_minutes": elapsed_min,
                        "latitude": lat,
                        "longitude": lon,
                        "utm_x": utm_x,
                        "utm_y": utm_y,
                        "geometry": Point(lon, lat),
                        "page": page_idx + 1,
                    }
                )

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf


def rasterize_arrival_times(
    gdf: gpd.GeoDataFrame,
    bbox_utm: dict[str, float],
    grid: dict[str, int],
    out_path: Path,
    crs: str = "EPSG:32610",
    max_elapsed_minutes: float = 720.0,
) -> None:
    """Rasterize minimum arrival time (in minutes) onto canonical grid."""
    rows, cols = grid["rows"], grid["cols"]
    arrival_grid = np.full((rows, cols), np.nan, dtype=np.float32)

    res = 30.0
    x_min, y_max = bbox_utm["x_min"], bbox_utm["y_max"]

    # Filter to points within bbox and 12-hour simulation window
    valid_points = gdf[
        (gdf["elapsed_minutes"] >= 0)
        & (gdf["elapsed_minutes"] <= max_elapsed_minutes)
        & (gdf["utm_x"] >= bbox_utm["x_min"])
        & (gdf["utm_x"] <= bbox_utm["x_max"])
        & (gdf["utm_y"] >= bbox_utm["y_min"])
        & (gdf["utm_y"] <= bbox_utm["y_max"])
    ]

    for _, row in valid_points.iterrows():
        col = int((row["utm_x"] - x_min) / res)
        r_idx = int((y_max - row["utm_y"]) / res)
        if 0 <= r_idx < rows and 0 <= col < cols:
            val = float(row["elapsed_minutes"])
            if np.isnan(arrival_grid[r_idx, col]) or val < arrival_grid[r_idx, col]:
                arrival_grid[r_idx, col] = val

    transform = from_bounds(
        bbox_utm["x_min"], bbox_utm["y_min"], bbox_utm["x_max"], bbox_utm["y_max"], cols, rows
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype=rasterio.float32,
        crs=crs,
        transform=transform,
        nodata=-9999.0,
        compress="lzw",
    ) as dst:
        out_grid = np.where(np.isnan(arrival_grid), -9999.0, arrival_grid)
        dst.write(out_grid.astype(np.float32), 1)


def ingest_observations_pipeline(config_path: str = "configs/phase0.yaml") -> dict[str, Any]:
    """Execute full NIST observation ingestion and rasterization pipeline."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    tiles_dir = Path(cfg["data"]["tiles_dir"])
    bbox_utm = cfg["domain"]["bbox_utm"]
    grid = cfg["domain"]["grid"]
    crs = cfg["domain"]["crs"]

    pdf_path = download_or_load_nist_report(raw_dir)
    gdf = parse_nist_appendix_f(pdf_path)

    # Save vector points as GeoParquet
    parquet_path = tiles_dir / "observations_nist.parquet"
    gdf.to_parquet(parquet_path)

    # Rasterize arrival times
    raster_path = tiles_dir / "arrival_time_ground_truth.tif"
    rasterize_arrival_times(gdf, bbox_utm, grid, raster_path, crs=crs)

    return {
        "parquet_path": parquet_path,
        "raster_path": raster_path,
        "point_count": len(gdf),
        "gdf": gdf,
    }


if __name__ == "__main__":
    res = ingest_observations_pipeline()
    print(f"Ingested {res['point_count']} NIST observations.")
    print(f"Saved GeoParquet: {res['parquet_path']}")
    print(f"Saved Ground Truth Raster: {res['raster_path']}")
