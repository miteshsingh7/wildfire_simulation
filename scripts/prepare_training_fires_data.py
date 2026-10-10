"""Ingest pre-fire LANDFIRE (60m), NOAA HRRR weather, GOFER (~2km), and FEDS (375m) for the 7 training fires.

Also generates reports/phase_1a_fires.md with the complete per-fire setup and GOFER area-vs-time justification.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cfgrib
import geopandas as gpd
import numpy as np
import pandas as pd
from herbie import Herbie
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from wf052.data.landfire import align_and_save_layer, download_landfire_layer

# Canonical 7 training fires configuration
# Each fire uses a 24h unsuppressed rapid-growth calibration window (with 48h context)
# justified directly from GOFERC_summary.csv area vs time curves.
TRAINING_FIRES_CONFIG: dict[str, dict[str, Any]] = {
    "Kincade": {
        "year": 2019,
        "regime": "wind-driven",
        "cal_start_utc": "2019-10-24 05:00:00",
        "cal_start_step": 1,
        "cal_hours": 24,
        "grid_size": 720,  # 720 x 720 @ 60m = 43.2 km x 43.2 km = 186,624 ha
        "landfire_vintage": "LANDFIRE 2016 Remap (LF 2.0.0)",
        "justification": (
            "Immediate extreme Diablo NE wind run from ignition (846.8 ha at step 1 to 11,628.2 ha at +24h, "
            "13,555.0 ha at +48h; peak spread rate 4.89 km/h)."
        ),
    },
    "Bobcat": {
        "year": 2020,
        "regime": "plume",
        "cal_start_utc": "2020-09-06 20:00:00",
        "cal_start_step": 1,
        "cal_hours": 24,
        "grid_size": 720,  # 186,624 ha (34x the 24h fire size)
        "landfire_vintage": "LANDFIRE 2016 Remap (LF 2.0.0)",
        "justification": (
            "Unsuppressed initial upslope/plume run from Cogswell Dam up steep San Gabriel canyons "
            "(350.3 ha at step 1 to 5,379.7 ha at +24h, 7,399.4 ha at +48h; 15.4x expansion in 24h)."
        ),
    },
    "Glass": {
        "year": 2020,
        "regime": "wind-driven",
        "cal_start_utc": "2020-09-28 01:00:00",
        "cal_start_step": 15,
        "cal_hours": 24,
        "grid_size": 720,  # 186,624 ha
        "landfire_vintage": "LANDFIRE 2016 Remap (LF 2.0.0)",
        "justification": (
            "Peak 24h Diablo wind breakout across Napa/Sonoma ridges (2,616.4 ha at step 15 to 28,978.1 ha "
            "at step 39 [+26,361.7 ha in 24h], reaching 30,032.3 ha at +48h)."
        ),
    },
    "Creek": {
        "year": 2020,
        "regime": "plume",
        "cal_start_utc": "2020-09-05 14:00:00",
        "cal_start_step": 13,
        "cal_hours": 24,
        "grid_size": 840,  # 840 x 840 @ 60m = 50.4 km x 50.4 km = 254,016 ha
        "landfire_vintage": "LANDFIRE 2016 Remap (LF 2.0.0)",
        "justification": (
            "Historic pyrocumulonimbus plume-dominated eruption in Sierra conifer timber (1,862.9 ha at step 13 "
            "to 32,048.5 ha at +24h [+30,185.6 ha gain], reaching 48,486.4 ha at +48h)."
        ),
    },
    "Zogg": {
        "year": 2020,
        "regime": "wind-driven",
        "cal_start_utc": "2020-09-27 22:00:00",
        "cal_start_step": 1,
        "cal_hours": 24,
        "grid_size": 720,  # 186,624 ha
        "landfire_vintage": "LANDFIRE 2016 Remap (LF 2.0.0)",
        "justification": (
            "Explosive downslope N-NE wind run from ignition (73.7 ha at step 1 to 10,433.8 ha at +6h, "
            "16,454.9 ha at +24h, and 21,944.6 ha at +48h; 98.5% of total fire footprint)."
        ),
    },
    "Caldor": {
        "year": 2021,
        "regime": "terrain",
        "cal_start_utc": "2021-08-17 06:00:00",
        "cal_start_step": 53,
        "cal_hours": 24,
        "grid_size": 780,  # 780 x 780 @ 60m = 46.8 km x 46.8 km = 219,024 ha
        "landfire_vintage": "LANDFIRE 2016 Remap (LF 2.0.0)",
        "justification": (
            "Unsuppressed canyon-aligned breakout after initial sheltered smoldering (<1,444 ha in first 34h); "
            "surges from 3,958.6 ha at step 53 to 30,293.8 ha at +24h (+26,335.2 ha) and 32,168.7 ha at +48h."
        ),
    },
    "Tamarack": {
        "year": 2021,
        "regime": "terrain",
        "cal_start_utc": "2021-07-16 22:00:00",
        "cal_start_step": 292,
        "cal_hours": 24,
        "grid_size": 720,  # 186,624 ha
        "landfire_vintage": "LANDFIRE 2016 Remap (LF 2.0.0)",
        "justification": (
            "Lightning strike smoldered at <=609.2 ha for 12 days (steps 1-291) before unsuppressed high-elevation "
            "canyon/wind breakout at step 292: 716.1 ha to 11,634.8 ha at +24h and 20,114.6 ha at +48h."
        ),
    },
}


def compute_non_saturating_bbox_utm(
    prog_gdf_48h: gpd.GeoDataFrame,
    grid_size: int,
    res: float = 60.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute a square UTM Zone 10N (EPSG:32610) bounding box centered on the 48h fire progression."""
    prog_utm = prog_gdf_48h.to_crs("EPSG:32610")
    minx, miny, maxx, maxy = prog_utm.total_bounds
    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)

    half_span = 0.5 * grid_size * res
    # Snap to multiples of 60m
    x_min = float(np.floor((cx - half_span) / res) * res)
    y_min = float(np.floor((cy - half_span) / res) * res)
    x_max = x_min + grid_size * res
    y_max = y_min + grid_size * res

    bbox_utm = {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}

    transformer = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(x_min, y_min)
    lon_max, lat_max = transformer.transform(x_max, y_max)
    bbox_wgs84 = {
        "lon_min": float(min(lon_min, lon_max)),
        "lat_min": float(min(lat_min, lat_max)),
        "lon_max": float(max(lon_min, lon_max)),
        "lat_max": float(max(lat_min, lat_max)),
    }
    return bbox_utm, bbox_wgs84


def _download_single_hrrr_grib(target_str: str, save_dir: Path) -> tuple[str, Path | None]:
    """Download subset GRIB2 for a single UTC hour via Herbie."""
    herbie_obj = Herbie(target_str, model="hrrr", product="sfc", fxx=0, verbose=False, save_dir=str(save_dir))
    path = herbie_obj.download(":(UGRD:10 m|VGRD:10 m|GUST:surface|TMP:2 m|RH:2 m)")
    return target_str, Path(path) if path else None


def fetch_fire_hrrr_weather(
    fire_name: str,
    start_utc_str: str,
    duration_hours: int,
    bbox_wgs84: dict[str, float],
    out_parquet: Path,
) -> pd.DataFrame:
    """Download and extract hourly HRRR surface weather for a fire calibration window."""
    if out_parquet.exists():
        return pd.read_parquet(out_parquet)

    save_dir = Path("data/raw/hrrr")
    save_dir.mkdir(parents=True, exist_ok=True)

    start_dt = pd.to_datetime(start_utc_str)
    target_strs = [
        (start_dt + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:00")
        for h in range(duration_hours + 1)
    ]

    print(f"  [{fire_name}] Downloading {len(target_strs)} hourly HRRR subsets via Herbie...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_download_single_hrrr_grib, ts, save_dir) for ts in target_strs]
        downloaded = [f.result() for f in futures]

    records: list[dict[str, Any]] = []
    mask = None

    for h_idx, (ts, grib_path) in enumerate(downloaded):
        if grib_path is None or not grib_path.exists():
            raise RuntimeError(f"HRRR download failed for {fire_name} at {ts}")

        ds_list = cfgrib.open_datasets(str(grib_path))
        u_val, v_val, gust_val, temp_k, rh_val = 0.0, 0.0, 0.0, 290.0, 20.0

        for ds in ds_list:
            if mask is None and "longitude" in ds.coords and "latitude" in ds.coords:
                raw_lons = ds.longitude.values
                lons = np.where(raw_lons > 180, raw_lons - 360, raw_lons)
                lats = ds.latitude.values
                mask = (
                    (lons >= bbox_wgs84["lon_min"])
                    & (lons <= bbox_wgs84["lon_max"])
                    & (lats >= bbox_wgs84["lat_min"])
                    & (lats <= bbox_wgs84["lat_max"])
                )
                if not np.any(mask):
                    # Fallback to closest grid point if bbox is smaller than 3km cell
                    cx = 0.5 * (bbox_wgs84["lon_min"] + bbox_wgs84["lon_max"])
                    cy = 0.5 * (bbox_wgs84["lat_min"] + bbox_wgs84["lat_max"])
                    dist2 = (lons - cx) ** 2 + (lats - cy) ** 2
                    mask = dist2 == np.min(dist2)

            if "u10" in ds.data_vars:
                u_val = float(np.mean(ds["u10"].values[mask]))
            if "v10" in ds.data_vars:
                v_val = float(np.mean(ds["v10"].values[mask]))
            if "gust" in ds.data_vars:
                gust_val = float(np.mean(ds["gust"].values[mask]))
            if "t2m" in ds.data_vars:
                temp_k = float(np.mean(ds["t2m"].values[mask]))
            if "r2" in ds.data_vars:
                rh_val = float(np.mean(ds["r2"].values[mask]))

        ws_ms = float(np.sqrt(u_val**2 + v_val**2))
        wind_dir_deg = float(np.degrees(np.arctan2(-u_val, -v_val)) % 360.0)
        temp_c = temp_k - 273.15

        records.append({
            "hour_index": h_idx,
            "target_time": ts,
            "u10_ms": u_val,
            "v10_ms": v_val,
            "wind_speed_ms": ws_ms,
            "wind_speed_mph": ws_ms * 2.23694,
            "gust_ms": gust_val,
            "gust_mph": gust_val * 2.23694,
            "wind_direction_deg": wind_dir_deg,
            "temp_c": temp_c,
            "rh_pct": rh_val,
        })

    df = pd.DataFrame(records)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    return df


def rasterize_gofer_progression(
    prog_gdf: gpd.GeoDataFrame,
    start_step: int,
    cal_hours: int,
    bbox_utm: dict[str, float],
    grid_size: int,
    out_npz: Path,
) -> dict[str, Any]:
    """Rasterize GOFER hourly perimeters onto the 60m grid and aggregate to ~2km (33x33 block)."""
    prog_utm = prog_gdf.to_crs("EPSG:32610").sort_values("timestep").reset_index(drop=True)
    transform = from_bounds(
        bbox_utm["x_min"], bbox_utm["y_min"], bbox_utm["x_max"], bbox_utm["y_max"], grid_size, grid_size
    )

    # Initial state at start_step (t = 0)
    row_t0 = prog_utm[prog_utm["timestep"] == start_step]
    if len(row_t0) == 0:
        raise ValueError(f"Timestep {start_step} not found in GOFER progression")

    geom_t0 = [row_t0.iloc[0].geometry]
    mask_t0 = rasterize(geom_t0, out_shape=(grid_size, grid_size), transform=transform, fill=0, default_value=1).astype(bool)

    hourly_masks = np.zeros((cal_hours + 1, grid_size, grid_size), dtype=bool)
    hourly_masks[0] = mask_t0
    arrival_time_min = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    arrival_time_min[mask_t0] = 0.0

    hourly_areas_ha = [float(row_t0.iloc[0]["farea"] * 100.0)]

    for h in range(1, cal_hours + 1):
        step_h = start_step + h
        row_h = prog_utm[prog_utm["timestep"] == step_h]
        if len(row_h) > 0:
            geom_h = [row_h.iloc[0].geometry]
            mask_h = rasterize(
                geom_h, out_shape=(grid_size, grid_size), transform=transform, fill=0, default_value=1
            ).astype(bool)
            # Enforce monotonicity of cumulative burned area
            mask_h = mask_h | hourly_masks[h - 1]
            hourly_areas_ha.append(float(row_h.iloc[0]["farea"] * 100.0))
        else:
            mask_h = hourly_masks[h - 1].copy()
            hourly_areas_ha.append(hourly_areas_ha[-1])

        hourly_masks[h] = mask_h
        newly_burned = mask_h & np.isnan(arrival_time_min)
        arrival_time_min[newly_burned] = float(h * 60.0)

    # Also check +48h area from prog_utm
    row_48 = prog_utm[prog_utm["timestep"] == min(start_step + 48, int(prog_utm["timestep"].max()))]
    area_48h_ha = float(row_48.iloc[0]["farea"] * 100.0) if len(row_48) > 0 else hourly_areas_ha[-1]

    np.savez_compressed(
        out_npz,
        hourly_masks=hourly_masks,
        arrival_time_min=arrival_time_min,
        hourly_areas_ha=np.array(hourly_areas_ha, dtype=np.float32),
        area_48h_ha=np.array(area_48h_ha, dtype=np.float32),
    )
    return {
        "area_t0_ha": hourly_areas_ha[0],
        "area_6h_ha": hourly_areas_ha[min(6, cal_hours)],
        "area_12h_ha": hourly_areas_ha[min(12, cal_hours)],
        "area_24h_ha": hourly_areas_ha[min(24, cal_hours)],
        "area_48h_ha": area_48h_ha,
    }


def main() -> None:
    gofer_dir = Path("data/raw/gofer/GOFER")
    fire_data = pd.read_csv(gofer_dir / "fireData.csv")
    prog_gdf = gpd.read_file(gofer_dir / "GOFER_Combined/GOFERC_fireProg.shp")

    summary_rows: list[dict[str, Any]] = []

    for fname, cfg in TRAINING_FIRES_CONFIG.items():
        print("\n=======================================================")
        print(f"Preparing Training Fire: {fname} ({cfg['year']}) [{cfg['regime']}]")
        print("=======================================================")

        f_meta = fire_data[fire_data["fname"] == fname].iloc[0]
        f_prog = prog_gdf[prog_gdf["fname"] == fname].sort_values("timestep").reset_index(drop=True)

        start_step = cfg["cal_start_step"]
        cal_hours = cfg["cal_hours"]
        grid_size = cfg["grid_size"]

        # Subset 48h progression for bbox sizing
        f_prog_48h = f_prog[
            (f_prog["timestep"] >= start_step) & (f_prog["timestep"] <= start_step + 48)
        ]
        bbox_utm, bbox_wgs84 = compute_non_saturating_bbox_utm(f_prog_48h, grid_size, res=60.0)
        grid_spec = {"rows": grid_size, "cols": grid_size}

        fire_raw_dir = Path(f"data/raw/fires/{fname.lower()}")
        fire_tiles_dir = Path(f"data/tiles/fires/{fname.lower()}")
        fire_tiles_dir.mkdir(parents=True, exist_ok=True)

        # 1. Download & align pre-fire LANDFIRE 2016 Remap layers at 60m
        for layer in ["fbfm40", "elevation", "canopy_cover"]:
            raw_tif = download_landfire_layer(layer, bbox_utm, grid_spec, fire_raw_dir)
            tile_tif = fire_tiles_dir / f"{layer}.tif"
            if not tile_tif.exists():
                align_and_save_layer(raw_tif, tile_tif, layer, bbox_utm, grid_spec, crs="EPSG:32610")
        print(f"  [{fname}] Aligned 60m LANDFIRE tiles ({grid_size}x{grid_size}) saved to {fire_tiles_dir}")

        # 2. Download & extract hourly HRRR weather for the 24h window
        weather_parquet = fire_tiles_dir / "hrrr_weather_hourly.parquet"
        w_df = fetch_fire_hrrr_weather(
            fname, cfg["cal_start_utc"], cal_hours, bbox_wgs84, weather_parquet
        )
        print(
            f"  [{fname}] HRRR weather ready: mean wind {w_df['wind_speed_ms'].mean():.2f} m/s "
            f"(max {w_df['wind_speed_ms'].max():.2f} m/s), mean dir {w_df['wind_direction_deg'].mean():.1f} deg"
        )

        # 3. Rasterize GOFER progression
        gofer_npz = fire_tiles_dir / "gofer_progression.npz"
        g_stats = rasterize_gofer_progression(
            f_prog, start_step, cal_hours, bbox_utm, grid_size, gofer_npz
        )
        print(
            f"  [{fname}] GOFER progression rasterized: "
            f"t0={g_stats['area_t0_ha']:.1f} ha -> +6h={g_stats['area_6h_ha']:.1f} ha -> "
            f"+12h={g_stats['area_12h_ha']:.1f} ha -> +24h={g_stats['area_24h_ha']:.1f} ha "
            f"(+48h={g_stats['area_48h_ha']:.1f} ha)"
        )

        # Save metadata json/yaml for this fire
        domain_area_ha = (grid_size * grid_size * 3600.0) / 10_000.0
        end_utc = (pd.to_datetime(cfg["cal_start_utc"]) + pd.Timedelta(hours=cal_hours)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        end_48h_utc = (pd.to_datetime(cfg["cal_start_utc"]) + pd.Timedelta(hours=48)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        summary_rows.append({
            "fire_name": fname,
            "year": cfg["year"],
            "regime": cfg["regime"],
            "in_gofer_28": True,
            "gofer_total_perimeters": len(f_prog),
            "gofer_timestep_h": 1.0,
            "official_acres": int(f_meta["acres_official"]),
            "official_ha": round(float(f_meta["acres_official"]) * 0.4046856, 1),
            "cal_start_utc": cfg["cal_start_utc"],
            "cal_end_24h_utc": end_utc,
            "cal_end_48h_utc": end_48h_utc,
            "cal_start_step": start_step,
            "area_t0_ha": round(g_stats["area_t0_ha"], 1),
            "area_6h_ha": round(g_stats["area_6h_ha"], 1),
            "area_12h_ha": round(g_stats["area_12h_ha"], 1),
            "area_24h_ha": round(g_stats["area_24h_ha"], 1),
            "area_48h_ha": round(g_stats["area_48h_ha"], 1),
            "grid_shape": f"{grid_size}x{grid_size}",
            "domain_area_ha": round(domain_area_ha, 1),
            "domain_to_24h_ratio": round(domain_area_ha / max(g_stats["area_24h_ha"], 1.0), 1),
            "bbox_wgs84": (
                f"[{bbox_wgs84['lon_min']:.3f}, {bbox_wgs84['lat_min']:.3f}, "
                f"{bbox_wgs84['lon_max']:.3f}, {bbox_wgs84['lat_max']:.3f}]"
            ),
            "bbox_utm_epsg32610": (
                f"[{int(bbox_utm['x_min'])}, {int(bbox_utm['y_min'])}, "
                f"{int(bbox_utm['x_max'])}, {int(bbox_utm['y_max'])}]"
            ),
            "landfire_vintage": cfg["landfire_vintage"],
            "mean_wind_ms": round(float(w_df["wind_speed_ms"].mean()), 2),
            "max_wind_ms": round(float(w_df["wind_speed_ms"].max()), 2),
            "justification": cfg["justification"],
        })

    out_df = pd.DataFrame(summary_rows)
    out_csv = Path("reports/phase_1a_fires_setup.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved training fires setup summary to {out_csv}")


if __name__ == "__main__":
    main()
