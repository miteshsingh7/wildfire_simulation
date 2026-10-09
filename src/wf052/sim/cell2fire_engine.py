"""Cell2Fire simulation runner with mapped inputs from LANDFIRE and HRRR.

Translates Scott & Burgan 40 fuel models (FBFM40) to Canadian FBP fuel types,
topography (elevation, slope, aspect), and hourly HRRR weather into Cell2Fire
instance formats (Forest.asc, Data.csv, Weather.csv, Ignitions.csv).
Executes the compiled C++ Cell2Fire binary and parses arrival-time outputs.
"""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml

# FBFM40 to Canadian FBP System crosswalk mapping
# (grid_value, FBP_code)
# Non-burnable models: grid 101, "NF "
FBFM40_TO_FBP: dict[int, tuple[int, str]] = {
    # Non-burnable
    91: (101, "NF "),  # Urban / Developed
    92: (101, "NF "),  # Snow / Ice
    93: (101, "NF "),  # Agriculture
    98: (101, "NF "),  # Open Water
    99: (101, "NF "),  # Barren / Rock
    # Grass (GR): Matted / Standing Grass
    101: (32, "O1b"),  # GR1 -> O-1b Standing grass
    102: (32, "O1b"),  # GR2 -> O-1b Standing grass
    103: (32, "O1b"),  # GR3 -> O-1b Standing grass
    104: (32, "O1b"),  # GR4 -> O-1b Standing grass
    105: (31, "O1a"),  # GR5 -> O-1a Matted grass
    106: (31, "O1a"),  # GR6 -> O-1a Matted grass
    107: (32, "O1b"),  # GR7 -> O-1b Standing grass
    108: (32, "O1b"),  # GR8 -> O-1b Standing grass
    # Grass-Shrub (GS): High surface spread rate
    121: (32, "O1b"),  # GS1 -> O-1b
    122: (32, "O1b"),  # GS2 -> O-1b
    123: (32, "O1b"),  # GS3 -> O-1b
    124: (32, "O1b"),  # GS4 -> O-1b
    # Shrub (SH): Chaparral / coastal brush
    141: (7, "C7 "),  # SH1 -> C-7 Ponderosa / open shrub
    142: (7, "C7 "),  # SH2 -> C-7
    143: (7, "C7 "),  # SH3 -> C-7
    144: (7, "C7 "),  # SH4 -> C-7
    145: (7, "C7 "),  # SH5 -> C-7 Heavy chaparral
    146: (7, "C7 "),  # SH6 -> C-7
    147: (7, "C7 "),  # SH7 -> C-7 Very high load shrub
    # Timber-Understory (TU): Pine/oak with heavy understory
    161: (7, "C7 "),  # TU1 -> C-7
    162: (7, "C7 "),  # TU2 -> C-7
    163: (7, "C7 "),  # TU3 -> C-7
    164: (3, "C3 "),  # TU4 -> C-3 Mature pine
    165: (7, "C7 "),  # TU5 -> C-7 Sierra mixed conifer / heavy understory
    # Timber Litter (TL): Needles / broadleaf litter
    181: (5, "C5 "),  # TL1 -> C-5 Conifer litter
    182: (11, "D1 "),  # TL2 -> D-1 Hardwood litter
    183: (5, "C5 "),  # TL3 -> C-5
    184: (5, "C5 "),  # TL4 -> C-5
    185: (5, "C5 "),  # TL5 -> C-5
    186: (11, "D1 "),  # TL6 -> D-1 Moderate broadleaf litter
    187: (7, "C7 "),  # TL7 -> C-7 Large downed logs
    188: (7, "C7 "),  # TL8 -> C-7 Long-needle pine litter
    189: (11, "D1 "),  # TL9 -> D-1 Very high broadleaf litter
    # Slash-Blowdown (SB)
    201: (21, "S1 "),  # SB1 -> S-1 Slash
    202: (21, "S1 "),  # SB2 -> S-1
    203: (22, "S2 "),  # SB3 -> S-2 White spruce slash
    204: (23, "S3 "),  # SB4 -> S-3 Heavy slash
}


@dataclass(frozen=True)
class Cell2FireRunResult:
    """Encapsulates execution metrics and arrival time raster from Cell2Fire."""

    engine: str
    seed: int
    wall_clock_seconds: float
    peak_rss_mb: float
    final_burned_cells: int
    burned_area_ha: float
    arrival_time_raster: np.ndarray
    grid_shape: tuple[int, int]


def prepare_cell2fire_instance(
    tiles_dir: Path,
    instance_dir: Path,
    config: dict[str, Any],
) -> tuple[int, int, int]:
    """Prepare Forest.asc, Data.csv, Weather.csv, and Ignitions.csv for Cell2Fire.

    Returns:
        tuple of (nrows, ncols, ignition_cell_id_1based)
    """
    instance_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tiles_dir / "fbfm40.tif") as src:
        fbfm_arr = src.read(1)
        h, w = fbfm_arr.shape

    with rasterio.open(tiles_dir / "elevation.tif") as src:
        elev_arr = src.read(1).astype(np.float32)

    with rasterio.open(tiles_dir / "slope.tif") as src:
        slope_deg = src.read(1).astype(np.float32)
        # Convert slope degrees to percent: ps = tan(deg) * 100
        slope_pct = np.tan(np.radians(slope_deg)) * 100.0

    with rasterio.open(tiles_dir / "aspect.tif") as src:
        aspect_arr = src.read(1).astype(np.float32)

    # 1. Generate Forest.asc
    forest_asc_path = instance_dir / "Forest.asc"
    bbox_utm = config["domain"]["bbox_utm"]
    cellsize = float(config["domain"]["resolution"])
    xllcorner = float(bbox_utm["x_min"])
    yllcorner = float(bbox_utm["y_min"])

    # Map FBFM40 to grid_values
    grid_val_arr = np.full((h, w), 101, dtype=np.int32)

    # Pre-build 2D code lookup for fast serialization
    for r in range(h):
        for c in range(w):
            fbfm_code = int(fbfm_arr[r, c])
            if fbfm_code in FBFM40_TO_FBP:
                gval, _ = FBFM40_TO_FBP[fbfm_code]
            else:
                gval = 101
            grid_val_arr[r, c] = gval

    print(f"Writing Forest.asc ({h}x{w})...")
    with open(forest_asc_path, "w") as f:
        f.write(f"ncols {w}\n")
        f.write(f"nrows {h}\n")
        f.write(f"xllcorner {xllcorner:.1f}\n")
        f.write(f"yllcorner {yllcorner:.1f}\n")
        f.write(f"cellsize {int(cellsize)}\n")
        f.write("NODATA_value -9999\n")
        for r in range(h):
            row_str = " ".join(str(grid_val_arr[r, c]) for c in range(w))
            f.write(row_str + "\n")

    # 2. Generate Data.csv
    data_csv_path = instance_dir / "Data.csv"
    print(f"Writing Data.csv ({h * w} cells)...")
    lat_min = float(config["domain"]["bbox_wgs84"]["lat_min"])
    lat_max = float(config["domain"]["bbox_wgs84"]["lat_max"])
    lon_min = float(config["domain"]["bbox_wgs84"]["lon_min"])
    lon_max = float(config["domain"]["bbox_wgs84"]["lon_max"])
    center_lat = (lat_min + lat_max) / 2.0
    center_lon = (lon_min + lon_max) / 2.0

    with open(data_csv_path, "w") as f:
        f.write(
            "fueltype,mon,jd,M,jd_min,lat,lon,elev,ffmc,ws,waz,bui,ps,saz,pc,pdf,gfl,cur,time,pattern\n"
        )
        for r in range(h):
            for c in range(w):
                fbfm_code = int(fbfm_arr[r, c])
                _, fcode = FBFM40_TO_FBP.get(fbfm_code, (101, "NF "))
                elev = float(elev_arr[r, c])
                ps = max(0.0, float(slope_pct[r, c]))
                saz = float(aspect_arr[r, c])
                # In Canadian FBP, grass fuels (O-1a, O-1b) require non-zero grass fuel load (gfl)
                # and curing percentage (cur) to compute surface fuel consumption and intensity.
                if "O1" in fcode:
                    gfl_val = 0.35
                    cur_val = 100
                else:
                    gfl_val = 0.0
                    cur_val = 0
                gfl_str = f"{gfl_val:.2f}" if gfl_val > 0 else ""
                cur_str = str(cur_val) if cur_val > 0 else ""
                f.write(
                    f"{fcode.strip()},,,,,{center_lat:.4f},{center_lon:.4f},{elev:.1f},,,,,{ps:.1f},{saz:.1f},,,{gfl_str},{cur_str},20,\n"
                )

    # 3. Generate Weather.csv
    weather_df = pd.read_parquet(tiles_dir / "hrrr_weather_hourly.parquet")
    weather_csv_path = instance_dir / "Weather.csv"
    print(f"Writing Weather.csv ({len(weather_df)} hourly periods)...")
    with open(weather_csv_path, "w") as f:
        f.write("Scenario,datetime,APCP,TMP,RH,WS,WD,FFMC,DMC,DC,ISI,BUI,FWI\n")
        for _idx, row in weather_df.iterrows():
            dt_str = str(row["target_time"])[:16]
            tmp_c = float(row["temp_c"])
            rh_pct = 10.0  # Butte County / Jarbo Gap observed RH: 8-12%
            # Wind speed in km/h for Canadian FBP
            ws_kmh = float(row["wind_speed_ms"]) * 3.6
            wd_deg = float(row["wind_direction_deg"])
            # Camp Fire Red Flag conditions: extreme dry fine fuels
            ffmc = 93.0
            dmc = 85.0
            dc = 550.0
            bui = 125.0
            # Initial Spread Index
            if ws_kmh < 40.0:
                fw = np.exp(0.05039 * ws_kmh)
            else:
                fw = 12.0 * (1.0 - np.exp(-0.0818 * (ws_kmh - 28.0)))
            isi = float(0.208 * fw * 12.5)
            fwi = float(0.1 * isi * bui / 10.0)
            f.write(
                f"CAMP,{dt_str},0.0,{tmp_c:.1f},{rh_pct:.0f},{ws_kmh:.1f},"
                f"{wd_deg:.0f},{ffmc:.1f},{dmc:.1f},{dc:.1f},{isi:.2f},{bui:.1f},{fwi:.1f}\n"
            )

    # 4. Generate Ignitions.csv
    # Compute ignition cell from Pulga coordinates. If exact tower pixel is classified
    # as 91 (Developed / Transmission corridor = Non-fuel), snap to nearest burnable pixel.
    ign_x = float(config["event"]["ignition"]["utm_x"])
    ign_y = float(config["event"]["ignition"]["utm_y"])
    raw_col = int(np.clip((ign_x - xllcorner) / cellsize, 0, w - 1))
    raw_row = int(np.clip((float(bbox_utm["y_max"]) - ign_y) / cellsize, 0, h - 1))

    # Search outward up to 3 cells (90m) for burnable grass/timber cell
    best_r, best_c = raw_row, raw_col
    if grid_val_arr[raw_row, raw_col] == 101:  # Non-fuel
        found = False
        for rad in range(1, 4):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    nr, nc = raw_row + dr, raw_col + dc
                    if 0 <= nr < h and 0 <= nc < w and grid_val_arr[nr, nc] != 101:
                        best_r, best_c = nr, nc
                        found = True
                        break
                if found:
                    break

    ign_cell_id = best_r * w + best_c + 1  # 1-based indexing
    ignitions_csv_path = instance_dir / "Ignitions.csv"
    with open(ignitions_csv_path, "w") as f:
        f.write("Year,Ncell\n")
        f.write(f"1,{ign_cell_id}\n")

    print(
        f"Ignition set at row={best_r}, col={best_c} "
        f"(raw row={raw_row}, col={raw_col}), cell_id={ign_cell_id}"
    )
    return h, w, ign_cell_id


def run_cell2fire_simulation(
    config_path: str = "configs/phase0.yaml",
    instance_dir: Path | str = "data/cell2fire_instance",
    output_dir: Path | str = "data/cell2fire_output",
    seed: int = 42,
    binary_path: str = "external/Cell2Fire/cell2fire/Cell2FireC/Cell2Fire",
    fire_period_len: float = 60.0,
    max_fire_periods: int = 12,
) -> Cell2FireRunResult:
    """Execute wildfire simulation using Cell2Fire C++ engine."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    inst_path = Path(instance_dir)
    out_path = Path(output_dir) / f"run_seed_{seed}"
    out_path.mkdir(parents=True, exist_ok=True)

    tiles_dir = Path(cfg["data"]["tiles_dir"])

    # Check if instance already prepared and valid
    instance_complete = (
        (inst_path / "Data.csv").exists()
        and (inst_path / "Forest.asc").exists()
        and (inst_path / "Weather.csv").exists()
        and (inst_path / "Weather.csv").stat().st_size > 200
        and (inst_path / "Ignitions.csv").exists()
    )
    if not instance_complete:
        h, w, ign_cell = prepare_cell2fire_instance(tiles_dir, inst_path, cfg)
    else:
        with rasterio.open(tiles_dir / "fbfm40.tif") as src:
            h, w = src.height, src.width
        ign_x = float(cfg["event"]["ignition"]["utm_x"])
        ign_y = float(cfg["event"]["ignition"]["utm_y"])
        bbox_utm = cfg["domain"]["bbox_utm"]
        cellsize = float(cfg["domain"]["resolution"])
        ign_col = int(np.clip((ign_x - bbox_utm["x_min"]) / cellsize, 0, w - 1))
        ign_row = int(np.clip((bbox_utm["y_max"] - ign_y) / cellsize, 0, h - 1))
        ign_cell = ign_row * w + ign_col + 1

    cmd = [
        binary_path,
        "--input-instance-folder",
        str(inst_path) + "/",
        "--output-folder",
        str(out_path) + "/",
        "--ignitions",
        "--sim-years",
        "1",
        "--nsims",
        "1",
        "--final-grid",
        "--grids",
        "--output-messages",
        "--weather",
        "rows",
        "--nweathers",
        "1",
        "--Fire-Period-Length",
        str(fire_period_len),
        "--Weather-Period-Length",
        "60",
        "--max-fire-periods",
        str(max_fire_periods),
        "--seed",
        str(seed),
        "--ROS-CV",
        "0.0",
    ]

    print(f"Executing Cell2Fire (seed {seed})...")
    t_start = time.perf_counter()
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    wall_clock_sec = time.perf_counter() - t_start

    print(f"Cell2Fire finished in {wall_clock_sec:.2f}s")
    # Parse output arrival raster
    arrival_time_min = np.full((h, w), np.nan, dtype=np.float32)

    # 1. Parse Messages file if available
    msg_file = out_path / "Messages" / "MessagesFile01.csv"
    if msg_file.exists():
        with open(msg_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    to_cell = int(parts[1]) - 1  # 0-based index
                    period = float(parts[2])
                    arr_r = to_cell // w
                    arr_c = to_cell % w
                    if 0 <= arr_r < h and 0 <= arr_c < w:
                        time_min = period * fire_period_len
                        cur_val = arrival_time_min[arr_r, arr_c]
                        if np.isnan(cur_val) or time_min < cur_val:
                            arrival_time_min[arr_r, arr_c] = time_min

    # Ignition cell is at time 0.0
    ign_r = (ign_cell - 1) // w
    ign_c = (ign_cell - 1) % w
    arrival_time_min[ign_r, ign_c] = 0.0

    # 2. Cross-check / augment from Grids/Grids1/ForestGrid*.csv if available
    grids_dir = out_path / "Grids" / "Grids1"
    if grids_dir.exists():
        for p in range(1, max_fire_periods + 1):
            grid_file = grids_dir / f"ForestGrid{p:02d}.csv"
            if not grid_file.exists():
                grid_file = grids_dir / f"ForestGrid{p}.csv"
            if grid_file.exists():
                grid_arr = np.loadtxt(grid_file, delimiter=",", dtype=np.int8)
                if grid_arr.shape == (h, w):
                    burnt_mask = grid_arr == 1
                    newly_burnt = burnt_mask & np.isnan(arrival_time_min)
                    arrival_time_min[newly_burnt] = float(p * fire_period_len)

    final_burned_cells = int(np.count_nonzero(~np.isnan(arrival_time_min)))
    burned_area_ha = (final_burned_cells * 900.0) / 10_000.0

    return Cell2FireRunResult(
        engine="Cell2Fire",
        seed=seed,
        wall_clock_seconds=wall_clock_sec,
        peak_rss_mb=0.0,
        final_burned_cells=final_burned_cells,
        burned_area_ha=burned_area_ha,
        arrival_time_raster=arrival_time_min,
        grid_shape=(h, w),
    )


if __name__ == "__main__":
    res = run_cell2fire_simulation(seed=42)
    print(
        f"Cell2Fire run complete: {res.final_burned_cells} burned cells "
        f"({res.burned_area_ha:.1f} ha) in {res.wall_clock_seconds:.2f}s"
    )
