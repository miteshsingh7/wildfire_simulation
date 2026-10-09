"""PyTorchFire simulation runner with mapped inputs from LANDFIRE and HRRR.

Translates Scott & Burgan 40 fuel models (FBFM40), topography (elevation, slope),
canopy cover, and hourly HRRR wind vectors into PyTorchFire's Cellular Automata
parameters and environment tensors.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from pytorchfire import WildfireModel

# FBFM40 to PyTorchFire p_veg mapping dictionary
# Non-burnable models: p_veg = -1.0 (stops CA fire spread completely)
# Flammable models: mapped to [0.0, 0.6] based on fuel bed depth, load, and rate of spread
FBFM40_TO_P_VEG: dict[int, float] = {
    # Non-burnable
    91: -1.0,  # Urban / Developed
    92: -1.0,  # Snow / Ice
    93: -1.0,  # Agriculture
    98: -1.0,  # Open Water
    99: -1.0,  # Barren / Rock
    # Grass (GR): High surface-area-to-volume ratio, rapid spread
    101: 0.30,  # GR1: Short, sparse dry grass
    102: 0.40,  # GR2: Low load dry grass
    103: 0.45,  # GR3: Low load very dry grass
    104: 0.50,  # GR4: Moderate load dry grass
    105: 0.55,  # GR5: Low load humid grass
    106: 0.55,  # GR6: Moderate load humid grass
    107: 0.60,  # GR7: High load dry grass
    108: 0.60,  # GR8: High load very dry grass
    # Grass-Shrub (GS)
    121: 0.35,  # GS1: Low load dry grass-shrub
    122: 0.45,  # GS2: Moderate load dry grass-shrub
    123: 0.50,  # GS3: Moderate load humid grass-shrub
    124: 0.55,  # GS4: High load humid grass-shrub
    # Shrub (SH): Dense chaparral / brush
    141: 0.35,  # SH1: Low load dry shrub
    142: 0.45,  # SH2: Moderate load dry shrub
    143: 0.40,  # SH3: Moderate load humid shrub
    144: 0.50,  # SH4: Low load humid shrub
    145: 0.55,  # SH5: High load dry shrub (chaparral)
    146: 0.55,  # SH6: Low load humid shrub
    147: 0.60,  # SH7: Very high load dry shrub
    # Timber-Understory (TU): Pine/oak overstory with shrub understory
    161: 0.25,  # TU1: Low load dry timber-grass-shrub
    162: 0.35,  # TU2: Moderate load humid timber-shrub
    163: 0.35,  # TU3: Moderate load humid timber-grass-shrub
    164: 0.30,  # TU4: Dwarf conifer understory
    165: 0.40,  # TU5: Very high load dry timber-shrub
    # Timber Litter (TL): Compact needle cast and hardwood leaf litter
    181: 0.15,  # TL1: Low load compact conifer litter
    182: 0.20,  # TL2: Low load broadleaf litter
    183: 0.25,  # TL3: Moderate load conifer litter
    184: 0.25,  # TL4: Small downed logs
    185: 0.30,  # TL5: High load conifer litter
    186: 0.35,  # TL6: Moderate load broadleaf litter
    187: 0.35,  # TL7: Large downed logs
    188: 0.40,  # TL8: Long-needle pine litter
    189: 0.45,  # TL9: Very high load broadleaf litter
    # Slash-Blowdown (SB)
    201: 0.30,  # SB1: Low load activity fuel
    202: 0.40,  # SB2: Moderate load activity fuel
    203: 0.50,  # SB3: High load activity fuel
    204: 0.60,  # SB4: High load blowdown
}


def load_environmental_tensors(
    tiles_dir: Path,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], tuple[int, int]]:
    """Load aligned GeoTIFF rasters and construct PyTorchFire environment dict."""
    with rasterio.open(tiles_dir / "fbfm40.tif") as src:
        fbfm_arr = src.read(1)
        h, w = fbfm_arr.shape

    with rasterio.open(tiles_dir / "elevation.tif") as src:
        elev_arr = src.read(1).astype(np.float32)

    with rasterio.open(tiles_dir / "canopy_cover.tif") as src:
        cc_arr = src.read(1).astype(np.float32)

    # 1. Map FBFM40 to p_veg
    p_veg_arr = np.full((h, w), 0.20, dtype=np.float32)
    for code, p_val in FBFM40_TO_P_VEG.items():
        p_veg_arr[fbfm_arr == code] = p_val
    # Unknown/negative codes mapped to non-burnable
    p_veg_arr[fbfm_arr <= 0] = -1.0

    # 2. Map Canopy Cover to p_den (density factor between -0.2 and +0.2)
    p_den_arr = np.clip((cc_arr / 100.0) * 0.4 - 0.2, -0.2, 0.2).astype(np.float32)

    # 3. Compute 3x3 directional slope angles in degrees from Elevation
    res = 30.0
    slope_arr = np.zeros((h, w, 3, 3), dtype=np.float32)
    # Offsets: row_off in [-1, 0, 1], col_off in [-1, 0, 1]
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            dist = np.sqrt(dr**2 + dc**2) * res
            # Shifted elevation
            elev_shifted = np.roll(np.roll(elev_arr, -dr, axis=0), -dc, axis=1)
            dz = elev_shifted - elev_arr
            angle_deg = np.degrees(np.arctan(dz / dist))
            slope_arr[:, :, dr + 1, dc + 1] = angle_deg

    env_tensors = {
        "p_veg": torch.from_numpy(p_veg_arr).to(device),
        "p_den": torch.from_numpy(p_den_arr).to(device),
        "slope": torch.from_numpy(slope_arr).to(device),
        "elev": torch.from_numpy(elev_arr).to(device),
    }
    return env_tensors, (h, w)


def run_pytorchfire_simulation(
    config_path: str = "configs/phase0.yaml",
    seed: int = 42,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Execute 12-hour wildfire spread simulation with PyTorchFire."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if device_name is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    tiles_dir = Path(cfg["data"]["tiles_dir"])
    env_tensors, (h, w) = load_environmental_tensors(tiles_dir, device)

    # Load hourly weather
    weather_df = pd.read_parquet(tiles_dir / "hrrr_weather_hourly.parquet")

    # Set up ignition cell at Pulga (Feather River Canyon / Hwy 70)
    # Ignition UTM: X = 633827, Y = 4408316
    bbox_utm = cfg["domain"]["bbox_utm"]
    res = cfg["domain"]["resolution"]
    ign_x = cfg["event"]["ignition"]["utm_x"]
    ign_y = cfg["event"]["ignition"]["utm_y"]
    ign_col = int(np.clip((ign_x - bbox_utm["x_min"]) / res, 0, w - 1))
    ign_row = int(np.clip((bbox_utm["y_max"] - ign_y) / res, 0, h - 1))

    initial_ignition = torch.zeros((h, w), dtype=torch.bool, device=device)
    # Seed 3x3 ignition kernel around initial tower point
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if 0 <= ign_row + dr < h and 0 <= ign_col + dc < w:
                initial_ignition[ign_row + dr, ign_col + dc] = True

    # Initial hourly wind from Hour 0
    w0 = weather_df.iloc[0]
    initial_ws = float(w0["wind_speed_ms"])
    # Convert met wind dir (from N) to CA towards dir (from E CCW)
    initial_towards_deg = float((270.0 - w0["wind_direction_deg"]) % 360.0)

    wind_velocity = torch.full((h, w), initial_ws, dtype=torch.float32, device=device)
    wind_towards_direction = torch.full(
        (h, w), initial_towards_deg, dtype=torch.float32, device=device
    )

    env_data = {
        "p_veg": env_tensors["p_veg"],
        "p_den": env_tensors["p_den"],
        "wind_velocity": wind_velocity,
        "wind_towards_direction": wind_towards_direction,
        "slope": env_tensors["slope"],
        "initial_ignition": initial_ignition,
    }

    # Calibration parameters for extreme dry Sierra foothill conditions
    params = {
        "a": torch.tensor(0.06, device=device),
        "p_h": torch.tensor(0.58, device=device),
        "c_1": torch.tensor(0.045, device=device),
        "c_2": torch.tensor(0.15, device=device),
        "p_continue": torch.tensor(0.3, device=device),
    }

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize model
    model = WildfireModel(env_data=env_data, params=params).to(device)
    model.eval()
    model._initialize_seed(seed)

    # Arrival time grid: init with NaN
    arrival_time_min = np.full((h, w), np.nan, dtype=np.float32)
    # Ignition cells arrive at t = 0 min
    arrival_time_min[initial_ignition.cpu().numpy()] = 0.0

    steps_per_hour = cfg["engines"]["pytorchfire"].get("steps_per_hour", 60)
    total_hours = cfg["event"]["simulation"]["duration_hours"]
    total_steps = steps_per_hour * total_hours
    min_per_step = 60.0 / steps_per_hour

    t_start = time.perf_counter()
    hourly_burned_counts: list[int] = []

    for step in range(1, total_steps + 1):
        # Update hourly wind if crossing hour boundary
        current_hour = min(step // steps_per_hour, len(weather_df) - 1)
        w_cur = weather_df.iloc[current_hour]
        cur_ws = float(w_cur["wind_speed_ms"])
        cur_towards = float((270.0 - w_cur["wind_direction_deg"]) % 360.0)
        model.wind_velocity.fill_(cur_ws)
        model.wind_towards_direction.fill_(cur_towards)

        model.compute()

        # Update arrival times for newly burned/burning cells
        state = model.state
        active_or_burnt = (state[0] | state[1]).cpu().numpy()
        new_cells = active_or_burnt & np.isnan(arrival_time_min)
        if np.any(new_cells):
            arrival_time_min[new_cells] = float(step * min_per_step)

        if step % steps_per_hour == 0:
            total_burned = int(np.count_nonzero(~np.isnan(arrival_time_min)))
            hourly_burned_counts.append(total_burned)

    wall_clock_sec = time.perf_counter() - t_start
    final_burned_cells = int(np.count_nonzero(~np.isnan(arrival_time_min)))
    burned_area_ha = (final_burned_cells * 900.0) / 10_000.0  # 30x30m = 900m2 per cell

    return {
        "engine": "PyTorchFire",
        "seed": seed,
        "device": str(device),
        "wall_clock_seconds": wall_clock_sec,
        "final_burned_cells": final_burned_cells,
        "burned_area_ha": burned_area_ha,
        "hourly_burned_counts": hourly_burned_counts,
        "arrival_time_raster": arrival_time_min,
        "grid_shape": (h, w),
    }


if __name__ == "__main__":
    res = run_pytorchfire_simulation(seed=42)
    print(f"Simulation completed in {res['wall_clock_seconds']:.2f}s")
    print(f"Burned cells: {res['final_burned_cells']} ({res['burned_area_ha']:.1f} ha)")
    print(f"Hourly progression: {res['hourly_burned_counts']}")
