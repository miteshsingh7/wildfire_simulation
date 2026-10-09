"""Benchmark PyTorchFire on a 256x256 tile for CPU and MPS (Task 8).

Runs a 12-hour simulation (720 steps) on:
1. CPU
2. Apple Silicon MPS (Metal Performance Shaders)

Measures:
- Wall-clock seconds
- ms per time step
- Throughput: Scenarios per hour (3,600 s / runtime)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from pytorchfire import WildfireModel

from wf052.sim.pytorchfire_engine import FBFM40_TO_P_VEG


def prepare_256_tile_tensors(
    tiles_dir: Path,
    device: torch.device,
    tile_size: int = 256,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, pd.DataFrame]:
    """Extract a 256x256 tile around the ignition point and prepare tensors."""
    # Load raw layers
    with rasterio.open(tiles_dir / "fbfm40.tif") as src:
        fbfm_arr = src.read(1)
        h_full, w_full = fbfm_arr.shape

    with rasterio.open(tiles_dir / "canopy_cover.tif") as src:
        cc_arr = src.read(1).astype(np.float32)

    with rasterio.open(tiles_dir / "elevation.tif") as src:
        elev_arr = src.read(1).astype(np.float32)

    # Pulga ignition point is around row 535, col 869
    # Center 256x256 around row 535, col 869
    r_center, c_center = 535, 869
    r0 = max(0, min(r_center - tile_size // 2, h_full - tile_size))
    r1 = r0 + tile_size
    c0 = max(0, min(c_center - tile_size // 2, w_full - tile_size))
    c1 = c0 + tile_size

    # Crop
    fbfm_crop = fbfm_arr[r0:r1, c0:c1]
    cc_crop = cc_arr[r0:r1, c0:c1]
    elev_crop = elev_arr[r0:r1, c0:c1]

    # Map p_veg
    p_veg_crop = np.vectorize(lambda code: FBFM40_TO_P_VEG.get(int(code), -1.0))(fbfm_crop).astype(np.float32)

    # Map p_den from canopy cover (density factor between -0.2 and +0.2)
    p_den_crop = np.clip((cc_crop / 100.0) * 0.4 - 0.2, -0.2, 0.2).astype(np.float32)

    # Compute 3x3 directional slope angles in degrees from Elevation
    res = 30.0
    slope_3x3 = np.zeros((tile_size, tile_size, 3, 3), dtype=np.float32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            dist = np.sqrt(dr**2 + dc**2) * res
            elev_shifted = np.roll(np.roll(elev_crop, -dr, axis=0), -dc, axis=1)
            dz = elev_shifted - elev_crop
            angle_deg = np.degrees(np.arctan(dz / dist))
            slope_3x3[:, :, dr + 1, dc + 1] = angle_deg

    # Ignition tensor (local coordinates in 256x256)
    ign_local_r = r_center - r0
    ign_local_c = c_center - c0
    initial_ignition = np.zeros((tile_size, tile_size), dtype=bool)
    initial_ignition[ign_local_r, ign_local_c] = True

    # Convert to PyTorch tensors
    env_tensors = {
        "p_veg": torch.from_numpy(p_veg_crop).to(device),
        "p_den": torch.from_numpy(p_den_crop).to(device),
        "slope": torch.from_numpy(slope_3x3).to(device),
        "elev": torch.from_numpy(elev_crop).to(device),
    }
    ign_tensor = torch.from_numpy(initial_ignition).to(device)

    # Load hourly weather
    weather_df = pd.read_parquet(tiles_dir / "hrrr_weather_hourly.parquet")

    return env_tensors, ign_tensor, weather_df


def benchmark_device(
    device_name: str,
    tiles_dir: Path,
    steps: int = 720,
    seed: int = 42,
    warmup_steps: int = 20,
) -> dict[str, float | str | int]:
    device = torch.device(device_name)
    print(f"\n--- Benchmarking PyTorchFire 256x256 on {device_name.upper()} ({steps} steps) ---")

    env_tensors, ign_tensor, weather_df = prepare_256_tile_tensors(tiles_dir, device, tile_size=256)

    h, w = 256, 256
    wind_vel = torch.full((h, w), float(weather_df.iloc[0]["wind_speed_ms"]), device=device)
    wind_dir = torch.full(
        (h, w),
        float((270.0 - weather_df.iloc[0]["wind_direction_deg"]) % 360.0),
        device=device,
    )

    env_data = {
        "p_veg": env_tensors["p_veg"],
        "p_den": env_tensors["p_den"],
        "wind_velocity": wind_vel,
        "wind_towards_direction": wind_dir,
        "slope": env_tensors["slope"],
        "initial_ignition": ign_tensor,
    }

    params = {
        "a": torch.tensor(0.06, device=device),
        "p_h": torch.tensor(0.58, device=device),
        "c_1": torch.tensor(0.045, device=device),
        "c_2": torch.tensor(0.15, device=device),
        "p_continue": torch.tensor(0.3, device=device),
    }

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = WildfireModel(env_data=env_data, params=params).to(device)
    model.eval()
    model._initialize_seed(seed)

    # Warmup
    for _ in range(warmup_steps):
        model.compute()

    # Re-initialize for measured run
    model._initialize_seed(seed)
    if device_name == "mps":
        torch.mps.synchronize()

    t_start = time.perf_counter()
    steps_per_hour = 60

    for step in range(1, steps + 1):
        cur_hour = min(step // steps_per_hour, len(weather_df) - 1)
        w_cur = weather_df.iloc[cur_hour]
        model.wind_velocity.fill_(float(w_cur["wind_speed_ms"]))
        model.wind_towards_direction.fill_(float((270.0 - w_cur["wind_direction_deg"]) % 360.0))
        model.compute()

    if device_name == "mps":
        torch.mps.synchronize()
    total_sec = time.perf_counter() - t_start

    ms_per_step = (total_sec / steps) * 1000.0
    scenarios_per_hour = 3600.0 / total_sec

    # Count final burned cells
    state = model.state
    burned_cells = int(torch.count_nonzero(state[0] | state[1]).item())
    burned_ha = (burned_cells * 900.0) / 10_000.0

    print(
        f"Result for {device_name.upper()}: {total_sec:.3f} s total | "
        f"{ms_per_step:.2f} ms/step | {scenarios_per_hour:.1f} scenarios/hour | "
        f"{burned_cells} cells ({burned_ha:.1f} ha) burned"
    )

    return {
        "device": device_name.upper(),
        "tile_shape": "256x256",
        "steps_12h": steps,
        "wall_clock_sec": total_sec,
        "ms_per_step": ms_per_step,
        "scenarios_per_hour": scenarios_per_hour,
        "burned_cells": burned_cells,
        "burned_ha": burned_ha,
    }


def main() -> None:
    tiles_dir = Path("data/tiles")
    results = []

    # 1. Benchmark CPU
    res_cpu = benchmark_device("cpu", tiles_dir=tiles_dir, steps=720)
    results.append(res_cpu)

    # 2. Benchmark MPS if available
    if torch.backends.mps.is_available():
        res_mps = benchmark_device("mps", tiles_dir=tiles_dir, steps=720)
        results.append(res_mps)
    else:
        print("Apple Silicon MPS not available on this system.")

    df = pd.DataFrame(results)
    out_csv = Path("reports/tile_speed_benchmark.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== 256x256 Tile Speed Benchmark Summary ===")
    print(df.to_string(index=False))
    print(f"\nSaved benchmark results to {out_csv}")


if __name__ == "__main__":
    main()
