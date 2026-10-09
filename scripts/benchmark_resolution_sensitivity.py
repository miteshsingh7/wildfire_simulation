"""Benchmark resolution sensitivity of PyTorchFire on Camp Fire domain (Step 1).

Compares 30 m, 60 m, and 90 m resolutions:
- Aggregates fuel by dominant class (mode).
- Aggregates terrain (elevation, canopy cover) by mean.
- Computes directional slope angles using resolution distance.
- Runs 12-hour simulation (720 steps at 1 min/step) on Apple Silicon MPS/CPU.
- Records runtime, latency per step, and hourly burned area progression curves.
- Saves reports/resolution_sensitivity.csv and reports/resolution_sensitivity_curves.png.
"""

import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from pytorchfire import WildfireModel

from wf052.sim.pytorchfire_engine import FBFM40_TO_P_VEG


def block_mode(arr: np.ndarray, factor: int) -> np.ndarray:
    """Downsample categorical 2D raster by block mode (dominant class)."""
    h, w = arr.shape
    new_h = h // factor
    new_w = w // factor
    cropped = arr[: new_h * factor, : new_w * factor]
    reshaped = cropped.reshape(new_h, factor, new_w, factor).transpose(0, 2, 1, 3).reshape(new_h, new_w, factor * factor)
    out = np.zeros((new_h, new_w), dtype=arr.dtype)
    for r in range(new_h):
        for c in range(new_w):
            vals = reshaped[r, c]
            counts = Counter(vals)
            out[r, c] = counts.most_common(1)[0][0]
    return out


def block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Downsample continuous 2D raster by block mean."""
    h, w = arr.shape
    new_h = h // factor
    new_w = w // factor
    cropped = arr[: new_h * factor, : new_w * factor]
    reshaped = cropped.reshape(new_h, factor, new_w, factor).transpose(0, 2, 1, 3).reshape(new_h, new_w, factor * factor)
    return reshaped.mean(axis=2).astype(np.float32)


def run_resolution_simulation(
    res_m: int,
    factor: int,
    fbfm_orig: np.ndarray,
    cc_orig: np.ndarray,
    elev_orig: np.ndarray,
    weather_df: pd.DataFrame,
    ign_utm: tuple[float, float],
    bbox_utm: dict[str, float],
    device: torch.device,
    seed: int = 42,
    steps: int = 720,
) -> dict[str, Any]:
    print(f"\n=== Running Resolution {res_m} m (Scale factor: {factor}x) ===")

    # 1. Resample layers
    if factor == 1:
        fbfm_res = fbfm_orig.copy()
        cc_res = cc_orig.copy()
        elev_res = elev_orig.copy()
    else:
        print(f"  Downsampling fuel by dominant class ({factor}x{factor} mode)...")
        fbfm_res = block_mode(fbfm_orig, factor)
        print(f"  Downsampling canopy cover and elevation by mean ({factor}x{factor} mean)...")
        cc_res = block_mean(cc_orig, factor)
        elev_res = block_mean(elev_orig, factor)

    h, w = fbfm_res.shape
    cell_area_ha = (res_m * res_m) / 10_000.0

    # 2. Map fuel to p_veg
    p_veg_arr = np.full((h, w), 0.20, dtype=np.float32)
    for code, p_val in FBFM40_TO_P_VEG.items():
        p_veg_arr[fbfm_res == code] = p_val
    p_veg_arr[fbfm_res <= 0] = -1.0

    # 3. Map canopy cover to p_den
    p_den_arr = np.clip((cc_res / 100.0) * 0.4 - 0.2, -0.2, 0.2).astype(np.float32)

    # 4. Compute 3x3 slope angles using res_m
    slope_3x3 = np.zeros((h, w, 3, 3), dtype=np.float32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            dist = np.sqrt(dr**2 + dc**2) * float(res_m)
            elev_shifted = np.roll(np.roll(elev_res, -dr, axis=0), -dc, axis=1)
            dz = elev_shifted - elev_res
            slope_3x3[:, :, dr + 1, dc + 1] = np.degrees(np.arctan(dz / dist))

    # 5. Map Ignition
    ign_x, ign_y = ign_utm
    ign_col = int(np.clip((ign_x - bbox_utm["x_min"]) / res_m, 0, w - 1))
    ign_row = int(np.clip((bbox_utm["y_max"] - ign_y) / res_m, 0, h - 1))
    initial_ignition = np.zeros((h, w), dtype=bool)
    initial_ignition[ign_row, ign_col] = True

    # Convert to tensors
    env_tensors = {
        "p_veg": torch.from_numpy(p_veg_arr).to(device),
        "p_den": torch.from_numpy(p_den_arr).to(device),
        "slope": torch.from_numpy(slope_3x3).to(device),
        "elev": torch.from_numpy(elev_res).to(device),
    }
    ign_tensor = torch.from_numpy(initial_ignition).to(device)

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

    if device.type == "mps":
        torch.mps.synchronize()

    print(f"  Starting 12h simulation ({steps} steps on {device.type.upper()})...")
    t_start = time.perf_counter()

    hourly_burned_ha = []
    hourly_burned_cells = []
    steps_per_hour = 60

    # Record t=0
    hourly_burned_ha.append(cell_area_ha)
    hourly_burned_cells.append(1)

    for step in range(1, steps + 1):
        cur_hour = min(step // steps_per_hour, len(weather_df) - 1)
        w_cur = weather_df.iloc[cur_hour]
        model.wind_velocity.fill_(float(w_cur["wind_speed_ms"]))
        model.wind_towards_direction.fill_(float((270.0 - w_cur["wind_direction_deg"]) % 360.0))
        model.compute()

        if step % steps_per_hour == 0:
            st = model.state
            b_cnt = int(torch.count_nonzero(st[0] | st[1]).item())
            hourly_burned_cells.append(b_cnt)
            hourly_burned_ha.append(b_cnt * cell_area_ha)

    if device.type == "mps":
        torch.mps.synchronize()
    total_sec = time.perf_counter() - t_start

    ms_per_step = (total_sec / steps) * 1000.0
    final_cells = hourly_burned_cells[-1]
    final_ha = hourly_burned_ha[-1]

    print(
        f"  Completed in {total_sec:.2f}s | Latency: {ms_per_step:.2f} ms/step | "
        f"Final Burned: {final_cells} cells ({final_ha:.1f} ha)"
    )

    return {
        "resolution_m": res_m,
        "grid_shape": f"{h}x{w}",
        "total_cells": h * w,
        "runtime_sec": total_sec,
        "ms_per_step": ms_per_step,
        "final_burned_cells": final_cells,
        "final_burned_ha": final_ha,
        "hourly_burned_ha": hourly_burned_ha,
    }


def main() -> None:
    with open("configs/phase0.yaml") as f:
        cfg = yaml.safe_load(f)

    bbox_utm = cfg["domain"]["bbox_utm"]
    tiles_dir = Path("data/tiles")

    # Load 30m base arrays
    with rasterio.open(tiles_dir / "fbfm40.tif") as src:
        fbfm_30m = src.read(1)
    with rasterio.open(tiles_dir / "canopy_cover.tif") as src:
        cc_30m = src.read(1).astype(np.float32)
    with rasterio.open(tiles_dir / "elevation.tif") as src:
        elev_30m = src.read(1).astype(np.float32)

    weather_df = pd.read_parquet(tiles_dir / "hrrr_weather_hourly.parquet")
    ign_utm = (float(cfg["event"]["ignition"]["utm_x"]), float(cfg["event"]["ignition"]["utm_y"]))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    results = []
    curve_data = {}

    configs_to_run = [
        {"res_m": 30, "factor": 1},
        {"res_m": 60, "factor": 2},
        {"res_m": 90, "factor": 3},
    ]

    for c in configs_to_run:
        res_info = run_resolution_simulation(
            res_m=c["res_m"],
            factor=c["factor"],
            fbfm_orig=fbfm_30m,
            cc_orig=cc_30m,
            elev_orig=elev_30m,
            weather_df=weather_df,
            ign_utm=ign_utm,
            bbox_utm=bbox_utm,
            device=device,
            steps=720,
        )
        curve_data[c["res_m"]] = res_info.pop("hourly_burned_ha")
        results.append(res_info)

    df_summary = pd.DataFrame(results)
    out_csv = Path("reports/resolution_sensitivity.csv")
    df_summary.to_csv(out_csv, index=False)
    print("\n=== Resolution Sensitivity Summary ===")
    print(df_summary.to_string(index=False))
    print(f"\nSaved summary to {out_csv}")

    # Plot Burned Area vs Time Curves
    hours = list(range(13))
    plot_path = Path("reports/resolution_sensitivity_curves.png")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    colors = {30: "tab:red", 60: "tab:blue", 90: "tab:green"}
    markers = {30: "o-", 60: "s-", 90: "^-"}

    for res_m, ha_curve in curve_data.items():
        ax.plot(
            hours,
            ha_curve,
            markers[res_m],
            color=colors[res_m],
            label=f"{res_m} m Grid ({df_summary.loc[df_summary['resolution_m']==res_m, 'grid_shape'].iloc[0]}) - {df_summary.loc[df_summary['resolution_m']==res_m, 'runtime_sec'].iloc[0]:.1f}s",
            linewidth=2,
        )

    # Also plot the observed GeoMAC IR flight point at +11.4h
    ax.scatter([11.4], [22090.3], color="black", s=100, zorder=5, label="Observed GeoMAC IR Flight (22,090 ha @ +11.4h)")

    ax.set_xlabel("Simulation Horizon (Hours)", fontsize=11)
    ax.set_ylabel("Cumulative Burned Area (ha)", fontsize=11)
    ax.set_title("PyTorchFire Resolution Sensitivity: Burned Area vs Time (30m vs 60m vs 90m)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved resolution sensitivity curves to {plot_path}")


if __name__ == "__main__":
    main()
