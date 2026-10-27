"""Step 2: Surrogate Model Inference and Latency Benchmarking.

Evaluates:
- Surrogate model (WindConditionedUNet) inference through ArrivalField adapter (p_thr=0.1).
- Benchmarks latency per 256x256 tile on MPS/CPU (mean, p95, n=100).
- Compares per-tile and full-domain (9 tiles) surrogate latency against Teacher v2 runtime (~4.0s).
- Validates ArrivalField contract conformance via validate_arrival_field.
"""

import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from wf052.routing.contracts import validate_arrival_field
from wf052.surrogate.adapter import TilePrediction, stitch_tiles_to_arrival_field, surrogate_output_to_arrival_field
from wf052.surrogate.dataset import rebuild_spatial_inputs
from wf052.surrogate.model import WindConditionedUNet


def main() -> None:
    print("=" * 80)
    print("STAND-IN WEIGHTS (Phase 3A Smoke Checkpoint)")
    print("STEP 2: SURROGATE INFERENCE & LATENCY BENCHMARKING")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = os.environ.get("CKPT", "checkpoints/best.pt")
    print(f"Loading surrogate model from: {ckpt_path}")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model = WindConditionedUNet(in_spatial_channels=13, base_channels=32, emb_dim=64, max_horizon_min=720.0)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load 13-channel spatial tensor for Camp Fire (focused tile r0=175, c0=50)
    print("\n--- Constructing Focused Tile Spatial Tensor (256x256, 13 channels) ---")
    spatial_tensor = rebuild_spatial_inputs(
        fire_id="camp_fire",
        crop_origin=(175, 50),
        ignition_cell=(232 - 175, 452 - 50),
        fires_base_dir="out/camp_fire_60m",
        tile_shape=(256, 256),
    )
    assert spatial_tensor.shape == (13, 256, 256), f"Unexpected shape {spatial_tensor.shape}"

    # Load HRRR wind vector series (12, 2)
    weather_df = pd.read_parquet("data/tiles/hrrr_weather_hourly.parquet")
    u_vals = weather_df["u10_ms"].values[:12].astype(np.float32)
    v_vals = weather_df["v10_ms"].values[:12].astype(np.float32)
    wind_series = np.column_stack([u_vals, v_vals])

    spatial_batch = spatial_tensor.unsqueeze(0).to(device)
    wind_batch = torch.from_numpy(wind_series).unsqueeze(0).to(device)

    # 1. Warm-up
    print("\nWarming up model (10 iterations)...")
    with torch.no_grad():
        for _ in range(10):
            if device.type == "mps":
                torch.mps.synchronize()
            _ = model(spatial_batch, wind_batch)
            if device.type == "mps":
                torch.mps.synchronize()

    # 2. Benchmark n=100 forward passes
    n_iters = 100
    latencies_ms: list[float] = []
    print(f"Benchmarking inference latency over n={n_iters} iterations...")
    with torch.no_grad():
        for _ in range(n_iters):
            if device.type == "mps":
                torch.mps.synchronize()
            t_start = time.perf_counter()

            out = model(spatial_batch, wind_batch)

            if device.type == "mps":
                torch.mps.synchronize()
            t_end = time.perf_counter()
            latencies_ms.append((t_end - t_start) * 1000.0)

    mean_lat = float(np.mean(latencies_ms))
    p50_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95.0))
    min_lat = float(np.min(latencies_ms))
    max_lat = float(np.max(latencies_ms))

    print(f"\nInference Latency per Tile (256x256, n={n_iters}):")
    print(f"  Mean:   {mean_lat:.2f} ms")
    print(f"  P50:    {p50_lat:.2f} ms")
    print(f"  P95:    {p95_lat:.2f} ms")
    print(f"  Min:    {min_lat:.2f} ms")
    print(f"  Max:    {max_lat:.2f} ms")

    # Full domain 3x3 grid (9 tiles) estimated surrogate latency
    full_domain_surrogate_ms = mean_lat * 9.0
    teacher_v2_domain_ms = 3980.0  # Measured across 5 seeds in Step 4: [3.1s - 5.0s, mean 3.98s]
    speedup = teacher_v2_domain_ms / full_domain_surrogate_ms

    print(f"\nFull-Domain Comparison (Camp Fire Domain):")
    print(f"  Teacher v2 full domain simulation:  {teacher_v2_domain_ms:.1f} ms (~4.0 s)")
    print(f"  Surrogate 9-tile full domain:       {full_domain_surrogate_ms:.2f} ms")
    print(f"  Surrogate single-tile inference:    {mean_lat:.2f} ms")
    print(f"  Speedup (Surrogate vs Teacher v2):  {speedup:.1f}x")

    # 3. Test ArrivalField adapter
    print("\n--- Validating ArrivalField Adapter (p_thr=0.1) ---")
    with torch.no_grad():
        out = model(spatial_batch, wind_batch)
        pred_quantiles = out.quantiles[0].cpu().numpy()  # (3, 256, 256)
        pred_burn_prob = torch.sigmoid(out.burn_logits[0, 0]).cpu().numpy()  # (256, 256)

    # Validate single tile conversion
    x_coords = 607260.0 + (50 + np.arange(256)) * 60.0
    y_coords = 4423470.0 - (175 + np.arange(256)) * 60.0
    af_tile = surrogate_output_to_arrival_field(
        quantiles=pred_quantiles,
        burn_prob=pred_burn_prob,
        x_coords=x_coords,
        y_coords=y_coords,
        p_thr=0.1,
    )
    validate_arrival_field(af_tile)
    unburned_tile_cells = int(np.isinf(af_tile["arrival_q50"].values).sum())
    burned_tile_cells = int(np.isfinite(af_tile["arrival_q50"].values).sum())
    print(f"Single tile ArrivalField validated successfully.")
    print(f"  Burned cells (P >= 0.1): {burned_tile_cells} ({burned_tile_cells/65536.0*100.0:.1f}%)")
    print(f"  Unburned cells (P < 0.1): {unburned_tile_cells} ({unburned_tile_cells/65536.0*100.0:.1f}%)")

    # Save summary report
    res_df = pd.DataFrame([{
        "checkpoint": ckpt_path,
        "weights_status": "STAND-IN WEIGHTS (Phase 3A Smoke)",
        "device": str(device),
        "n_samples": n_iters,
        "tile_lat_mean_ms": round(mean_lat, 2),
        "tile_lat_p50_ms": round(p50_lat, 2),
        "tile_lat_p95_ms": round(p95_lat, 2),
        "tile_lat_min_ms": round(min_lat, 2),
        "tile_lat_max_ms": round(max_lat, 2),
        "surrogate_9tile_full_domain_ms": round(full_domain_surrogate_ms, 2),
        "teacher_v2_domain_ms": round(teacher_v2_domain_ms, 2),
        "speedup_vs_teacher_v2": round(speedup, 1),
    }])
    Path("reports").mkdir(exist_ok=True)
    out_csv = Path("reports/step2_surrogate_inference_latency.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved Step 2 latency report to {out_csv}")


if __name__ == "__main__":
    main()
