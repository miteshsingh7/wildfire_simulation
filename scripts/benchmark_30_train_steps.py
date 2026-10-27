#!/usr/bin/env python3
"""Phase 3C Step 3: 30-Step Model Training Benchmark.

Benchmarks 30 full training steps (forward, multi-task loss, backward, optimizer step)
on the MPS/CUDA accelerator with batch size 8.
Reports mean and p95 iteration wall time, and MPS/CUDA memory allocations.
"""

import time
import numpy as np
import torch
from wf052.surrogate.dataset import CompactSurrogateDataset, collate_surrogate_batch
from wf052.surrogate.losses import SurrogateMultiTaskLoss
from wf052.surrogate.model import WindConditionedUNet
from torch.utils.data import DataLoader, Subset


def main() -> None:
    print("=" * 75)
    print("30-STEP MODEL TRAINING BENCHMARK (Phase 3C Step 3)")
    print("=" * 75)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dev_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        dev_name = "Apple Silicon MPS"
    else:
        device = torch.device("cpu")
        dev_name = "CPU"
    print(f"Device: {dev_name}")

    # Use smoke shards
    ds = CompactSurrogateDataset("data/shards_smoke_2i", "data/tiles/fires")
    train_fires = {"kincade", "glass", "zogg", "creek"}
    train_indices = [i for i, f in enumerate(ds.fire_ids) if f in train_fires]

    loader = DataLoader(
        Subset(ds, train_indices),
        batch_size=8,
        shuffle=True,
        collate_fn=collate_surrogate_batch,
    )

    model = WindConditionedUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = SurrogateMultiTaskLoss()
    model.train()

    # Warmup step
    for batch in loader:
        sp = batch.spatial_features.to(device)
        w = batch.wind_series.to(device)
        t_q = batch.target_quantiles.to(device)
        c_q = batch.censor_quantiles.to(device)
        bp = batch.burn_prob.to(device)
        optimizer.zero_grad()
        out = model(sp, w)
        lb = loss_fn(out.quantiles, out.burn_logits, target_quantiles=t_q, censor_quantiles=c_q, burn_prob=bp)
        lb.total_loss.backward()
        optimizer.step()
        break

    step_times = []
    step_count = 0
    target_steps = 30

    while step_count < target_steps:
        for batch in loader:
            if step_count >= target_steps:
                break
            sp = batch.spatial_features.to(device)
            w = batch.wind_series.to(device)
            t_q = batch.target_quantiles.to(device)
            c_q = batch.censor_quantiles.to(device)
            bp = batch.burn_prob.to(device)

            t0 = time.perf_counter()
            optimizer.zero_grad()
            out = model(sp, w)
            lb = loss_fn(out.quantiles, out.burn_logits, target_quantiles=t_q, censor_quantiles=c_q, burn_prob=bp)
            lb.total_loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            step_times.append(dt)
            step_count += 1

    mean_sec = float(np.mean(step_times))
    p95_sec = float(np.percentile(step_times, 95))

    print(f"\nBenchmark Results ({len(step_times)} steps, B=8):")
    print(f"  Mean step time: {mean_sec * 1000.0:.2f} ms ({mean_sec:.4f} s)")
    print(f"  P95 step time:  {p95_sec * 1000.0:.2f} ms ({p95_sec:.4f} s)")

    print("\nMemory Counters:")
    if device.type == "cuda":
        cuda_mem = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        print(f"  CUDA torch.cuda.max_memory_allocated: {cuda_mem:.2f} MB")
    elif device.type == "mps":
        cur_mem = float(torch.mps.current_allocated_memory() / (1024 * 1024))
        drv_mem = float(torch.mps.driver_allocated_memory() / (1024 * 1024))
        print(f"  MPS torch.mps.current_allocated_memory: {cur_mem:.2f} MB")
        print(f"  MPS torch.mps.driver_allocated_memory:  {drv_mem:.2f} MB")
        print("  Reported Counter: Both reported. current_allocated_memory measures active tensor allocations; driver_allocated_memory measures Metal runtime driver heap allocation.")


if __name__ == "__main__":
    main()
