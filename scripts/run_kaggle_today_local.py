#!/usr/bin/env python3
"""Phase 3C Step 3: Local End-to-End Dry Run of notebooks/kaggle_today.ipynb.

Executes all 6 cells of notebooks/kaggle_today.ipynb sequentially with
Kaggle paths remapped to local environment.
Measures per-stage perf_counter wall times, benchmarks training steps,
and reports MPS / CUDA memory.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path
import torch

def main() -> None:
    print("=" * 75)
    print("PHASE 3C STEP 3: LOCAL END-TO-END DRY RUN")
    print("=" * 75)

    nb_path = Path("notebooks/kaggle_today.ipynb")
    assert nb_path.exists(), f"Missing {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Remap paths and configure dry mode
    os.environ["MODE"] = "dry"
    os.environ["SHARDS_DIR"] = str(Path("data/shards_kaggle_today_dry").resolve())
    os.environ["CHECKPOINT_DIR"] = str(Path("checkpoints/kaggle_today_dry").resolve())
    os.environ["REPORTS_DIR"] = str(Path("reports/kaggle_today_dry").resolve())
    os.environ["TILES_DIR"] = str(Path("data/tiles/fires").resolve())

    # Clean local target dirs for a fresh dry run
    for d in [Path("data/shards_kaggle_today_dry"), Path("checkpoints/kaggle_today_dry"), Path("reports/kaggle_today_dry")]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    cell_globals: dict = {
        "__name__": "__main__",
        "__file__": str(nb_path.resolve()),
    }

    stage_timings: dict[str, float] = {}
    total_t0 = time.perf_counter()

    for idx, cell in enumerate(nb["cells"]):
        cell_num = idx + 1
        cell_code = "".join(cell["source"])
        first_line = cell["source"][0].strip() if cell["source"] else f"Cell {cell_num}"
        stage_name = f"Cell {cell_num}: {first_line.lstrip('# ')}"

        print("\n" + "#" * 75)
        print(f"RUNNING {stage_name}")
        print("#" * 75)

        t_stage_start = time.perf_counter()
        exec(cell_code, cell_globals)
        t_stage_dur = time.perf_counter() - t_stage_start
        stage_timings[f"Cell {cell_num}"] = t_stage_dur
        print(f"\n[COMPLETED {stage_name} in {t_stage_dur:.2f} s]")

    total_dur = time.perf_counter() - total_t0

    # Summary table of wall times
    print("\n" + "=" * 75)
    print("PER-STAGE WALL CLOCK TIMES SUMMARY (perf_counter)")
    print("=" * 75)
    for c_name, dur in stage_timings.items():
        print(f"  {c_name:<15}: {dur:>7.2f} s ({dur / 60.0:>5.2f} min)")
    print("-" * 75)
    print(f"  {'Total Wall Time':<15}: {total_dur:>7.2f} s ({total_dur / 60.0:>5.2f} min)")
    print("=" * 75)

    # Memory report
    print("\n--- Memory Allocation Report ---")
    if torch.cuda.is_available():
        cuda_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"  CUDA torch.cuda.max_memory_allocated: {cuda_peak:.2f} MB")
        print("  Reported Counter: CUDA peak active tensor memory.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps_cur = torch.mps.current_allocated_memory() / (1024 * 1024) if hasattr(torch.mps, "current_allocated_memory") else 0.0
        mps_drv = torch.mps.driver_allocated_memory() / (1024 * 1024) if hasattr(torch.mps, "driver_allocated_memory") else 0.0
        print(f"  MPS torch.mps.current_allocated_memory: {mps_cur:.2f} MB")
        print(f"  MPS torch.mps.driver_allocated_memory:  {mps_drv:.2f} MB")
        print("  Reported Counter: Both reported. current_allocated_memory tracks active tensor buffers; driver_allocated_memory tracks Metal runtime driver heap allocation.")
    else:
        print("  Device: CPU (no GPU memory counter)")


if __name__ == "__main__":
    main()
