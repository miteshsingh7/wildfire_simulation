#!/usr/bin/env python3
"""Phase 3C Step 2: Generator for notebooks/kaggle_today.ipynb.

Creates notebooks/kaggle_today.ipynb with exactly 6 cells:
1. Environment & Asserts
2. Timing benchmarks (3 teacher + 3 train steps)
3. Time-boxed label generation (dry / full)
4. Training via scripts/train_surrogate.py subprocess
5. Val/test per-bin tables vs B0 & B1, wind-shuffle probe, evaluation guards
6. Output listing with SHA-256
"""

import json
from pathlib import Path

CELL_1_CODE = """# Cell 1: Environment, Versions, and Teacher Config Assertions
import hashlib
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

# Print MODE at the very top of cell 1
MODE = os.environ.get("MODE", "dry").lower()
print("=" * 75)
print(f"KAGGLE RUN MODE: {MODE.upper()}")
print("=" * 75)

# Setup paths for Kaggle vs local execution
if Path("/kaggle/input/wf052-code").exists():
    sys.path.insert(0, "/kaggle/input/wf052-code/src")
    sys.path.insert(0, "/kaggle/input/wf052-code/scripts")
    cfg_base = Path("/kaggle/input/wf052-code/configs")
    commit_file = Path("/kaggle/input/wf052-code/GIT_COMMIT.txt")
else:
    sys.path.insert(0, str(Path("src").resolve()))
    sys.path.insert(0, str(Path("scripts").resolve()))
    cfg_base = Path("configs")
    commit_file = Path("GIT_COMMIT.txt")

# Print GIT_COMMIT
print("\\n--- GIT_COMMIT ---")
if commit_file.exists():
    print(commit_file.read_text(encoding="utf-8").strip())
else:
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        s = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
        print(f"commit: {c}\\ndirty: {bool(s)}")
    except Exception as e:
        print(f"GIT_COMMIT check failed: {e}")

# Hardware / GPU info
print("\\n--- Hardware & Accelerator ---")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except Exception:
        pass
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("Apple Silicon Metal (MPS)")
    try:
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        print(f"Hardware: {brand}")
    except Exception:
        pass
else:
    print("CPU compute")

# Versions
print(f"Python:   {sys.version.split()[0]}")
print(f"PyTorch:  {torch.__version__}")
print(f"NumPy:    {np.__version__}")

# Import pytorchfire
try:
    import pytorchfire
    print(f"pytorchfire: imported successfully ({getattr(pytorchfire, '__file__', 'in-tree')})")
except ImportError as e:
    raise ImportError(f"CRITICAL: Failed to import pytorchfire: {e}") from e

# Teacher config SHA-256 assert
from wf052.surrogate.standin_generator import EXPECTED_TEACHER_CONFIG_HASH
teacher_cfg_path = cfg_base / "teacher_v2.yaml"
if not teacher_cfg_path.exists():
    teacher_cfg_path = Path("configs/teacher_v2.yaml")
assert teacher_cfg_path.exists(), f"Missing teacher_v2.yaml at {teacher_cfg_path}"
teacher_cfg_bytes = teacher_cfg_path.read_bytes()
computed_hash = hashlib.sha256(teacher_cfg_bytes).hexdigest()
print(f"teacher_v2.yaml SHA-256: {computed_hash}")
assert computed_hash == EXPECTED_TEACHER_CONFIG_HASH, (
    f"Teacher config hash mismatch: {computed_hash} != {EXPECTED_TEACHER_CONFIG_HASH}"
)

# Fail-fast check on dataset paths
tiles_dir_env = os.environ.get("TILES_DIR", "")
if tiles_dir_env and Path(tiles_dir_env).exists():
    tiles_base = Path(tiles_dir_env)
elif Path("/kaggle/input/wf052-tiles-train/tiles/fires").exists():
    tiles_base = Path("/kaggle/input/wf052-tiles-train/tiles/fires")
elif Path("data/tiles/fires").exists():
    tiles_base = Path("data/tiles/fires")
else:
    raise FileNotFoundError("CRITICAL: Training tiles directory not found at any expected path!")

required_fires = ["bobcat", "caldor", "creek", "glass", "kincade", "tamarack", "zogg"]
for rf in required_fires:
    f_dir = tiles_base / rf
    if not f_dir.exists():
        raise FileNotFoundError(f"CRITICAL: Required training fire '{rf}' missing in tiles directory: {f_dir}")

print(f"Dataset path verified: {tiles_base.resolve()} (all 7 fires present)")
"""

CELL_2_CODE = """# Cell 2: Timing Benchmarks (3 K=1 Teacher Runs and 3 Train Steps)
import time
import numpy as np
import torch
import torch.nn as nn
from wf052.surrogate.dataset import calculate_upwind_third_ignition
from wf052.surrogate.model import WindConditionedUNet
from wf052.surrogate.standin_generator import (
    build_tile_fire_dataset_60m,
    load_and_verify_frozen_teacher_v2,
    load_fire_rasters_crop,
    simulate_ensemble_teacher_v2,
)

print("=" * 75)
print("CELL 2: TIMING BENCHMARKS")
print("=" * 75)

# Resolve device and name
if torch.cuda.is_available():
    device = torch.device("cuda")
    dev_name = torch.cuda.get_device_name(0)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    dev_name = "Apple Silicon MPS"
else:
    device = torch.device("cpu")
    dev_name = "CPU"
print(f"Benchmark Device / GPU: {dev_name}")

# Benchmark 1: 3 K=1 Teacher Runs
cfg_dict, cfg_hash, sim_hash, t_params, e_params = load_and_verify_frozen_teacher_v2(teacher_cfg_path)
fbfm, elev, canopy, wind = load_fire_rasters_crop("caldor", tile_size=(256, 256), fires_base_dir=tiles_base)
ign_r, ign_c = calculate_upwind_third_ignition(wind, tile_shape=(256, 256))
ign_mask = np.zeros((256, 256), dtype=bool)
ign_mask[ign_r, ign_c] = True

ds_tile = build_tile_fire_dataset_60m("caldor", fbfm, elev, canopy, wind, device=device)

teacher_times = []
for i in range(3):
    t0 = time.perf_counter()
    _ = simulate_ensemble_teacher_v2(
        ds=ds_tile,
        t_params=t_params,
        e_params=e_params,
        ign_mask=ign_mask,
        k_seeds=1,
        scenario_seed=1000 + i,
    )
    dt = time.perf_counter() - t0
    teacher_times.append(dt)
    print(f"  Teacher Run {i+1}/3 (K=1): {dt:.3f} s")

# Benchmark 2: 3 Model Train Steps
model = WindConditionedUNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.MSELoss()

# Dummy batch: B=8, 13 channels spatial, 12x2 wind
dummy_spatial = torch.randn(8, 13, 256, 256, device=device)
dummy_wind = torch.randn(8, 12, 2, device=device)
dummy_t_q = torch.randn(8, 3, 256, 256, device=device)
dummy_t_bp = torch.rand(8, 1, 256, 256, device=device)

train_step_times = []
for i in range(3):
    t0 = time.perf_counter()
    optimizer.zero_grad()
    out = model(dummy_spatial, dummy_wind)
    loss = criterion(out.quantiles, dummy_t_q) + criterion(torch.sigmoid(out.burn_logits), dummy_t_bp)
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    train_step_times.append(dt)
    print(f"  Train Step {i+1}/3 (B=8):   {dt * 1000:.2f} ms ({dt:.4f} s)")

print(f"\\nTiming Benchmark Summary ({dev_name}):")
print(f"  Teacher K=1 Runs (mean):   {float(np.mean(teacher_times)):.3f} s")
print(f"  Model Train Steps (mean):  {float(np.mean(train_step_times)) * 1000:.2f} ms")
"""

CELL_3_CODE = """# Cell 3: Time-Boxed Label Generation
import os
import time
from pathlib import Path
import numpy as np
from wf052.surrogate.standin_generator import (
    generate_stratified_samples_rejection,
    save_shard_with_metadata,
)

print("=" * 75)
print(f"CELL 3: TIME-BOXED LABEL GENERATION (MODE={MODE})")
print("=" * 75)

# Working directory for shards
if Path("/kaggle/working").exists():
    shards_dir = Path("/kaggle/working/shards")
    ckpt_dir = Path("/kaggle/working/checkpoints")
    rep_dir = Path("/kaggle/working/reports")
else:
    shards_dir = Path(os.environ.get("SHARDS_DIR", "data/shards_kaggle_today"))
    ckpt_dir = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
    rep_dir = Path(os.environ.get("REPORTS_DIR", "reports"))
shards_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)
rep_dir.mkdir(parents=True, exist_ok=True)

# Configuration by MODE
if MODE == "dry":
    scenarios_per_fire = int(os.environ.get("SCENARIOS_PER_FIRE", 2))
    k_ensemble = int(os.environ.get("K_ENSEMBLE", 2))
    stall_cap = int(os.environ.get("STALL_CAP", 20))
    label_max_min = float(os.environ.get("LABEL_MAX_MINUTES", 6.0))
    train_epochs = int(os.environ.get("EPOCHS", 1))
    train_max_min = float(os.environ.get("TRAIN_MAX_MINUTES", 5.0))
    batch_size = int(os.environ.get("BATCH", 8))
    target_quotas = [2, 2, 2, 2, 2]
else:  # full mode
    scenarios_per_fire = int(os.environ.get("SCENARIOS_PER_FIRE", 30))
    k_ensemble = int(os.environ.get("K_ENSEMBLE", 16))
    stall_cap = int(os.environ.get("STALL_CAP", 100))
    label_max_min = float(os.environ.get("LABEL_MAX_MINUTES", 75.0))
    train_epochs = int(os.environ.get("EPOCHS", 60))
    train_max_min = float(os.environ.get("TRAIN_MAX_MINUTES", 45.0))
    batch_size = int(os.environ.get("BATCH", 8))
    target_quotas = [3, 6, 8, 8, 6]

print("Effective Config:")
print(f"  Quotas per fire per bin:  {target_quotas}")
print(f"  K (ensemble seeds):       {k_ensemble}")
print(f"  STALL_CAP:                {stall_cap}")
print(f"  Label MAX_MINUTES:        {label_max_min:.1f} min")
print(f"  Train MAX_MINUTES:        {train_max_min:.1f} min")
print(f"  EPOCHS:                   {train_epochs}")
print(f"  BATCH:                    {batch_size}")
print(f"  Shards Output Directory:  {shards_dir.resolve()}")

fires_list = ["bobcat", "caldor", "creek", "glass", "kincade", "tamarack", "zogg"]
max_wall_sec = label_max_min * 60.0

t_label_start = time.perf_counter()
samples, report = generate_stratified_samples_rejection(
    fire_names=fires_list,
    scenarios_per_fire=scenarios_per_fire,
    k_ensemble=k_ensemble,
    max_wall_sec=max_wall_sec,
    stall_cap=stall_cap,
    fires_base_dir=tiles_base,
    teacher_config_path=teacher_cfg_path,
    device=device,
)
t_label_dur = time.perf_counter() - t_label_start
print(f"\\nLabel generation completed in {t_label_dur:.1f} s ({t_label_dur / 60.0:.2f} min). Accepted samples: {len(samples)}")

# Save accepted samples to shards
samples_per_shard = 200
n_shards = max(1, int(np.ceil(len(samples) / samples_per_shard)))
for s_idx in range(n_shards):
    shard_samples = samples[s_idx * samples_per_shard : (s_idx + 1) * samples_per_shard]
    if shard_samples:
        s_id = f"shard_{s_idx:03d}"
        save_shard_with_metadata(shard_samples, s_id, shards_dir)
        print(f"Saved {len(shard_samples)} samples to {shards_dir / f'{s_id}.npz'}")
"""

CELL_4_CODE = """# Cell 4: Training via scripts/train_surrogate.py Subprocess
import os
import subprocess
import sys
import time
from pathlib import Path

print("=" * 75)
print("CELL 4: SURROGATE TRAINING VIA SUBPROCESS")
print("=" * 75)

# Locate train_surrogate.py
train_script = Path("scripts/train_surrogate.py")
if not train_script.exists() and Path("/kaggle/input/wf052-code/scripts/train_surrogate.py").exists():
    train_script = Path("/kaggle/input/wf052-code/scripts/train_surrogate.py")
assert train_script.exists(), f"Missing train_surrogate.py at {train_script}"

env = os.environ.copy()
env["SHARDS_DIR"] = str(shards_dir.resolve())
env["TILES_DIR"] = str(tiles_base.resolve())
env["CHECKPOINT_DIR"] = str(ckpt_dir.resolve())
env["REPORTS_DIR"] = str(rep_dir.resolve())
env["EPOCHS"] = str(train_epochs)
env["MAX_MINUTES"] = str(train_max_min)
env["BATCH"] = str(batch_size)
env["SEED"] = "42000"

pythonpath_parts = [str(Path("src").resolve()), str(Path("scripts").resolve())]
if Path("/kaggle/input/wf052-code").exists():
    pythonpath_parts.insert(0, "/kaggle/input/wf052-code/src")
    pythonpath_parts.insert(0, "/kaggle/input/wf052-code/scripts")
env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

print(f"Launching training subprocess: {train_script}")
print(f"  SHARDS_DIR:     {env['SHARDS_DIR']}")
print(f"  CHECKPOINT_DIR: {env['CHECKPOINT_DIR']}")
print(f"  REPORTS_DIR:    {env['REPORTS_DIR']}")
print(f"  EPOCHS:         {env['EPOCHS']}")
print(f"  MAX_MINUTES:    {env['MAX_MINUTES']}")
print(f"  BATCH:          {env['BATCH']}")

t_tr_start = time.perf_counter()
proc = subprocess.run([sys.executable, str(train_script)], env=env, check=True)
t_tr_dur = time.perf_counter() - t_tr_start

print(f"\\nTraining subprocess finished with return code {proc.returncode} in {t_tr_dur:.1f} s ({t_tr_dur / 60.0:.2f} min)")
"""

CELL_5_CODE = """# Cell 5: Val/Test Evaluation, Wind-Shuffle Probe, and Evaluation Guards
import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from wf052.surrogate.dataset import CompactSurrogateDataset, collate_surrogate_batch
from wf052.surrogate.evaluate import (
    aggregate_per_bin_metrics,
    check_evaluation_guards,
    evaluate_model_on_dataset,
)
from wf052.surrogate.model import WindConditionedUNet

print("=" * 75)
print("CELL 5: EVALUATION AUDIT & WIND-SHUFFLE PROBE")
print("=" * 75)

best_pt_path = ckpt_dir / "best.pt"
last_pt_path = ckpt_dir / "last.pt"
target_ckpt = best_pt_path if best_pt_path.exists() else last_pt_path
assert target_ckpt.exists(), f"No checkpoint found at {target_ckpt}!"

# 1. Print held-out test evaluation table
test_eval_csv = rep_dir / "test_eval.csv"
if test_eval_csv.exists():
    df_test_eval = pd.read_csv(test_eval_csv)
    print("\\n--- Held-Out Test Evaluation Summary (from reports/test_eval.csv) ---")
    cols = ["model", "fire", "bin", "bin_name", "n", "flag", "brier", "iou", "mae_q10", "mae_q50", "mae_q90", "false_unburned_share"]
    available_cols = [c for c in cols if c in df_test_eval.columns]
    print(df_test_eval[available_cols].to_string(index=False))

# 2. Load dataset and model for validation audit
ds = CompactSurrogateDataset(shard_paths=shards_dir, fires_base_dir=tiles_base)
val_indices = [i for i, f in enumerate(ds.fire_ids) if f == "bobcat"]

model = WindConditionedUNet().to(device)
ckpt = torch.load(target_ckpt, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Val evaluation table
df_val_samples = evaluate_model_on_dataset(model, ds, target_indices=val_indices, device=device)
df_val_bins = aggregate_per_bin_metrics(df_val_samples)
print("\\n--- Per-Bin Val Fidelity Table (bobcat) ---")
print(df_val_bins.to_string(index=False))

# Evaluation guards check (prints loud WARNING if guards fail)
guards_warnings = check_evaluation_guards(df_val_samples)

# 3. Wind-Shuffle Probe on trained model on validation split
print("\\n--- Wind-Shuffle Probe on Trained Model (Val Split) ---")
val_subset = Subset(ds, val_indices)
val_loader = DataLoader(
    val_subset,
    batch_size=len(val_subset),
    shuffle=False,
    collate_fn=collate_surrogate_batch,
)

for batch in val_loader:
    spatial = batch.spatial_features.to(device)
    wind = batch.wind_series.to(device)
    target_bp = batch.burn_prob.to(device)
    target_mask = (target_bp >= 0.5)

    with torch.no_grad():
        out_orig = model(spatial, wind)
        p_orig = torch.sigmoid(out_orig.burn_logits).squeeze(1)
        q50_orig = out_orig.quantiles[:, 1, :, :]

        # Shuffle wind across samples in the batch
        wind_shuffled = torch.roll(wind, shifts=1, dims=0)
        out_shuf = model(spatial, wind_shuffled)
        p_shuf = torch.sigmoid(out_shuf.burn_logits).squeeze(1)
        q50_shuf = out_shuf.quantiles[:, 1, :, :]

        n_target_cells = int(target_mask.sum().item())
        if n_target_cells > 0:
            delta_p = torch.abs(p_shuf - p_orig)[target_mask]
            delta_q50 = torch.abs(q50_shuf - q50_orig)[target_mask]
            mean_abs_p = float(delta_p.mean().item())
            mean_abs_q50 = float(delta_q50.mean().item())
        else:
            mean_abs_p = float("nan")
            mean_abs_q50 = float("nan")

        print(f"Validation target >= 0.5 cell count: {n_target_cells}")
        print(f"Mean Absolute Change in burn_prob:   {mean_abs_p:.5f}")
        print(f"Mean Absolute Change in q50:         {mean_abs_q50:.2f} min")
"""

CELL_6_CODE = """# Cell 6: Output Artifact Listing and SHA-256 Checksums
import hashlib
from pathlib import Path

print("=" * 75)
print("CELL 6: OUTPUT ARTIFACT LISTING & SHA-256 CHECKSUMS")
print("=" * 75)

search_dirs = [rep_dir, ckpt_dir, shards_dir]
found_files = []
for sd in search_dirs:
    if sd.exists():
        for p in sorted(sd.rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                found_files.append(p)

print(f"{'Path':<55} {'Size (KB)':<12} {'SHA-256 Checksum':<64}")
print("-" * 135)
for p in found_files:
    data = p.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    sz_kb = len(data) / 1024.0
    print(f"{str(p):<55} {sz_kb:<12.1f} {sha:<64}")
print("=" * 135)
"""


def make_cell(source: str, cell_type: str = "code") -> dict:
    lines = [line + "\n" for line in source.splitlines()]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return {
        "cell_type": cell_type,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }


def main() -> None:
    nb_path = Path("notebooks/kaggle_today.ipynb")
    nb_path.parent.mkdir(parents=True, exist_ok=True)

    cells = [
        make_cell(CELL_1_CODE),
        make_cell(CELL_2_CODE),
        make_cell(CELL_3_CODE),
        make_cell(CELL_4_CODE),
        make_cell(CELL_5_CODE),
        make_cell(CELL_6_CODE),
    ]

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=1)

    print(f"Created {nb_path} with exactly {len(cells)} cells.")


if __name__ == "__main__":
    main()
