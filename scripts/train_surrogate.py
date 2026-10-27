#!/usr/bin/env python3
"""Phase 3A Step 3: Standalone Surrogate Training Pipeline & Kaggle Trainer.

Features:
- Configurable via environment variables (SHARDS_DIR, TILES_DIR, MAX_MINUTES, EPOCHS, BATCH, LR, SEED, RESUME, AMP).
- Full audit prints at start: GIT_COMMIT, device, GPU name, parameter count (must be 4,094,852),
  shard counts per fire and bin_kmean, label hashes.
- Atomic checkpointing: saves last.pt every epoch; best.pt on val total loss.
- Resumable: RESUME=1 restores model, optimizer, scheduler, epoch, RNG state, best val.
- Early stopping patience: 6 epochs.
- Per-bin evaluation via wf052.surrogate.evaluate on validation fire ('bobcat').
- Post-training test evaluation on held-out test fires ('caldor', 'tamarack') against Baselines B0 & B1.
"""

import math
import os
import random
import shutil
import subprocess
import sys
import time
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from wf052.surrogate.dataset import (
    CompactSurrogateDataset,
    collate_surrogate_batch,
)
from wf052.surrogate.evaluate import (
    aggregate_per_bin_and_fire_metrics,
    aggregate_per_bin_metrics,
    check_evaluation_guards,
    evaluate_baseline_on_dataset,
    evaluate_model_on_dataset,
    fit_baseline_b0,
    fit_baseline_b1,
)
from wf052.surrogate.losses import LossBreakdown, SurrogateMultiTaskLoss
from wf052.surrogate.model import WindConditionedUNet
from wf052.surrogate.standin_generator import (
    EXPECTED_CANONICAL_SIM_HASH,
    EXPECTED_TEACHER_CONFIG_HASH,
)


def get_git_commit_info() -> str:
    """Read commit hash and dirty status."""
    for p in [Path("GIT_COMMIT.txt"), Path("/kaggle/input/wf052-code/GIT_COMMIT.txt")]:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        stat = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        commit = res.stdout.strip()
        dirty = bool(stat.stdout.strip())
        return f"commit: {commit}\ndirty: {dirty}"
    except Exception:
        return "commit: unknown\ndirty: unknown"


def get_device_info() -> tuple[torch.device, str]:
    """Resolve compute device and hardware name."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        # On macOS MPS
        try:
            res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            name = res.stdout.strip() or "Apple Silicon (MPS)"
        except Exception:
            name = "Apple Silicon (MPS)"
    else:
        dev = torch.device("cpu")
        name = "CPU"
    return dev, name


def set_seed(seed: int = 42000) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_gpu_memory_mb(device: torch.device) -> float:
    """Get peak or current GPU memory in MB."""
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return float(torch.mps.current_allocated_memory() / (1024 * 1024))
    return 0.0


def main() -> None:
    print("=" * 75)
    print("SURROGATE TRAINING RUN (Phase 3A)")
    print("=" * 75)

    # 1. Environment and configuration
    git_info = get_git_commit_info()
    device, gpu_name = get_device_info()

    shards_dir_env = os.environ.get("SHARDS_DIR", "")
    if shards_dir_env:
        shards_dir = Path(shards_dir_env)
    elif Path("/kaggle/input/wf052-shards").exists():
        shards_dir = Path("/kaggle/input/wf052-shards")
    else:
        shards_dir = Path("data/shards_smoke_2i")

    tiles_dir_env = os.environ.get("TILES_DIR", "")
    if tiles_dir_env:
        tiles_dir = Path(tiles_dir_env)
    elif Path("/kaggle/input/wf052-tiles-train/tiles/fires").exists():
        tiles_dir = Path("/kaggle/input/wf052-tiles-train/tiles/fires")
    else:
        tiles_dir = Path("data/tiles/fires")

    max_minutes = float(os.environ.get("MAX_MINUTES", 240.0))
    epochs = int(os.environ.get("EPOCHS", 5))
    batch_size = int(os.environ.get("BATCH", 8))
    lr = float(os.environ.get("LR", 3e-4))
    weight_decay = 1e-4
    seed = int(os.environ.get("SEED", 42000))
    resume = int(os.environ.get("RESUME", 0)) == 1
    amp = int(os.environ.get("AMP", 0)) == 1
    ckpt_dir = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path(os.environ.get("REPORTS_DIR", "reports"))
    rep_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    # Instantiate model & verify parameter count
    model = WindConditionedUNet().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"GIT_COMMIT:\n{git_info}")
    print(f"Device:               {device}")
    print(f"GPU Name:             {gpu_name}")
    print(f"Parameter Count:      {param_count:,} (must be 4,094,852)")
    assert param_count == 4094852, f"Expected exactly 4,094,852 params, got {param_count}!"

    print(f"Shards Directory:     {shards_dir.resolve()}")
    print(f"Tiles Directory:      {tiles_dir.resolve()}")
    print(f"Hyperparameters:      epochs={epochs}, batch={batch_size}, lr={lr}, seed={seed}, amp={amp}, resume={resume}")
    print(f"Max Minutes:          {max_minutes:.1f} min")
    print(f"Expected Hashes:      cfg={EXPECTED_TEACHER_CONFIG_HASH[:16]}..., sim={EXPECTED_CANONICAL_SIM_HASH[:16]}...")

    # Load dataset
    ds = CompactSurrogateDataset(shard_paths=shards_dir, fires_base_dir=tiles_dir)
    print(f"Total Shard Samples:  {len(ds)}")

    # Verify label hashes on shards
    for sp in ds.shard_paths:
        npz = np.load(sp, allow_pickle=True)
        h_cfg = str(npz["teacher_config_hash"][0]) if "teacher_config_hash" in npz else ""
        h_sim = str(npz["canonical_sim_hash"][0]) if "canonical_sim_hash" in npz else ""
        assert h_cfg == EXPECTED_TEACHER_CONFIG_HASH, f"Hash mismatch in {sp}: {h_cfg} != {EXPECTED_TEACHER_CONFIG_HASH}"
        assert h_sim == EXPECTED_CANONICAL_SIM_HASH, f"Hash mismatch in {sp}: {h_sim} != {EXPECTED_CANONICAL_SIM_HASH}"

    # Splits from scenarios.yaml
    train_fires = ["kincade", "glass", "zogg", "creek"]
    val_fires = ["bobcat"]
    test_fires = ["caldor", "tamarack"]

    # Strict check: No fire in more than one split
    assert set(train_fires).isdisjoint(set(val_fires)), "Data leakage: train and val share fires!"
    assert set(train_fires).isdisjoint(set(test_fires)), "Data leakage: train and test share fires!"
    assert set(val_fires).isdisjoint(set(test_fires)), "Data leakage: val and test share fires!"

    train_indices = [i for i, f in enumerate(ds.fire_ids) if f in train_fires]
    val_indices = [i for i, f in enumerate(ds.fire_ids) if f in val_fires]
    test_indices = [i for i, f in enumerate(ds.fire_ids) if f in test_fires]

    # Print shard counts per fire and per bin_kmean
    counts_table = []
    for idx, f in enumerate(ds.fire_ids):
        t_bp = ds[idx]["burn_prob"].numpy()
        mean_bf = float(np.mean(t_bp))
        bin_k = int(np.digitize(mean_bf, [0.002, 0.02, 0.1, 0.5]))
        counts_table.append({"fire": f, "bin_kmean": bin_k})
    df_counts = pd.DataFrame(counts_table)

    print("\n--- Shard Counts per Fire x Bin ---")
    ct = pd.crosstab(df_counts["fire"], df_counts["bin_kmean"], margins=True)
    print(ct)

    print(f"\nSplit Sizes: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    # Phase 3C Step 1: Base-Rate Initialization from TRAIN shards only
    total_bp_sum = 0.0
    total_cells = 0
    for idx in train_indices:
        bp = ds[idx]["burn_prob"].numpy()
        total_bp_sum += float(np.sum(bp))
        total_cells += bp.size
    mean_train_bp = float(total_bp_sum / max(1, total_cells))
    init_bias = model.init_burn_prob_bias(mean_train_bp)
    param_count_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count_after == 4094852, f"Expected 4,094,852 params, got {param_count_after}"

    print(f"\n--- Phase 3C Step 1: Base-Rate Head Initialization ---")
    print(f"  Mean burn_prob over TRAIN shards: {mean_train_bp:.6f}")
    print(f"  Head bias initialized to:         {init_bias:+.6f}")
    print(f"  Parameter count (must be 4,094,852): {param_count_after:,}")

    # Dataloaders
    train_subset = Subset(ds, train_indices)
    val_subset = Subset(ds, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_surrogate_batch,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_surrogate_batch,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)
    loss_fn = SurrogateMultiTaskLoss(
        quantiles=(0.1, 0.5, 0.9),
        lambda_order=1.0,
        lambda_bce=0.5,
        horizon_scale=720.0,
    )

    start_epoch = 0
    best_val_loss = float("inf")
    patience = 6
    patience_counter = 0

    # Resuming logic
    last_pt_path = ckpt_dir / "last.pt"
    best_pt_path = ckpt_dir / "best.pt"

    if resume and last_pt_path.exists():
        print(f"\n[RESUME] Restoring checkpoint from {last_pt_path} ...")
        ckpt = torch.load(last_pt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        rng = ckpt.get("rng_state", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng and rng["torch"] is not None:
            torch.set_rng_state(rng["torch"].cpu().byte())
        if rng.get("mps") is not None and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.set_rng_state(rng["mps"].cpu().byte())
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu().byte() for s in rng["cuda"]])
        print(f"[RESUME] Resuming at Epoch {start_epoch} (Best Val Loss: {best_val_loss:.5f})")

    # Fit baselines on train split for later comparison
    print("\nFitting Baselines B0 & B1 on training fires ...")
    b0 = fit_baseline_b0(ds, train_fires=train_fires)
    b1 = fit_baseline_b1(ds, train_fires=train_fires)
    print("Baselines fitted.")

    # Logging records
    train_log_path = rep_dir / "train_log.csv"
    train_history: list[dict[str, Any]] = []

    t_train_start = time.perf_counter()
    step_times = []
    global_step = 0

    print("\n" + "=" * 75)
    print(f"STARTING TRAINING LOOP ({epochs} Epochs)")
    print("=" * 75)

    for epoch in range(start_epoch, epochs):
        t_ep_start = time.perf_counter()
        if (time.perf_counter() - t_train_start) / 60.0 >= max_minutes:
            print(f"Max minutes cap ({max_minutes:.1f} min) reached. Stopping training.")
            break

        model.train()
        train_loss_accum = {"total": 0.0, "pinball": 0.0, "q10": 0.0, "q50": 0.0, "q90": 0.0, "order": 0.0, "bce": 0.0}
        n_train_batches = 0

        for batch in train_loader:
            t_s0 = time.perf_counter()

            spatial = batch.spatial_features.to(device)
            wind = batch.wind_series.to(device)
            t_quantiles = batch.target_quantiles.to(device)
            c_quantiles = batch.censor_quantiles.to(device)
            burn_prob = batch.burn_prob.to(device)

            optimizer.zero_grad()

            if amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    out = model(spatial, wind)
                    lb = loss_fn(out.quantiles, out.burn_logits, target_quantiles=t_quantiles, censor_quantiles=c_quantiles, burn_prob=burn_prob)
            else:
                out = model(spatial, wind)
                lb = loss_fn(out.quantiles, out.burn_logits, target_quantiles=t_quantiles, censor_quantiles=c_quantiles, burn_prob=burn_prob)

            lb.total_loss.backward()
            optimizer.step()

            t_step = time.perf_counter() - t_s0
            if len(step_times) < 50:
                step_times.append(t_step)

            train_loss_accum["total"] += float(lb.total_loss.item())
            train_loss_accum["pinball"] += float(lb.pinball_loss.item())
            train_loss_accum["q10"] += float(lb.loss_q10.item())
            train_loss_accum["q50"] += float(lb.loss_q50.item())
            train_loss_accum["q90"] += float(lb.loss_q90.item())
            train_loss_accum["order"] += float(lb.ordering_loss.item())
            train_loss_accum["bce"] += float(lb.bce_loss.item())
            n_train_batches += 1
            global_step += 1

        scheduler.step()

        # Mean train losses
        mean_tr = {k: v / max(1, n_train_batches) for k, v in train_loss_accum.items()}

        # Evaluate on validation split ('bobcat')
        model.eval()
        val_loss_accum = {"total": 0.0, "pinball": 0.0, "q10": 0.0, "q50": 0.0, "q90": 0.0, "order": 0.0, "bce": 0.0}
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                spatial = batch.spatial_features.to(device)
                wind = batch.wind_series.to(device)
                t_quantiles = batch.target_quantiles.to(device)
                c_quantiles = batch.censor_quantiles.to(device)
                burn_prob = batch.burn_prob.to(device)

                out = model(spatial, wind)
                lb = loss_fn(out.quantiles, out.burn_logits, target_quantiles=t_quantiles, censor_quantiles=c_quantiles, burn_prob=burn_prob)

                val_loss_accum["total"] += float(lb.total_loss.item())
                val_loss_accum["pinball"] += float(lb.pinball_loss.item())
                val_loss_accum["q10"] += float(lb.loss_q10.item())
                val_loss_accum["q50"] += float(lb.loss_q50.item())
                val_loss_accum["q90"] += float(lb.loss_q90.item())
                val_loss_accum["order"] += float(lb.ordering_loss.item())
                val_loss_accum["bce"] += float(lb.bce_loss.item())
                n_val_batches += 1

        mean_val = {k: v / max(1, n_val_batches) for k, v in val_loss_accum.items()}

        # Evaluate per-bin metrics on validation split using evaluate.py
        df_val_samples = evaluate_model_on_dataset(model, ds, target_indices=val_indices, device=device)
        df_val_bins = aggregate_per_bin_metrics(df_val_samples)

        # Check evaluation guards (Phase 3C Step 4)
        check_evaluation_guards(df_val_samples)

        ep_duration = time.perf_counter() - t_ep_start
        print(
            f"Epoch {epoch:2d}/{epochs} ({ep_duration:.1f}s) | "
            f"Train Total: {mean_tr['total']:.5f} (q10={mean_tr['q10']:.4f}, q50={mean_tr['q50']:.4f}, q90={mean_tr['q90']:.4f}, ord={mean_tr['order']:.4f}, bce={mean_tr['bce']:.4f}) | "
            f"Val Total: {mean_val['total']:.5f} | Val Brier: {df_val_bins['brier'].mean():.4f}"
        )
        print("Per-Bin Val Fidelity Table:")
        print(df_val_bins.to_string(index=False))

        # Atomic checkpoint save: last.pt
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "mps": torch.mps.get_rng_state() if hasattr(torch, "mps") and torch.backends.mps.is_available() else None,
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "rng_state": rng_state,
        }
        tmp_last = ckpt_dir / "last.pt.tmp"
        torch.save(ckpt_data, tmp_last)
        os.replace(tmp_last, last_pt_path)

        # Check for best model
        val_tot = mean_val["total"]
        if val_tot < best_val_loss:
            best_val_loss = val_tot
            patience_counter = 0
            tmp_best = ckpt_dir / "best.pt.tmp"
            torch.save(ckpt_data, tmp_best)
            os.replace(tmp_best, best_pt_path)
            print(f"  --> Saved new best model (val_loss={best_val_loss:.5f}) to {best_pt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

        # Log epoch summary
        log_entry = {
            "epoch": epoch,
            "train_total_loss": mean_tr["total"],
            "train_pinball_loss": mean_tr["pinball"],
            "train_loss_q10": mean_tr["q10"],
            "train_loss_q50": mean_tr["q50"],
            "train_loss_q90": mean_tr["q90"],
            "train_ordering_loss": mean_tr["order"],
            "train_bce_loss": mean_tr["bce"],
            "val_total_loss": mean_val["total"],
            "val_brier": df_val_bins["brier"].mean(),
            "val_iou": df_val_bins["iou"].dropna().mean(),
            "val_mae_q50": df_val_bins["mae_q50"].dropna().mean(),
        }
        train_history.append(log_entry)

    pd.DataFrame(train_history).to_csv(train_log_path, index=False)
    print(f"\nSaved training history to {train_log_path}")

    # Performance stats (first 50 steps)
    if step_times:
        mean_step_sec = float(np.mean(step_times))
        p95_step_sec = float(np.percentile(step_times, 95))
        gpu_mem_mb = get_gpu_memory_mb(device)
        step_lbl = f"all {len(step_times)} steps" if len(step_times) < 30 else f"first {len(step_times)} steps"
        print(f"\nPerformance Benchmark ({step_lbl}):")
        print(f"  Time per iteration (mean): {mean_step_sec * 1000.0:.2f} ms ({mean_step_sec:.4f} s)")
        print(f"  Time per iteration (p95):  {p95_step_sec * 1000.0:.2f} ms ({p95_step_sec:.4f} s)")
        if device.type == "cuda":
            peak_cuda_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            print(f"  CUDA Memory (max_memory_allocated): {peak_cuda_mb:.2f} MB")
        elif device.type == "mps":
            mps_cur_mb = float(torch.mps.current_allocated_memory() / (1024 * 1024)) if hasattr(torch.mps, "current_allocated_memory") else 0.0
            mps_drv_mb = float(torch.mps.driver_allocated_memory() / (1024 * 1024)) if hasattr(torch.mps, "driver_allocated_memory") else 0.0
            print(f"  MPS Current Allocated Memory:       {mps_cur_mb:.2f} MB")
            print(f"  MPS Driver Allocated Memory:        {mps_drv_mb:.2f} MB")
        else:
            print(f"  Peak GPU Memory:           {gpu_mem_mb:.1f} MB")

    # =========================================================================
    # POST-TRAINING TEST EVALUATION (Held-out Test Fires: caldor, tamarack)
    # =========================================================================
    print("\n" + "=" * 75)
    print("FINAL TEST EVALUATION ON TEST FIRES ('caldor', 'tamarack')")
    print("=" * 75)

    if best_pt_path.exists():
        best_ckpt = torch.load(best_pt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {best_ckpt['epoch']}")

    # 1. Evaluate surrogate model on test fires
    df_surrogate_test = evaluate_model_on_dataset(model, ds, target_indices=test_indices, device=device)
    agg_surrogate = aggregate_per_bin_and_fire_metrics(df_surrogate_test, fires=test_fires, model_name="Surrogate (UNet)")

    # 2. Evaluate Baseline B0 on test fires
    df_b0_test = evaluate_baseline_on_dataset(b0, ds, test_indices, baseline_name="B0")
    agg_b0 = aggregate_per_bin_and_fire_metrics(df_b0_test, fires=test_fires, model_name="Baseline B0")

    # 3. Evaluate Baseline B1 on test fires
    df_b1_test = evaluate_baseline_on_dataset(b1, ds, test_indices, baseline_name="B1")
    agg_b1 = aggregate_per_bin_and_fire_metrics(df_b1_test, fires=test_fires, model_name="Baseline B1")

    df_test_final = pd.concat([agg_surrogate, agg_b0, agg_b1], ignore_index=True)
    test_eval_csv = rep_dir / "test_eval.csv"
    df_test_final.to_csv(test_eval_csv, index=False)
    print(f"\n--- Test Evaluation Summary (Saved to {test_eval_csv}) ---")
    cols_to_show = ["model", "fire", "bin", "bin_name", "n", "flag", "brier", "iou", "mae_q10", "mae_q50", "mae_q90", "false_unburned_share"]
    print(df_test_final[cols_to_show].to_string(index=False))

    # Print SHA-256 of all checkpoints
    print("\n--- Checkpoint SHA-256 Checksums ---")
    for cp in [best_pt_path, last_pt_path]:
        if cp.exists():
            h = hashlib.sha256(cp.read_bytes()).hexdigest()
            print(f"  {cp}: {h}")


if __name__ == "__main__":
    main()
