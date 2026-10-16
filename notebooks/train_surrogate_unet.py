"""Kaggle-ready training script for wind-conditioned U-Net surrogate model.

Can be run directly as a Python script or pasted into a Kaggle Notebook.
Saves model checkpoints to /kaggle/working/checkpoints/ at each epoch.
"""

import sys
from pathlib import Path

import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wf052.surrogate.evaluation import (  # noqa: E402
    evaluate_model_on_samples,
)
from wf052.surrogate.model import WindConditionedUNet  # noqa: E402
from wf052.surrogate.standin_generator import (  # noqa: E402
    SEVEN_TRAINING_FIRES,
    generate_ensemble_samples,
)
from wf052.surrogate.trainer import (  # noqa: E402
    SurrogateTrainer,
    TrainConfig,
    set_reproducible_seed,
    split_leave_fire_out,
)


def main() -> None:
    # 1. Device configuration & reproducibility
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[*] Training on device: {device}")
    set_reproducible_seed(42000)

    # 2. Checkpoint directory: Kaggle working directory or local fallback
    kaggle_dir = Path("/kaggle/working/checkpoints")
    checkpoint_dir = kaggle_dir if Path("/kaggle/working").exists() else Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Checkpoints will be saved to: {checkpoint_dir.resolve()}")

    # 3. Generate or load stand-in ensemble samples across 7 training fires
    # Strictly NEVER uses the Camp Fire domain
    print("[*] Generating stand-in scenario samples (Phase 2A scaffold)...")
    samples = generate_ensemble_samples(
        fire_names=SEVEN_TRAINING_FIRES,
        num_members=4,  # Kaggle smoke/quick run setting (use 20 for full training)
        tile_size=(256, 256),
    )
    print(
        f"[*] Generated {len(samples)} stand-in scenario samples across {len(SEVEN_TRAINING_FIRES)} fires."
    )

    # 4. Leave-fire-out split: hold out 'bobcat' for validation
    held_out_fire = "bobcat"
    train_samples, val_samples = split_leave_fire_out(samples, held_out_fire=held_out_fire)
    print(
        f"[*] Leave-Fire-Out Split: Train={len(train_samples)}, Val ({held_out_fire})={len(val_samples)}"
    )

    # 5. Initialize Model and verify parameter count
    model = WindConditionedUNet(
        in_spatial_channels=13,
        base_channels=32,
        emb_dim=64,
        max_horizon_min=720.0,
    ).to(device)

    param_count = model.count_parameters()
    print(f"[*] WindConditionedUNet parameter count: {param_count:,} (must be between 2M and 6M)")
    assert 2_000_000 <= param_count <= 6_000_000, f"Parameter count {param_count} out of bounds!"

    # 6. Training setup and execution via SurrogateTrainer (AMP and per-epoch checkpointing)
    config = TrainConfig(
        batch_size=4,
        learning_rate=3e-4,
        weight_decay=1e-4,
        epochs=3,
        checkpoint_dir=str(checkpoint_dir),
    )

    trainer = SurrogateTrainer(
        model=model,
        train_samples=train_samples,
        val_samples=val_samples,
        config=config,
        device=device,
    )

    print("\n[*] Starting training loop via SurrogateTrainer (AMP enabled)...")
    history = trainer.fit()
    print(f"[*] Training complete across {len(history)} epochs.")

    # 8. Evaluate fidelity to teacher on held-out validation fire
    print(f"\n[*] Evaluating Fidelity to Teacher on held-out fire '{held_out_fire}'...")
    val_fidelity = evaluate_model_on_samples(
        model=model,
        samples=val_samples,
        device=device,
        batch_size=config.batch_size,
    )

    print("\n" + "=" * 60)
    print("SURROGATE EVALUATION: FIDELITY TO TEACHER REPORT")
    print("=" * 60)
    print(f"Held-out Fire:            {val_fidelity.fire_name}")
    print(f"Uncensored MAE:           {val_fidelity.uncensored_mae_minutes:.2f} minutes")
    print(f"P10-P90 Coverage:         {val_fidelity.p10_p90_coverage * 100.0:.1f}% (target ~80%)")
    for h_name, hm in val_fidelity.horizon_metrics.items():
        print(f"Horizon {h_name:4s}:            F1 = {hm.f1:.3f}, IoU = {hm.iou:.3f}")
    print(f"Inference Latency:        {val_fidelity.inference_latency_ms:.2f} ms/tile")
    print(f"Speedup vs Teacher:       {val_fidelity.speedup_vs_teacher:.1f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
