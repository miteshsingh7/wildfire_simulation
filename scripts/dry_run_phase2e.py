"""Production end-to-end dry run for Phase 2E.

Runs:
1. Production generator on real tiles (3 scenarios, K=4 seeds each).
2. Device detection and reporting.
3. Save shard in compact format (.npz + metadata.json + metadata.csv).
4. Reload shard with CompactSurrogateDataset, validate_shard, collate batch.
5. Execute 2 optimizer steps of SurrogateTrainer (print finite loss and shapes).
6. Diagnostics:
   - seconds per (tile, seed) and per scenario
   - burned fraction for each seed (min, max)
   - distinct arrival-field hashes across seeds
   - burn_prob histogram
   - fraction censored per quantile
   - monotonicity check q10 <= q50 <= q90 on finite cells
"""

import hashlib
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from wf052.surrogate.dataset import (
    CompactSurrogateDataset,
    calculate_upwind_third_ignition,
    collate_surrogate_batch,
)
from wf052.surrogate.model import WindConditionedUNet
from wf052.surrogate.standin_generator import (
    load_and_verify_frozen_teacher_v2,
    load_fire_rasters_crop,
    save_shard_with_metadata,
    simulate_ensemble_teacher_v2,
)
from wf052.surrogate.trainer import SurrogateTrainer, TrainConfig


def run_phase2e_dry_run():
    print("=" * 70)
    print("PHASE 2E LOCAL END-TO-END DRY RUN")
    print("=" * 70)

    # 1. Device detection
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Active Device: {device} ({device.type.upper()})")

    # 2. Frozen teacher setup
    cfg_path = Path("configs/teacher_v2.yaml")
    cfg, cfg_hash, sim_hash, t_params, e_params = load_and_verify_frozen_teacher_v2(cfg_path)
    print(f"Teacher v2 Config Hash: {cfg_hash}")
    print(f"Canonical Sim Hash:     {sim_hash}")

    # 3. Generate 3 scenarios on real training fires (K=4 seeds each)
    scenarios_meta = [
        {"fire": "caldor", "crop_r": 120, "crop_c": 120, "base_seed": 10001},
        {"fire": "tamarack", "crop_r": 150, "crop_c": 150, "base_seed": 20002},
        {"fire": "bobcat", "crop_r": 100, "crop_c": 100, "base_seed": 30003},
    ]

    generated_samples = []
    scenario_timings = []
    tile_seed_timings = []
    scenario_diagnostics = []

    k_ensemble = 4

    for sc_idx, sc in enumerate(scenarios_meta, start=1):
        fire_name = sc["fire"]
        r0, c0 = sc["crop_r"], sc["crop_c"]
        b_seed = sc["base_seed"]

        print(f"\n--- Scenario {sc_idx}/3: Fire='{fire_name}', origin=({r0}, {c0}), base_seed={b_seed} ---")
        fbfm, elev, canopy, wind = load_fire_rasters_crop(
            fire_name=fire_name,
            tile_size=(256, 256),
            crop_center_row=r0,
            crop_center_col=c0,
        )
        ign_r, ign_c = calculate_upwind_third_ignition(wind, tile_shape=(256, 256))

        # Construct FireDataset60m for teacher v2
        from wf052.surrogate.dataset import (
            SurrogateSample,
            compute_slope_aspect,
            fbfm40_to_fuel_group,
        )
        from wf052.surrogate.standin_generator import build_tile_fire_dataset_60m

        sim_device = torch.device("cpu")  # Cellular / ember simulator runs on CPU
        ds_tile = build_tile_fire_dataset_60m(
            fire_name=fire_name,
            fbfm_arr=fbfm,
            elev_arr=elev,
            canopy_arr=canopy,
            hourly_wind_uv=wind,
            device=sim_device,
            res_m=60.0,
        )

        ign_mask = np.zeros((256, 256), dtype=np.float32)
        ign_mask[max(0, ign_r - 1) : min(256, ign_r + 2), max(0, ign_c - 1) : min(256, ign_c + 2)] = 1.0

        t0_sc = time.perf_counter()
        # Run production teacher ensemble
        (
            burn_prob,
            q10,
            q50,
            q90,
            c10,
            c50,
            c90,
            arr_seeds,
        ) = simulate_ensemble_teacher_v2(
            ds=ds_tile,
            t_params=t_params,
            e_params=e_params,
            ign_mask=ign_mask,
            k_seeds=k_ensemble,
            scenario_seed=b_seed,
            horizon_min=720.0,
        )
        sc_elapsed = time.perf_counter() - t0_sc
        sec_per_seed = sc_elapsed / k_ensemble

        fuel_grp = fbfm40_to_fuel_group(fbfm)
        slope, asp_sin, asp_cos = compute_slope_aspect(elev, resolution_m=60.0)
        elev_z = ((elev - np.mean(elev)) / (np.std(elev) + 1e-5)).astype(np.float32)

        sample = SurrogateSample(
            fire_name=fire_name,
            sample_id=f"dryrun_sample_{sc_idx:02d}",
            fuel_group=fuel_grp,
            slope=slope,
            aspect_sin=asp_sin,
            aspect_cos=asp_cos,
            elevation_z=elev_z,
            canopy_cover=canopy,
            ignition_mask=ign_mask,
            hourly_wind_uv=wind,
            target_arrival_time=q50,
            censor_mask=c50,
            scaffold_tag=f"scaffold_{cfg_hash[:8]}",
            seed=b_seed,
            source="teacher_v2_ensemble",
            baseline_only=False,
            burn_prob=burn_prob,
            arrival_q10=q10,
            arrival_q50=q50,
            arrival_q90=q90,
            censor_mask_q10=c10,
            censor_mask_q50=c50,
            censor_mask_q90=c90,
            k_ensemble=k_ensemble,
            teacher_config_hash=cfg_hash,
            canonical_sim_hash=sim_hash,
            fire_id=fire_name,
            tile_id=f"tile_{fire_name}_{r0}_{c0}",
            crop_origin=(r0, c0),
            ignition_cell=(ign_r, ign_c),
            schema_version="2e",
        )

        bf_seeds = [float(np.mean(arr < 720.0)) for arr in arr_seeds]

        scenario_timings.append(sc_elapsed)
        tile_seed_timings.append(sec_per_seed)
        generated_samples.append(sample)

        # Diagnostics per scenario
        # 1. Distinct hashes
        hashes = set()
        for arr in arr_seeds:
            h = hashlib.sha256(arr.astype(np.float32).tobytes()).hexdigest()
            hashes.add(h)

        # 2. Burned fraction min, max
        bf_min = float(np.min(bf_seeds))
        bf_max = float(np.max(bf_seeds))

        # 3. burn_prob histogram (10 bins in [0, 1])
        bp = sample.burn_prob
        hist_counts, bin_edges = np.histogram(bp, bins=10, range=(0.0, 1.0))

        # 4. Fraction censored per quantile
        frac_censored = [
            float(np.mean(sample.censor_mask_q10)),
            float(np.mean(sample.censor_mask_q50)),
            float(np.mean(sample.censor_mask_q90)),
        ]

        # 5. Monotonicity check: q10 <= q50 <= q90 on finite (uncensored) cells
        finite_mask = (sample.censor_mask_q90 == 0.0)
        q10_sub = sample.arrival_q10[finite_mask]
        q50_sub = sample.arrival_q50[finite_mask]
        q90_sub = sample.arrival_q90[finite_mask]
        mono_q10_q50 = np.all(q10_sub <= q50_sub + 1e-4) if len(q10_sub) > 0 else True
        mono_q50_q90 = np.all(q50_sub <= q90_sub + 1e-4) if len(q50_sub) > 0 else True
        is_monotonic = bool(mono_q10_q50 and mono_q50_q90)

        diag = {
            "scenario": sc_idx,
            "fire": fire_name,
            "seconds_scenario": sc_elapsed,
            "seconds_per_seed": sec_per_seed,
            "bf_min": bf_min,
            "bf_max": bf_max,
            "distinct_hashes": len(hashes),
            "burn_prob_hist": hist_counts.tolist(),
            "frac_censored_q10": frac_censored[0],
            "frac_censored_q50": frac_censored[1],
            "frac_censored_q90": frac_censored[2],
            "is_monotonic": is_monotonic,
        }
        scenario_diagnostics.append(diag)
        print(f"  Elapsed: {sc_elapsed:.3f} s ({sec_per_seed:.3f} s/seed)")
        print(f"  Burned Fraction range: [{bf_min:.4f}, {bf_max:.4f}]")
        print(f"  Distinct arrival hashes across {k_ensemble} seeds: {len(hashes)}/{k_ensemble}")
        print(f"  Monotonicity check (q10 <= q50 <= q90): {is_monotonic}")

    # 4. Save shard in compact format
    out_dir = Path("out/phase2e_dryrun")
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_id = "shard_dryrun"

    print("\nSaving shard in Phase 2E compact format (.npz + metadata.csv + metadata.json)...")
    df_meta = save_shard_with_metadata(
        samples=generated_samples,
        shard_id=shard_id,
        shards_dir=out_dir,
    )
    shard_npz_path = out_dir / f"{shard_id}.npz"
    print(f"Saved: {shard_npz_path} ({shard_npz_path.stat().st_size:,} bytes)")

    # 5. Reload shard with CompactSurrogateDataset and validate_shard
    print("\nValidating shard metadata with SurrogateTrainer.validate_shard...")
    SurrogateTrainer.validate_shard(shard_npz_path, expected_schema_version="2e")
    print("Shard validation passed successfully (schema_version='2e' verified).")

    print("\nLoading dataset via CompactSurrogateDataset...")
    dataset = CompactSurrogateDataset([shard_npz_path])
    print(f"Loaded CompactSurrogateDataset: {len(dataset)} samples.")

    # 6. Build batch and run 2 optimizer steps
    print("\nBuilding batch with collate_surrogate_batch...")
    batch_items = [dataset[0], dataset[1]]
    batch = collate_surrogate_batch(batch_items)
    batch = batch.to(device)

    print(f"Batch spatial_features shape: {tuple(batch.spatial_features.shape)}")
    print(f"Batch wind_series shape:       {tuple(batch.wind_series.shape)}")
    print(f"Batch target_quantiles shape: {tuple(batch.target_quantiles.shape)}")
    print(f"Batch censor_mask shape:      {tuple(batch.censor_mask.shape)}")

    print("\nInitializing WindConditionedUNet and SurrogateTrainer...")
    model = WindConditionedUNet(in_spatial_channels=13, base_channels=32, emb_dim=64).to(device)
    
    # Run 2 real optimizer steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = model

    from wf052.surrogate.losses import SurrogateMultiTaskLoss
    crit = SurrogateMultiTaskLoss(lambda_order=1.0, lambda_bce=0.5)

    print("\nExecuting 2 Optimizer Steps:")
    for step in range(1, 3):
        optimizer.zero_grad()
        out = model(batch.spatial_features, batch.wind_series)
        losses = crit(
            pred_quantiles=out.quantiles,
            pred_burn_logits=out.burn_logits,
            target_arrival=batch.target_arrival,
            censor_mask=batch.censor_mask,
        )
        losses.total_loss.backward()
        optimizer.step()

        print(f"  Step {step}: Total Loss = {losses.total_loss.item():.4f} (finite: {np.isfinite(losses.total_loss.item())})")
        print(f"           Pinball Loss = {losses.pinball_loss.item():.4f}, Order Loss = {losses.ordering_loss.item():.4f}, BCE Loss = {losses.bce_loss.item():.4f}")
        print(f"           Predicted Quantiles Shape:   {tuple(out.quantiles.shape)}")
        print(f"           Predicted Burn Logits Shape: {tuple(out.burn_logits.shape)}")
        assert np.isfinite(losses.total_loss.item()), f"Non-finite loss encountered at step {step}!"

    print("\n" + "=" * 70)
    print("STEP 3 SUMMARY METRICS")
    print("=" * 70)
    print(f"Mean time per scenario:     {np.mean(scenario_timings):.3f} s")
    print(f"Mean time per (tile, seed): {np.mean(tile_seed_timings):.3f} s")
    print("\nPer Scenario Diagnostics:")
    for d in scenario_diagnostics:
        print(f"Scenario {d['scenario']} ({d['fire']}):")
        print(f"  Latency: {d['seconds_scenario']:.3f} s ({d['seconds_per_seed']:.3f} s/seed)")
        print(f"  Burned Fraction [min, max]: [{d['bf_min']:.4f}, {d['bf_max']:.4f}]")
        print(f"  Distinct arrival hashes:    {d['distinct_hashes']}")
        print(f"  Burn prob histogram (10 bins): {d['burn_prob_hist']}")
        print(f"  Fraction censored [q10, q50, q90]: [{d['frac_censored_q10']:.4f}, {d['frac_censored_q50']:.4f}, {d['frac_censored_q90']:.4f}]")
        print(f"  Monotonicity (q10 <= q50 <= q90):  {d['is_monotonic']}")

    return scenario_diagnostics


if __name__ == "__main__":
    run_phase2e_dry_run()
