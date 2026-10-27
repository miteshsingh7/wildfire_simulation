#!/usr/bin/env python3
"""Phase 4B Step 2e: Implementation and Evaluation of Agent A3-reignite.

Defines the A3-reignite variant:
- Uses surrogate for localized single-tile planning (focused tile r0=175, c0=50).
- Monitors the boundary of the focused 256x256 tile.
- When an uncontained fire front reaches the tile perimeter (P >= 0.1, arrival < 720 min)
  and adjacent exterior cells contain burnable fuel (FBFM > 0), triggers a REIGNITION EVENT.
- Spawns an adjacent downstream tile, sets the boundary crossing cell as ignition,
  runs the surrogate with time offset t_boundary, and stitches the expanded field.
- If out-of-scope, explains architectural and physical trade-offs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from wf052.routing.contracts import validate_arrival_field
from wf052.surrogate.adapter import TilePrediction, stitch_tiles_to_arrival_field, surrogate_output_to_arrival_field
from wf052.surrogate.dataset import rebuild_spatial_inputs
from wf052.surrogate.model import WindConditionedUNet


def check_uncontained_boundary_edges(
    arrival_q50: np.ndarray,
    burn_prob: np.ndarray,
    fbfm_domain: np.ndarray,
    r0: int,
    c0: int,
    p_thr: float = 0.1,
) -> list[dict[str, int | float]]:
    """Detect perimeter cells on the tile boundary touching burnable external fuel."""
    th, tw = arrival_q50.shape
    h_dom, w_dom = fbfm_domain.shape
    boundary_crossings = []

    # Check 4 borders of the tile: North (r=0), South (r=th-1), West (c=0), East (c=tw-1)
    borders = [
        ("north", 0, slice(0, tw), -1, 0),
        ("south", th - 1, slice(0, tw), 1, 0),
        ("west", slice(0, th), 0, 0, -1),
        ("east", slice(0, th), tw - 1, 0, 1),
    ]

    for b_name, r_idx, c_idx, dr, dc in borders:
        if isinstance(r_idx, int):
            # North or South border
            for c in range(tw):
                bp = burn_prob[r_idx, c]
                arr_t = arrival_q50[r_idx, c]
                if bp >= p_thr and np.isfinite(arr_t) and arr_t < 720.0:
                    dom_r = r0 + r_idx + dr
                    dom_c = c0 + c + dc
                    if 0 <= dom_r < h_dom and 0 <= dom_c < w_dom:
                        ext_fuel = fbfm_domain[dom_r, dom_c]
                        if ext_fuel > 0:  # Burnable fuel
                            boundary_crossings.append({
                                "border": b_name,
                                "tile_r": r_idx,
                                "tile_c": c,
                                "dom_r": dom_r,
                                "dom_c": dom_c,
                                "arrival_min": float(arr_t),
                                "burn_prob": float(bp),
                                "ext_fuel_code": int(ext_fuel),
                            })
        else:
            # West or East border
            for r in range(th):
                bp = burn_prob[r, c_idx]
                arr_t = arrival_q50[r, c_idx]
                if bp >= p_thr and np.isfinite(arr_t) and arr_t < 720.0:
                    dom_r = r0 + r + dr
                    dom_c = c0 + c_idx + dc
                    if 0 <= dom_r < h_dom and 0 <= dom_c < w_dom:
                        ext_fuel = fbfm_domain[dom_r, dom_c]
                        if ext_fuel > 0:
                            boundary_crossings.append({
                                "border": b_name,
                                "tile_r": r,
                                "tile_c": c_idx,
                                "dom_r": dom_r,
                                "dom_c": dom_c,
                                "arrival_min": float(arr_t),
                                "burn_prob": float(bp),
                                "ext_fuel_code": int(ext_fuel),
                            })

    return boundary_crossings


def main() -> None:
    print("=" * 80)
    print("PHASE 4B STEP 2e: A3-REIGNITE VARIANT EVALUATION")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Surrogate model
    ckpt_path = Path("checkpoints/kaggle_today_dry/best.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/smoke_clean/best.pt")
    assert ckpt_path.exists(), f"No checkpoint at {ckpt_path}"
    print(f"Loading surrogate model: {ckpt_path}")

    model = WindConditionedUNet(in_spatial_channels=13, base_channels=32, emb_dim=64, max_horizon_min=720.0)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load HRRR Wind
    weather_df = pd.read_parquet("data/tiles/hrrr_weather_hourly.parquet")
    u_vals = weather_df["u10_ms"].values[:12].astype(np.float32)
    v_vals = weather_df["v10_ms"].values[:12].astype(np.float32)
    wind_batch = torch.from_numpy(np.column_stack([u_vals, v_vals])).unsqueeze(0).to(device)

    # 1. Base focused tile (r0=175, c0=50)
    r0, c0 = 175, 50
    rel_ign_r = 232 - r0
    rel_ign_c = 452 - c0

    print(f"Running initial single focused tile at (r0={r0}, c0={c0})...")
    tile_tensor = rebuild_spatial_inputs(
        fire_id="camp_fire",
        crop_origin=(r0, c0),
        ignition_cell=(rel_ign_r, rel_ign_c),
        fires_base_dir="out/camp_fire_60m",
        tile_shape=(256, 256),
    )
    with torch.no_grad():
        out = model(tile_tensor.unsqueeze(0).to(device), wind_batch)
        q_tile0 = out.quantiles[0].cpu().numpy()
        p_tile0 = torch.sigmoid(out.burn_logits[0, 0]).cpu().numpy()

    # Load domain FBFM map to check fuel continuity
    fbfm_path = Path("out/camp_fire_60m/camp_fire/fbfm40.tif")
    import rasterio
    with rasterio.open(fbfm_path) as src:
        fbfm_dom = src.read(1)
    print(f"Domain FBFM raster shape: {fbfm_dom.shape}")

    # 2. Check for boundary crossings
    crossings = check_uncontained_boundary_edges(
        arrival_q50=q_tile0[1],
        burn_prob=p_tile0,
        fbfm_domain=fbfm_dom,
        r0=r0,
        c0=c0,
        p_thr=0.1,
    )
    print(f"\nBoundary crossings detected: {len(crossings)} perimeter cells")

    if crossings:
        df_cross = pd.DataFrame(crossings)
        earliest_exit = df_cross.sort_values("arrival_min").iloc[0]
        print("\nEarliest Boundary Exit Event:")
        print(f"  Border:             {earliest_exit['border']}")
        print(f"  Domain Cell (r, c): ({int(earliest_exit['dom_r'])}, {int(earliest_exit['dom_c'])})")
        print(f"  Exit Arrival Time:  {earliest_exit['arrival_min']:.1f} min")
        print(f"  Burn Probability:   {earliest_exit['burn_prob']:.4f}")
        print(f"  External Fuel Code: {int(earliest_exit['ext_fuel_code'])}")

        # 3. Trigger Reignition Event on adjacent tile
        # Align new tile centered on the boundary exit cell
        new_r0 = max(0, min(fbfm_dom.shape[0] - 256, int(earliest_exit['dom_r']) - 128))
        new_c0 = max(0, min(fbfm_dom.shape[1] - 256, int(earliest_exit['dom_c']) - 128))
        new_ign_r = int(earliest_exit['dom_r']) - new_r0
        new_ign_c = int(earliest_exit['dom_c']) - new_c0
        t_reignite = float(earliest_exit['arrival_min'])

        print(f"\n--- Spawning Reignition Tile at (r0={new_r0}, c0={new_c0}) with t0={t_reignite:.1f} min ---")
        tile_tensor2 = rebuild_spatial_inputs(
            fire_id="camp_fire",
            crop_origin=(new_r0, new_c0),
            ignition_cell=(new_ign_r, new_ign_c),
            fires_base_dir="out/camp_fire_60m",
            tile_shape=(256, 256),
        )
        with torch.no_grad():
            out2 = model(tile_tensor2.unsqueeze(0).to(device), wind_batch)
            q_tile2 = out2.quantiles[0].cpu().numpy() + t_reignite  # Shift quantiles by reignition onset
            p_tile2 = torch.sigmoid(out2.burn_logits[0, 0]).cpu().numpy()

        # 4. Stitch both tiles into continuous field
        h_full, w_full = fbfm_dom.shape
        origin_x = 607260.0
        origin_y = 4423470.0
        tp0 = TilePrediction(quantiles=q_tile0, burn_prob=p_tile0, r0=r0, c0=c0)
        tp1 = TilePrediction(quantiles=q_tile2, burn_prob=p_tile2, r0=new_r0, c0=new_c0)

        af_reignite = stitch_tiles_to_arrival_field(
            tile_predictions=[tp0, tp1],
            domain_shape=(h_full, w_full),
            origin_x=origin_x,
            origin_y=origin_y,
            resolution_m=60.0,
            crs="EPSG:32610",
            ignition_time="2018-11-08T14:33:00Z",
            p_thr=0.1,
        )
        validate_arrival_field(af_reignite)

        burned_before = int(np.sum(p_tile0 >= 0.1))
        burned_stitched = int(np.sum(np.isfinite(af_reignite["arrival_q50"].values)))
        print(f"Stitched ArrivalField validated successfully.")
        print(f"  Burned cells before reignite: {burned_before} cells ({burned_before*3600/10000:.1f} ha)")
        print(f"  Burned cells after reignite:  {burned_stitched} cells ({burned_stitched*3600/10000:.1f} ha)")
        print(f"  Expansion gained:             +{burned_stitched - burned_before} cells (+{(burned_stitched - burned_before)*3600/10000:.1f} ha)")

        print("\n" + "=" * 80)
        print("STEP 2e STATUS & SCOPE CONCLUSION")
        print("=" * 80)
        print("STATUS: IMPLEMENTED & OPERATIONAL.")
        print("Findings:")
        print("1. A3-reignite successfully detects perimeter boundary crossings on the focused tile")
        print("   and dynamically spawns an adjacent tile with time-offset ignition.")
        print("2. Trade-off: Point-ignition approximation along a linear fire front. Because the UNet")
        print("   was trained on single-point ignitions (channel 12), spawning from a single point on")
        print("   the boundary is an approximation. The precomputed 9-tile full domain grid (Phase 4A)")
        print("   avoids this approximation by covering all possible spread paths simultaneously.")
    else:
        print("No uncontained perimeter edge touched the tile boundary within horizon.")


if __name__ == "__main__":
    main()
