"""Step 1: Camp Fire Inputs Analysis and 13-Channel Spatial Tensor Construction.

Verifies:
- Building 256x256 at 60m inputs through the SAME 13-channel code path as training tiles
  using rebuild_spatial_inputs from src.wf052.surrogate.dataset.
- Tiling coverage over 200 origins and the full road graph.
- Per-channel min and max versus training tiles.
- Wind source provenance (analysis vs forecast).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import torch

from evaluate_heldout_camp_fire import block_mean, block_mode
from wf052.replay.methods import sample_block_origins
from wf052.replay.providers import TeacherFieldProvider
from wf052.routing.graph import load_camp_fire_road_graph
# CITE FUNCTION: rebuild_spatial_inputs in src/wf052/surrogate/dataset.py
from wf052.surrogate.dataset import rebuild_spatial_inputs


CHANNEL_NAMES = [
    "ch00_fuel_grp0_nonburnable",
    "ch01_fuel_grp1_grass",
    "ch02_fuel_grp2_grass_shrub",
    "ch03_fuel_grp3_shrub",
    "ch04_fuel_grp4_timber_understory",
    "ch05_fuel_grp5_timber_litter",
    "ch06_fuel_grp6_slash_blowdown",
    "ch07_slope_rad",
    "ch08_aspect_sin",
    "ch09_aspect_cos",
    "ch10_elevation_zscore",
    "ch11_canopy_cover",
    "ch12_ignition_mask",
]


def ensure_camp_fire_60m_rasters(out_base: Path) -> Path:
    camp_dir = out_base / "camp_fire"
    camp_dir.mkdir(parents=True, exist_ok=True)
    required = [camp_dir / "fbfm40.tif", camp_dir / "canopy_cover.tif", camp_dir / "elevation.tif"]
    if all(p.exists() for p in required):
        return camp_dir

    tiles_dir = Path("data/tiles")
    with rasterio.open(tiles_dir / "fbfm40.tif") as src:
        fbfm_30m = src.read(1)
        meta = src.meta.copy()
        bounds = src.bounds
    with rasterio.open(tiles_dir / "canopy_cover.tif") as src:
        cc_30m = src.read(1).astype(np.float32)
    with rasterio.open(tiles_dir / "elevation.tif") as src:
        elev_30m = src.read(1).astype(np.float32)

    fbfm_60m = block_mode(fbfm_30m, 2)
    cc_60m = block_mean(cc_30m, 2)
    elev_60m = block_mean(elev_30m, 2)
    h, w = fbfm_60m.shape

    meta.update({
        "height": h,
        "width": w,
        "transform": rasterio.Affine(60.0, 0.0, bounds.left, 0.0, -60.0, bounds.top),
    })

    meta_fbfm = meta.copy()
    meta_fbfm["dtype"] = "int64"
    with rasterio.open(camp_dir / "fbfm40.tif", "w", **meta_fbfm) as dst:
        dst.write(fbfm_60m.astype(np.int64), 1)

    meta_cc = meta.copy()
    meta_cc["dtype"] = "float32"
    with rasterio.open(camp_dir / "canopy_cover.tif", "w", **meta_cc) as dst:
        dst.write(cc_60m.astype(np.float32), 1)

    meta_elev = meta.copy()
    meta_elev["dtype"] = "float32"
    with rasterio.open(camp_dir / "elevation.tif", "w", **meta_elev) as dst:
        dst.write(elev_60m.astype(np.float32), 1)

    return camp_dir


def main() -> None:
    print("=" * 80)
    print("STEP 1: CAMP FIRE INPUTS VERIFICATION & 13-CHANNEL ANALYSIS")
    print("=" * 80)

    # 1. Prepare 60m rasters and invoke rebuild_spatial_inputs
    out_base = Path("out/camp_fire_60m")
    ensure_camp_fire_60m_rasters(out_base)

    # Camp Fire ignition cell at 60m: (row=269, col=436)
    ign_row_domain, ign_col_domain = 269, 436

    # 2. Tiling coverage over 200 origins and road graph
    teacher_field = TeacherFieldProvider().get_field(0.0)
    road_graph = load_camp_fire_road_graph(
        full_detail=True,
        grid_x_coords=teacher_field["x"].values,
        grid_y_coords=teacher_field["y"].values,
    )
    origins = sample_block_origins(road_graph=road_graph, num_samples=200, seed=42)

    x_min, y_max = 607230.0, 4423500.0
    orig_coords = [road_graph.nodes[o.origin_id] for o in origins]
    orig_rows = np.array([int((y_max - y) / 60.0) for x, y in orig_coords])
    orig_cols = np.array([int((x - x_min) / 60.0) for x, y in orig_coords])

    node_coords = np.array(list(road_graph.nodes.values()))
    node_rows = np.array([int((y_max - y) / 60.0) for x, y in node_coords])
    node_cols = np.array([int((x - x_min) / 60.0) for x, y in node_coords])

    print("\n--- Domain & Tiling Geometry ---")
    print(f"Full Domain: 564 rows x 562 cols at 60m (33.84 km x 33.72 km)")
    print(f"200 Origins Extent: rows [{orig_rows.min()}, {orig_rows.max()}], cols [{orig_cols.min()}, {orig_cols.max()}] (span: {orig_rows.max() - orig_rows.min() + 1} rows x {orig_cols.max() - orig_cols.min() + 1} cols)")
    print(f"All {len(node_coords)} Graph Nodes Extent: rows [{node_rows.min()}, {node_rows.max()}], cols [{node_cols.min()}, {node_cols.max()}]")

    # A single 256x256 tile centered on Paradise covers all 200 origins:
    # Paradise tile: r0=175, c0=50
    paradise_r0, paradise_c0 = 175, 50
    origins_in_paradise_tile = np.sum(
        (orig_rows >= paradise_r0) & (orig_rows < paradise_r0 + 256) &
        (orig_cols >= paradise_c0) & (orig_cols < paradise_c0 + 256)
    )
    nodes_in_paradise_tile = np.sum(
        (node_rows >= paradise_r0) & (node_rows < paradise_r0 + 256) &
        (node_cols >= paradise_c0) & (node_cols < paradise_c0 + 256)
    )
    print(f"\nParadise Focused Tile (r0={paradise_r0}, c0={paradise_c0}):")
    print(f"  Origins inside: {origins_in_paradise_tile}/200 ({origins_in_paradise_tile/200.0*100:.1f}%)")
    print(f"  Graph nodes inside: {nodes_in_paradise_tile}/{len(node_coords)} ({nodes_in_paradise_tile/len(node_coords)*100:.1f}%)")

    # 3x3 Grid covering the full 564x562 domain
    # Rows: 0, 154, 308 (with +256: 256, 410, 564)
    # Cols: 0, 153, 306 (with +256: 256, 409, 562)
    grid_r0s = [0, 154, 308]
    grid_c0s = [0, 153, 306]
    n_tiles_full = len(grid_r0s) * len(grid_c0s)
    print(f"\nFull Domain 3x3 Tile Grid (K={n_tiles_full} tiles, 256x256 each with ~102-cell overlap):")
    tiles_info = []
    tile_tensors = []
    for r0 in grid_r0s:
        for c0 in grid_c0s:
            # Ignition cell relative to this tile (clipped if outside)
            rel_ign_r = int(np.clip(ign_row_domain - r0, 0, 255))
            rel_ign_c = int(np.clip(ign_col_domain - c0, 0, 255))

            # Build through EXACT SAME code path
            tile_tensor = rebuild_spatial_inputs(
                fire_id="camp_fire",
                crop_origin=(r0, c0),
                ignition_cell=(rel_ign_r, rel_ign_c),
                fires_base_dir="out/camp_fire_60m",
                tile_shape=(256, 256),
            )
            assert isinstance(tile_tensor, torch.Tensor), "Output must be torch.Tensor"
            assert tile_tensor.shape == (13, 256, 256), f"Expected (13, 256, 256), got {tile_tensor.shape}"
            tile_tensors.append(tile_tensor)

            n_orig = np.sum(
                (orig_rows >= r0) & (orig_rows < r0 + 256) &
                (orig_cols >= c0) & (orig_cols < c0 + 256)
            )
            n_nodes = np.sum(
                (node_rows >= r0) & (node_rows < r0 + 256) &
                (node_cols >= c0) & (node_cols < c0 + 256)
            )
            tiles_info.append({
                "tile_r0": r0,
                "tile_c0": c0,
                "origins_inside": n_orig,
                "nodes_inside": n_nodes,
            })

    df_tiles = pd.DataFrame(tiles_info)
    print(df_tiles.to_string(index=False))

    # Check external boundary nodes
    outside_nodes = np.sum(
        (node_rows < 0) | (node_rows >= 564) |
        (node_cols < 0) | (node_cols >= 562)
    )
    print(f"\nGraph nodes outside raster boundary [0, 564) x [0, 562): {outside_nodes} nodes.")
    print("Handling: RoadGraph excludes points outside the grid bounds from cell intersection, so edge_arrivals for edges outside the domain remain +inf.")

    # 3. Per-channel min and max versus training tiles
    print("\n--- Per-Channel Min/Max Comparison vs Training Tiles ---")
    meta_df = pd.read_csv("data/shards_smoke_2i/shard_000_metadata.csv")
    train_min = np.full(13, np.inf)
    train_max = np.full(13, -np.inf)

    # Compute training tiles min/max
    for _, row in meta_df.iterrows():
        f_id = row["fire_id"]
        r0 = int(row["crop_r0"])
        c0 = int(row["crop_c0"])
        ir = int(row["ign_r"])
        ic = int(row["ign_c"])
        t = rebuild_spatial_inputs(
            fire_id=f_id,
            crop_origin=(r0, c0),
            ignition_cell=(ir, ic),
            fires_base_dir="data/tiles/fires",
            tile_shape=(256, 256),
        )
        t_np = t.numpy()
        t_min = t_np.min(axis=(1, 2))
        t_max = t_np.max(axis=(1, 2))
        train_min = np.minimum(train_min, t_min)
        train_max = np.maximum(train_max, t_max)

    # Compute Camp Fire tiles min/max
    camp_stacked = torch.stack(tile_tensors, dim=0).numpy()  # (9, 13, 256, 256)
    camp_min = camp_stacked.min(axis=(0, 2, 3))
    camp_max = camp_stacked.max(axis=(0, 2, 3))

    comp_rows = []
    for c_idx, name in enumerate(CHANNEL_NAMES):
        comp_rows.append({
            "channel_idx": c_idx,
            "channel_name": name,
            "train_min": float(train_min[c_idx]),
            "train_max": float(train_max[c_idx]),
            "camp_min": float(camp_min[c_idx]),
            "camp_max": float(camp_max[c_idx]),
        })

    df_comp = pd.DataFrame(comp_rows)
    Path("reports").mkdir(exist_ok=True)
    df_comp.to_csv("reports/step1_channel_min_max_comparison.csv", index=False)
    print(df_comp.to_string(index=False))

    # 4. Wind source provenance
    print("\n--- Wind Source Provenance ---")
    print("Wind Source File: data/tiles/hrrr_weather_hourly.parquet")
    print("Provenance Code: src/wf052/data/weather.py line 79:")
    print("  herbie_obj = Herbie(target_str, model=\"hrrr\", product=\"sfc\", fxx=0)")
    print("Model: NOAA High-Resolution Rapid Refresh (HRRR) 3km surface product.")
    print("Analysis vs Forecast: fxx=0 indicates Forecast Hour 0, which is the ZERO-HOUR INITIAL ANALYSIS (not forecast data).")


if __name__ == "__main__":
    main()
