"""Phase 2F Step 4: Audit fuel group mapping across all 7 fire tiles.

Checks:
1. FBFM40 code -> fuel_group -> one-hot channel index for every code present in 7 tiles with cell counts.
2. State how former group 7 is encoded now (7 one-hot channels 0..6).
3. State whether urban/WUI (code 91) and never-burn codes (98, 99, 92, 93) are distinguishable in input.
   If not, report affected cells and cross-tab of teacher outcome (burned or not) by input class.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from wf052.surrogate.dataset import fbfm40_to_fuel_group


def main():
    fires_base = Path("data/tiles/fires")
    fire_names = ["bobcat", "caldor", "creek", "glass", "kincade", "tamarack", "zogg"]

    all_counts: dict[int, int] = {}
    fire_counts: dict[str, dict[int, int]] = {f: {} for f in fire_names}

    fbfm_rasters = {}

    for f in fire_names:
        fbfm_path = fires_base / f / "fbfm40.tif"
        with rasterio.open(fbfm_path) as src:
            arr = src.read(1)
            fbfm_rasters[f] = arr
            unq, cnt = np.unique(arr, return_counts=True)
            for u, c in zip(unq, cnt):
                u_int = int(u)
                all_counts[u_int] = all_counts.get(u_int, 0) + int(c)
                fire_counts[f][u_int] = int(c)

    # Standard FBFM descriptions
    desc_map = {
        91: "Urban/Developed (WUI)",
        92: "Snow/Ice",
        93: "Agriculture",
        98: "Open Water",
        99: "Barren",
        101: "GR1 Short Sparse Dry Climate Grass",
        102: "GR2 Low Load Dry Climate Grass",
        103: "GR3 Low Load Very Coarse Grass",
        104: "GR4 Moderate Load Dry Climate Grass",
        105: "GR5 Low Load Humid Climate Grass",
        106: "GR6 Moderate Load Humid Climate Grass",
        107: "GR7 High Load Dry Climate Grass",
        108: "GR8 High Load Very Coarse Grass",
        109: "GR9 Very High Load Humid Grass",
        121: "GS1 Low Load Dry Climate Grass-Shrub",
        122: "GS2 Moderate Load Dry Climate Grass-Shrub",
        123: "GS3 Moderate Load Humid Climate Grass-Shrub",
        124: "GS4 High Load Humid Climate Grass-Shrub",
        141: "SH1 Low Load Dry Climate Shrub",
        142: "SH2 Moderate Load Dry Climate Shrub",
        143: "SH3 Moderate Load Humid Climate Shrub",
        144: "SH4 Low Load Humid Climate Timber-Shrub",
        145: "SH5 High Load Dry Climate Shrub",
        146: "SH6 Low Load Humid Climate Shrub",
        147: "SH7 Very High Load Dry Climate Shrub",
        148: "SH8 High Load Humid Climate Shrub",
        149: "SH9 Very High Load Humid Climate Shrub",
        161: "TU1 Low Load Dry Climate Timber-Understory",
        162: "TU2 Moderate Load Humid Climate Timber-Understory",
        163: "TU3 Moderate Load Humid Climate Timber-Understory",
        164: "TU4 Dwarf Conifer Timber-Understory",
        165: "TU5 Very High Load Timber-Understory",
        181: "TL1 Low Load Compact Conifer Litter",
        182: "TL2 Low Load Broadleaf Litter",
        183: "TL3 Moderate Load Conifer Litter",
        184: "TL4 Small Downed Logs",
        185: "TL5 High Load Conifer Litter",
        186: "TL6 Moderate Load Broadleaf Litter",
        187: "TL7 Large Downed Logs",
        188: "TL8 Long-Needle Litter",
        189: "TL9 Very High Load Broadleaf Litter",
        201: "SB1 Low Load Activity Fuel",
        202: "SB2 Moderate Load Activity Fuel",
        203: "SB3 High Load Activity Fuel",
        204: "SB4 High Load Blowdown",
    }

    group_names = {
        0: "NB (Non-burnable / Urban / Water)",
        1: "GR (Grass)",
        2: "GS (Grass-Shrub)",
        3: "SH (Shrub)",
        4: "TU (Timber-Understory)",
        5: "TL (Timber-Litter)",
        6: "SB (Slash-Blowdown)",
    }

    records = []
    total_cells = sum(all_counts.values())

    sorted_codes = sorted(all_counts.keys())
    for code in sorted_codes:
        fg = int(fbfm40_to_fuel_group(np.array([code]))[0])
        onehot_ch = fg  # in num_classes=7 onehot, group g maps to channel g
        cnt = all_counts[code]
        pct = (cnt / total_cells) * 100.0
        records.append({
            "fbfm40_code": code,
            "description": desc_map.get(code, "Unknown"),
            "fuel_group": fg,
            "group_name": group_names.get(fg, "Unknown"),
            "onehot_channel": onehot_ch,
            "cell_count": cnt,
            "percent": pct,
        })

    df = pd.DataFrame(records)
    print("=" * 80)
    print("TABLE: FBFM40 CODE -> FUEL GROUP -> ONE-HOT CHANNEL INDEX")
    print("=" * 80)
    print(df.to_string(index=False))
    print(f"\nTotal cells across 7 tiles: {total_cells:,}")

    # Summary by group
    grp_df = df.groupby(["fuel_group", "group_name", "onehot_channel"])["cell_count"].sum().reset_index()
    grp_df["percent"] = (grp_df["cell_count"] / total_cells) * 100.0
    print("\n" + "=" * 80)
    print("SUMMARY BY FUEL GROUP / ONE-HOT CHANNEL:")
    print("=" * 80)
    print(grp_df.to_string(index=False))

    # Examine non-burnable and urban/WUI codes
    print("\n" + "=" * 80)
    print("URBAN/WUI (CODE 91) VS NON-BURNABLE CODES (92, 93, 98, 99)")
    print("=" * 80)
    nb_codes = [c for c in sorted_codes if c in (91, 92, 93, 98, 99) or c <= 100]
    nb_df = df[df["fbfm40_code"].isin(nb_codes)]
    print(nb_df.to_string(index=False))

    # Cross-tab of teacher outcome on dryrun shard if available
    dryrun_shard = Path("out/phase2e_dryrun/shard_dryrun.npz")
    if dryrun_shard.exists():
        print("\n" + "=" * 80)
        print("TEACHER OUTCOME AUDIT ON DRY-RUN SHARD SAMPLES")
        print("=" * 80)
        npz = np.load(dryrun_shard, allow_pickle=True)
        # Check burn outcome on cells of each fuel code
        # For each sample, reconstruct the crop FBFM40 and check burned status
        crop_origins = npz["crop_origins"]
        fire_ids = npz["fire_ids"]
        # censor_quantiles_packed has shape (N, 3, 8192)
        c_packed = npz["censor_quantiles_packed"]
        # unpack censor q50: index 1
        n_samples = len(fire_ids)
        
        burn_stats: dict[int, dict[str, int]] = {}
        for i in range(n_samples):
            fid = str(fire_ids[i])
            r0, c0 = int(crop_origins[i, 0]), int(crop_origins[i, 1])
            fbfm_crop = fbfm_rasters[fid][r0 : r0 + 256, c0 : c0 + 256]
            c50 = np.unpackbits(c_packed[i, 1]).reshape((256, 256))
            # burned by q50 means c50 == 0
            burned_q50 = (c50 == 0)
            
            for code in np.unique(fbfm_crop):
                mask = (fbfm_crop == code)
                n_code = int(np.sum(mask))
                n_burned = int(np.sum(burned_q50 & mask))
                if code not in burn_stats:
                    burn_stats[code] = {"total_cells": 0, "burned_cells": 0}
                burn_stats[code]["total_cells"] += n_code
                burn_stats[code]["burned_cells"] += n_burned

        bs_records = []
        for code, st in sorted(burn_stats.items()):
            fg = int(fbfm40_to_fuel_group(np.array([code]))[0])
            tot = st["total_cells"]
            bnd = st["burned_cells"]
            pct = (bnd / tot * 100.0) if tot > 0 else 0.0
            bs_records.append({
                "code": code,
                "desc": desc_map.get(code, "Unknown"),
                "fuel_group": fg,
                "total_cells": tot,
                "burned_cells": bnd,
                "burned_pct": pct,
            })
        bs_df = pd.DataFrame(bs_records)
        print(bs_df.to_string(index=False))


if __name__ == "__main__":
    main()
