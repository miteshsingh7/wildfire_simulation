"""Phase 2I Step 2: Wind vs Burned Fraction Analysis on 1d Smoke Attempt Log.

Analyzes:
1. For all K=1 candidates (per fire and pooled):
   - count, median and P90 burned fraction by wind bin (<2, 2-4, 4-8, 8-12, >=12 m/s)
   - counts in the 5 burned fraction bins ([0, 0.002), [0.002, 0.02), [0.02, 0.1), [0.1, 0.5), [0.5, 1.0])
   Both for wind_mean and wind_peak.
2. Wind bin of every accepted sample in bins 3 and 4.
3. Definition of the 'Max Wind Speed' column from 2H Table 3b.

Saves:
- reports/step2_wind_vs_burned_fraction_pooled.csv
- reports/step2_wind_vs_burned_fraction_per_fire.csv
- reports/step2_accepted_bins_3_and_4_wind.csv
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def get_wind_bin(val: float) -> str:
    if val < 2.0:
        return "<2"
    elif val < 4.0:
        return "2-4"
    elif val < 8.0:
        return "4-8"
    elif val < 12.0:
        return "8-12"
    else:
        return ">=12"


WIND_BINS = ["<2", "2-4", "4-8", "8-12", ">=12"]


def compute_wind_stats_table(df_k1: pd.DataFrame, wind_col: str, group_col: str | None = None) -> pd.DataFrame:
    records = []
    groups = [("Pooled", df_k1)] if group_col is None else [(g, sub) for g, sub in df_k1.groupby(group_col)]

    for grp_name, sub_df in groups:
        for w_bin in WIND_BINS:
            bin_sub = sub_df[sub_df[wind_col + "_bin"] == w_bin]
            cnt = len(bin_sub)
            if cnt > 0:
                bf_vals = bin_sub["bf_k1"].values
                med_bf = float(np.median(bf_vals))
                p90_bf = float(np.percentile(bf_vals, 90))
                b0 = int((bin_sub["bin_k1"] == 0).sum())
                b1 = int((bin_sub["bin_k1"] == 1).sum())
                b2 = int((bin_sub["bin_k1"] == 2).sum())
                b3 = int((bin_sub["bin_k1"] == 3).sum())
                b4 = int((bin_sub["bin_k1"] == 4).sum())
            else:
                med_bf = 0.0
                p90_bf = 0.0
                b0 = b1 = b2 = b3 = b4 = 0

            records.append({
                "group": grp_name,
                "wind_metric": wind_col,
                "wind_bin": w_bin,
                "candidate_count": cnt,
                "median_burned_fraction": round(med_bf, 6),
                "p90_burned_fraction": round(p90_bf, 6),
                "bf_bin0_count": b0,
                "bf_bin1_count": b1,
                "bf_bin2_count": b2,
                "bf_bin3_count": b3,
                "bf_bin4_count": b4,
            })
    return pd.DataFrame(records)


def main() -> None:
    print("=" * 70)
    print("PHASE 2I STEP 2: WIND VS BURNED FRACTION ANALYSIS")
    print("=" * 70)

    log_path = Path("reports/smoke_2i_attempts_log.csv")
    meta_path = Path("data/shards_smoke_2i/shard_000_metadata.csv")

    if not log_path.exists():
        raise FileNotFoundError(f"Attempt log not found at {log_path}")

    df_att = pd.read_csv(log_path)
    print(f"Loaded {len(df_att)} total attempt records from {log_path}.")

    # Filter for K=1 candidate simulations (exclude fuel group 0 non-burnable rejections)
    df_k1 = df_att[~df_att["is_fuel_0"]].copy()
    print(f"Total simulated K=1 candidates: {len(df_k1)}")

    # Ensure wind bins exist
    if "wind_mean_bin" not in df_k1.columns:
        df_k1["wind_mean_bin"] = df_k1["wind_mean"].apply(get_wind_bin)
    if "wind_peak_bin" not in df_k1.columns:
        df_k1["wind_peak_bin"] = df_k1["wind_peak"].apply(get_wind_bin)

    # 1. Pooled stats by wind_mean and wind_peak
    df_pooled_mean = compute_wind_stats_table(df_k1, "wind_mean", group_col=None)
    df_pooled_peak = compute_wind_stats_table(df_k1, "wind_peak", group_col=None)
    df_pooled_all = pd.concat([df_pooled_mean, df_pooled_peak], ignore_index=True)
    df_pooled_all.to_csv("reports/step2_wind_vs_burned_fraction_pooled.csv", index=False)

    print("\n--- Pooled Wind Mean vs Burned Fraction ---")
    print(df_pooled_mean.to_string(index=False))

    print("\n--- Pooled Wind Peak vs Burned Fraction ---")
    print(df_pooled_peak.to_string(index=False))

    # 2. Per-fire stats by wind_mean and wind_peak
    df_fire_mean = compute_wind_stats_table(df_k1, "wind_mean", group_col="fire")
    df_fire_peak = compute_wind_stats_table(df_k1, "wind_peak", group_col="fire")
    df_fire_all = pd.concat([df_fire_mean, df_fire_peak], ignore_index=True)
    df_fire_all.to_csv("reports/step2_wind_vs_burned_fraction_per_fire.csv", index=False)
    print("\nSaved reports/step2_wind_vs_burned_fraction_per_fire.csv")

    # 3. Accepted samples in bins 3 and 4
    if meta_path.exists():
        df_meta = pd.read_csv(meta_path)
        sub_34 = df_meta[df_meta["bin_kmean"].isin([3, 4])].copy()
        rows_34 = []
        for idx, row in sub_34.iterrows():
            # Match to attempt log
            match_att = df_att[
                (df_att["fire"] == row["fire"])
                & (df_att["ign_r"] == row["ign_r"])
                & (df_att["ign_c"] == row["ign_c"])
                & (df_att["accepted"])
            ]
            w_mean = match_att["wind_mean"].values[0] if len(match_att) > 0 and "wind_mean" in match_att else float("nan")
            w_peak = float(row["peak_wind"]) if "peak_wind" in row else match_att["wind_peak"].values[0] if len(match_att) > 0 else float("nan")

            rows_34.append({
                "sample_id": row["sample_id"],
                "fire": row["fire"],
                "bin_k1": int(row["bin_k1"]),
                "bin_kmean": int(row["bin_kmean"]),
                "bf_k1": float(row["burned_fraction_k1"]),
                "bf_kmean": float(row["burned_fraction_kmean"]),
                "wind_mean": round(w_mean, 4) if np.isfinite(w_mean) else "NOT RECORDED",
                "wind_mean_bin": get_wind_bin(w_mean) if np.isfinite(w_mean) else "UNKNOWN",
                "wind_peak": round(w_peak, 4),
                "wind_peak_bin": get_wind_bin(w_peak),
            })
        df_acc_34 = pd.DataFrame(rows_34)
    else:
        # Fallback to attempt log accepted in bin 3 or 4
        acc_34 = df_att[df_att["accepted"] & df_att["bin_k1"].isin([3, 4])].copy()
        rows_34 = []
        for idx, row in acc_34.iterrows():
            w_mean = float(row["wind_mean"])
            w_peak = float(row["wind_peak"])
            rows_34.append({
                "attempt": row["attempt"],
                "fire": row["fire"],
                "bin_k1": int(row["bin_k1"]),
                "bf_k1": float(row["bf_k1"]),
                "wind_mean": round(w_mean, 4),
                "wind_mean_bin": get_wind_bin(w_mean),
                "wind_peak": round(w_peak, 4),
                "wind_peak_bin": get_wind_bin(w_peak),
            })
        df_acc_34 = pd.DataFrame(rows_34)

    df_acc_34.to_csv("reports/step2_accepted_bins_3_and_4_wind.csv", index=False)
    print("\n--- Accepted Samples in Bins 3 and 4 ---")
    print(df_acc_34.to_string(index=False))
    print("\nSaved reports/step2_accepted_bins_3_and_4_wind.csv")


if __name__ == "__main__":
    main()
