"""Inspect detailed hourly progression around rapid growth windows for all 7 fires."""

from pathlib import Path

import pandas as pd


def main() -> None:
    gofer_dir = Path("data/raw/gofer/GOFER")
    summary_df = pd.read_csv(gofer_dir / "GOFER_Combined/GOFERC_summary.csv")

    target_fires = ["Kincade", "Bobcat", "Glass", "Creek", "Zogg", "Caldor", "Tamarack"]
    for fname in target_fires:
        f_sum = summary_df[summary_df["fname"] == fname].sort_values("timestep").reset_index(drop=True)
        f_sum["area_ha"] = f_sum["farea"] * 100.0
        f_sum["gain_24h"] = f_sum["area_ha"].shift(-24) - f_sum["area_ha"]
        f_sum["gain_12h"] = f_sum["area_ha"].shift(-12) - f_sum["area_ha"]

        # Best 24h window where starting area is relatively compact (< 5000 ha) and growth is continuous
        compact = f_sum[f_sum["area_ha"] <= 5000.0]
        if len(compact) > 0 and compact["gain_24h"].notna().any():
            best_idx = int(compact["gain_24h"].idxmax())
        else:
            best_idx = 0

        r0 = f_sum.iloc[best_idx]
        r6 = f_sum.iloc[min(best_idx + 6, len(f_sum) - 1)]
        r12 = f_sum.iloc[min(best_idx + 12, len(f_sum) - 1)]
        r24 = f_sum.iloc[min(best_idx + 24, len(f_sum) - 1)]
        r48 = f_sum.iloc[min(best_idx + 48, len(f_sum) - 1)]

        print(f"=== {fname} ===")
        print(
            f"  Candidate Start (compact <=5000ha max 24h gain): step {r0['timestep']} ({r0['tUTC']}) "
            f"area={r0['area_ha']:.1f} ha"
        )
        print(
            f"    +6h ({r6['tUTC']}): {r6['area_ha']:.1f} ha | "
            f"+12h ({r12['tUTC']}): {r12['area_ha']:.1f} ha | "
            f"+24h ({r24['tUTC']}): {r24['area_ha']:.1f} ha | "
            f"+48h ({r48['tUTC']}): {r48['area_ha']:.1f} ha"
        )
        # Also show from step index 0 (first row in summary)
        s0 = f_sum.iloc[0]
        s6 = f_sum.iloc[min(6, len(f_sum) - 1)]
        s12 = f_sum.iloc[min(12, len(f_sum) - 1)]
        s24 = f_sum.iloc[min(24, len(f_sum) - 1)]
        s48 = f_sum.iloc[min(48, len(f_sum) - 1)]
        print(
            f"  From Summary Row 0: step {s0['timestep']} ({s0['tUTC']}) area={s0['area_ha']:.1f} ha -> "
            f"+6h: {s6['area_ha']:.1f} ha | +12h: {s12['area_ha']:.1f} ha | "
            f"+24h: {s24['area_ha']:.1f} ha | +48h: {s48['area_ha']:.1f} ha"
        )


if __name__ == "__main__":
    main()
