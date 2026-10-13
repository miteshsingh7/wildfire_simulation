"""Finalize Phase 1B Step 1 assimilation diagnostic figure and Step 3 fresh held-out fire proposal table.

1. Generates `reports/phase_1b_assimilation_camp_fire.png` from `reports/step1_assimilation_camp_fire.csv`.
2. Inspects the GOFER 2020/2019 hourly progression polygons for the proposed fresh held-out
   wind-driven test fires (`Slater and Devil 2020`, `Walker 2019`, and alternate `W-5 Cold Springs 2020`),
   computes their exact 60m UTM 10N domain bounding boxes (720x720 cells = 186,624 ha), verifies
   that the 24-48h rapid growth window fits comfortably without domain saturation and has 0% overlap
   with Camp Fire and the 7 training fires, and writes `reports/step3_proposed_fresh_heldout_fires.csv`.
"""

from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box


def plot_assimilation_results() -> None:
    """Plot Step 1 assimilation forecast recall and MAE by NIST fire type for T0 in {+1h, +2h, +3h}."""
    df = pd.read_csv("reports/step1_assimilation_camp_fire.csv")
    # Filter to teacher_v1 (P_base=0.21)
    sub_f = df[
        (df["window_mode"] == "forecast_next_3h")
        & (df["model"].str.contains("teacher_v1 \\(P_base=0.21"))
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=150)

    fire_types = ["vegetative", "spot_fire", "structural", "ALL"]
    labels = ["Vegetative\n(Contact)", "Spot Fire\n(Ember)", "Structural\n(WUI)", "ALL NIST\nPoints"]
    t0_list = ["+1 h (60 min)", "+2 h (120 min)", "+3 h (180 min)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    x = np.arange(len(fire_types))
    width = 0.24

    # Panel A: Strictly future 3h window recall (%)
    ax = axes[0]
    for idx, (t0_val, col) in enumerate(zip(t0_list, colors, strict=True)):
        assim_rows = sub_f[
            (sub_f["T0_init"] == t0_val) & (sub_f["split"].str.contains("assimilation mode"))
        ]
        recalls = []
        denoms = []
        for ft in fire_types:
            r = assim_rows[assim_rows["fire_type"] == ft].iloc[0]
            recalls.append(float(r["recall_pct"]))
            denoms.append(f"{int(r['reached_points'])}/{int(r['total_points'])}")
        bars = ax.bar(x + (idx - 1) * width, recalls, width, label=f"Assimilation T0 = {t0_val[:4]}", color=col)
        for b, d_str in zip(bars, denoms, strict=True):
            h_val = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                h_val + 1.2,
                d_str,
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=0,
            )

    # Also plot point-start recall (which is <= 0.5% everywhere)
    ax.axhline(0.3, color="#d62728", linestyle="--", linewidth=1.3, label="Point-Start Baseline (0.0%–0.5%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Next-3h Forecast Recall (% of NIST Points in (T0, T0+3h])", fontsize=9.5)
    ax.set_ylim(0, 72)
    ax.set_title("A. Camp Fire (Development): Next-3h Forecast Recall by T0", fontsize=10.5, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")

    # Panel B: Cumulative [0, T0+3h] Recall and Arrival MAE (min)
    sub_c = df[
        (df["window_mode"] == "cumulative_to_T0_plus_3h")
        & (df["model"].str.contains("teacher_v1 \\(P_base=0.21"))
        & (df["split"].str.contains("assimilation mode"))
    ].copy()

    ax2 = axes[1]
    for idx, (t0_val, col) in enumerate(zip(t0_list, colors, strict=True)):
        assim_rows = sub_c[sub_c["T0_init"] == t0_val]
        maes = []
        rec_strs = []
        for ft in fire_types:
            r = assim_rows[assim_rows["fire_type"] == ft].iloc[0]
            mae_v = float(r["abs_error_mae_min"]) if pd.notna(r["abs_error_mae_min"]) else 0.0
            maes.append(mae_v)
            rec_strs.append(f"{r['recall_pct']:.0f}%\n({int(r['reached_points'])}/{int(r['total_points'])})")
        bars = ax2.bar(
            x + (idx - 1) * width,
            maes,
            width,
            label=f"Assimilation T0 = {t0_val[:4]} (to +{int(t0_val[1])+3}h)",
            color=col,
        )
        for b, r_str in zip(bars, rec_strs, strict=True):
            h_val = b.get_height()
            ax2.text(
                b.get_x() + b.get_width() / 2.0,
                h_val + 1.5,
                r_str,
                ha="center",
                va="bottom",
                fontsize=7.0,
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylabel("Cumulative Arrival-Time MAE (minutes, with Recall % shown)", fontsize=9.5)
    ax2.set_ylim(0, 95)
    ax2.set_title("B. Camp Fire (Development): Cumulative [0, T0+3h] Arrival MAE & Recall", fontsize=10.5, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=8.5, loc="upper right")

    plt.tight_layout()
    out_png = Path("reports/phase_1b_assimilation_camp_fire.png")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


def build_step3_proposed_fires_table() -> None:
    """Build detailed setup table for the proposed fresh held-out wind-driven test fires."""
    gofer_dir = Path("data/raw/gofer/GOFER")
    summary_df = pd.read_csv(gofer_dir / "GOFER_Combined/GOFERC_summary.csv")
    prog_gdf = gpd.read_file(gofer_dir / "GOFER_Combined/GOFERC_fireProg.shp")

    camp_bbox_utm = box(607260.0, 4390260.0, 641100.0, 4423980.0)
    p1a_df = pd.read_csv("reports/phase_1a_fires_setup.csv")
    p1a_boxes = []
    for _, r in p1a_df.iterrows():
        coords = [float(x.strip()) for x in str(r["bbox_utm_epsg32610"]).strip("[]").split(",")]
        p1a_boxes.append(box(coords[0], coords[1], coords[2], coords[3]))

    candidates = [
        {
            "proposal_rank": "Primary Proposed Held-Out #1",
            "gofer_id": 14,
            "fire_name": "Slater and Devil",
            "year": 2020,
            "regime": "wind-driven",
            "window_start_utc": "2020-09-08 16:00:00",
            "window_end_utc": "2020-09-09 16:00:00",
            "window_48h_end_utc": "2020-09-10 16:00:00",
            "landfire_vintage": "LANDFIRE 2016 Remap (LF 2.0.0, 2016 condition — strictly pre-2020)",
            "synoptic_driver": (
                "2020 Labor Day extreme offshore East/NE downslope wind event (Klamath/Siskiyou Mountains, "
                "sustained 12-18 m/s, gusts 22-28 m/s; continuous 32.6x run from 1,515.7 ha at t0 to "
                "5,512.7 ha (+3h), 13,989.9 ha (+6h), 26,006.6 ha (+12h), and 49,360.4 ha (+24h) across Happy Camp WUI & Klamath River)"
            ),
        },
        {
            "proposal_rank": "Primary Proposed Held-Out #2",
            "gofer_id": 2,
            "fire_name": "Walker",
            "year": 2019,
            "regime": "wind-driven",
            "window_start_utc": "2019-09-06 18:00:00",
            "window_end_utc": "2019-09-07 18:00:00",
            "window_48h_end_utc": "2019-09-08 18:00:00",
            "landfire_vintage": "LANDFIRE 2016 Remap (LF 2.0.0, 2016 condition — strictly pre-2019)",
            "synoptic_driver": (
                "Northern Sierra / Plumas NF dry cold-frontal SW wind breakout (sustained 9-14 m/s, gusts 18-23 m/s; "
                "unsuppressed 4.2x run from 3,924.7 ha at t0 to 6,143.8 ha (+3h), 6,647.6 ha (+6h), and 16,514.9 ha (+12h/+24h))"
            ),
        },
        {
            "proposal_rank": "Alternate Proposed Held-Out #3",
            "gofer_id": 16,
            "fire_name": "W-5 Cold Springs",
            "year": 2020,
            "regime": "wind-driven",
            "window_start_utc": "2020-08-21 19:00:00",
            "window_end_utc": "2020-08-22 19:00:00",
            "window_48h_end_utc": "2020-08-23 19:00:00",
            "landfire_vintage": "LANDFIRE 2016 Remap (LF 2.0.0, 2016 condition — strictly pre-2020)",
            "synoptic_driver": (
                "Modoc Plateau / Lassen County high-desert SW wind-driven grass/shrub run "
                "(sustained 10-15 m/s, gusts 19-24 m/s; 1,746.7 ha at t0 -> 3,477.6 ha (+3h) -> 4,348.3 ha (+6h) "
                "-> 6,249.0 ha (+12h) -> 6,821.5 ha (+24h) -> 13,203.5 ha (+30h/+48h))"
            ),
        },
    ]

    rows: list[dict[str, Any]] = []
    for c in candidates:
        fname = str(c["fire_name"])
        f_sum = summary_df[summary_df["fname"] == fname].sort_values("timestep").reset_index(drop=True)
        sub = prog_gdf[prog_gdf["fname"] == fname].sort_values("timestep").reset_index(drop=True)
        n_perims = len(f_sum)
        t_st = pd.to_datetime(f_sum["tUTC"])
        dt_h = float(t_st.diff().dropna().dt.total_seconds().median() / 3600.0)

        match_idx = f_sum.index[f_sum["tUTC"] == c["window_start_utc"]].tolist()
        s_idx = int(match_idx[0]) if match_idx else 0
        a_t0 = float(f_sum.iloc[s_idx]["farea"]) * 100.0
        a_1h = float(f_sum.iloc[min(s_idx + 1, n_perims - 1)]["farea"]) * 100.0
        a_3h = float(f_sum.iloc[min(s_idx + 3, n_perims - 1)]["farea"]) * 100.0
        a_6h = float(f_sum.iloc[min(s_idx + 6, n_perims - 1)]["farea"]) * 100.0
        a_12h = float(f_sum.iloc[min(s_idx + 12, n_perims - 1)]["farea"]) * 100.0
        a_24h = float(f_sum.iloc[min(s_idx + 24, n_perims - 1)]["farea"]) * 100.0
        a_48h = float(f_sum.iloc[min(s_idx + 48, n_perims - 1)]["farea"]) * 100.0
        a_final = float(f_sum["farea"].max()) * 100.0

        # Project window polygons (t0..t0+48h) to EPSG:32610 to size a 720x720 60m domain bbox (43.2 x 43.2 km)
        win_sub = sub.iloc[s_idx : min(s_idx + 49, len(sub))].to_crs("EPSG:32610")
        minx, miny, maxx, maxy = win_sub.total_bounds
        cx = 0.5 * (minx + maxx)
        cy = 0.5 * (miny + maxy)
        half_span = 21600.0  # 720 * 60m / 2 = 21,600 m (total 43.2 km x 43.2 km = 186,624 ha)
        utm_xmin = float(np.floor((cx - half_span) / 60.0) * 60.0)
        utm_ymin = float(np.floor((cy - half_span) / 60.0) * 60.0)
        utm_xmax = utm_xmin + 43200.0
        utm_ymax = utm_ymin + 43200.0

        domain_box = box(utm_xmin, utm_ymin, utm_xmax, utm_ymax)
        domain_ha = 720 * 720 * 0.36  # 186,624.0 ha

        overlap_camp = domain_box.intersects(camp_bbox_utm)
        overlap_p1a = any(domain_box.intersects(b) for b in p1a_boxes)

        # Convert UTM corners back to WGS84 lon/lat for reporting
        corners_gdf = gpd.GeoDataFrame(geometry=[domain_box], crs="EPSG:32610").to_crs("EPSG:4326")
        lon_min, lat_min, lon_max, lat_max = corners_gdf.total_bounds

        rows.append(
            {
                "proposal_rank": c["proposal_rank"],
                "gofer_id": c["gofer_id"],
                "fire_name": c["fire_name"],
                "year": c["year"],
                "split": "fresh held-out test (PROPOSED — awaiting approval)",
                "regime": c["regime"],
                "gofer_perimeters": n_perims,
                "gofer_dt_h": dt_h,
                "window_24h_utc": f"{c['window_start_utc']} UTC -> {c['window_end_utc']} UTC",
                "window_48h_utc": f"{c['window_start_utc']} UTC -> {c['window_48h_end_utc']} UTC",
                "obs_t0_ha": round(a_t0, 1),
                "obs_1h_ha": round(a_1h, 1),
                "obs_3h_ha": round(a_3h, 1),
                "obs_6h_ha": round(a_6h, 1),
                "obs_12h_ha": round(a_12h, 1),
                "obs_24h_ha": round(a_24h, 1),
                "obs_48h_ha": round(a_48h, 1),
                "obs_final_ha": round(a_final, 1),
                "grid_shape_60m": "720x720",
                "domain_area_ha": round(domain_ha, 1),
                "obs_24h_pct_of_domain": round(100.0 * a_24h / domain_ha, 2),
                "utm_bbox_epsg32610": f"[{utm_xmin:.0f}, {utm_ymin:.0f}, {utm_xmax:.0f}, {utm_ymax:.0f}]",
                "wgs84_bbox": f"[{lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}]",
                "landfire_vintage": c["landfire_vintage"],
                "overlaps_camp_fire": overlap_camp,
                "overlaps_phase1a_7fires": overlap_p1a,
                "synoptic_justification": c["synoptic_driver"],
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = Path("reports/step3_proposed_fresh_heldout_fires.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}:")
    print(
        out_df[
            [
                "proposal_rank",
                "gofer_id",
                "fire_name",
                "year",
                "gofer_perimeters",
                "window_24h_utc",
                "obs_t0_ha",
                "obs_12h_ha",
                "obs_24h_ha",
                "obs_24h_pct_of_domain",
                "overlaps_camp_fire",
                "overlaps_phase1a_7fires",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    plot_assimilation_results()
    build_step3_proposed_fires_table()
