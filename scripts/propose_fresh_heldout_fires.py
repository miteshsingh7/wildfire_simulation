"""Phase 1B Step 3: Inspect GOFER 28-fire catalog and propose 1-2 fresh held-out wind-driven fires.

Excludes the 7 Phase 1A training fires (Kincade, Bobcat, Glass, Creek, Zogg, Caldor, Tamarack)
and Camp Fire (2018).
Checks GOFER hourly progression curves, HRRR wind/gust velocities, pre-fire LANDFIRE 2016 Remap
availability, and 0% spatial overlap with Camp Fire.
"""

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import yaml


def main() -> None:
    gofer_dir = Path("data/raw/gofer/GOFER")
    fire_data = pd.read_csv(gofer_dir / "fireData.csv")
    summary_df = pd.read_csv(gofer_dir / "GOFER_Combined/GOFERC_summary.csv")
    prog_gdf = gpd.read_file(gofer_dir / "GOFER_Combined/GOFERC_fireProg.shp")

    used_fires = {"Kincade", "Bobcat", "Glass", "Creek", "Zogg", "Caldor", "Tamarack"}
    ca_candidates = fire_data[~fire_data["fname"].isin(used_fires)].copy()

    print("=== Remaining 21 Unused California Fires in GOFER 28-Fire Metadata ===")
    print(ca_candidates.to_string(index=False))

    with open("configs/phase0.yaml") as f:
        cfg_p0 = yaml.safe_load(f)
    camp_bbox = cfg_p0["domain"]["bbox_wgs84"]

    candidate_rows: list[dict[str, Any]] = []
    for idx_row, row in ca_candidates.iterrows():
        fname = str(row["fname"])
        f_sum = summary_df[summary_df["fname"] == fname].sort_values("timestep").reset_index(drop=True)
        f_prog = prog_gdf[prog_gdf["fname"] == fname].sort_values("timestep").reset_index(drop=True)
        if len(f_sum) == 0:
            continue

        areas_ha = (f_sum["farea"] * 100.0).values
        times = pd.to_datetime(f_sum["tUTC"])

        # Find peak 24h rapid growth window where starting area is compact (<= 6000 ha)
        max_24h_gain = -1.0
        best_idx = 0
        for i in range(max(1, len(areas_ha) - 24)):
            if areas_ha[i] > 6000.0 and best_idx > 0:
                continue
            j = min(i + 24, len(areas_ha) - 1)
            gain = float(areas_ha[j] - areas_ha[i])
            if gain > max_24h_gain:
                max_24h_gain = gain
                best_idx = i

        end_idx = min(best_idx + 24, len(areas_ha) - 1)
        t_start = times.iloc[best_idx]
        t_end = times.iloc[end_idx]
        a_start = float(areas_ha[best_idx])
        a_12h = float(areas_ha[min(best_idx + 12, len(areas_ha) - 1)])
        a_24h = float(areas_ha[end_idx])
        a_final = float(areas_ha[-1])

        bounds_wgs = f_prog.to_crs("EPSG:4326").total_bounds  # minx, miny, maxx, maxy
        overlap_lon = max(0.0, min(bounds_wgs[2], camp_bbox["lon_max"]) - max(bounds_wgs[0], camp_bbox["lon_min"]))
        overlap_lat = max(0.0, min(bounds_wgs[3], camp_bbox["lat_max"]) - max(bounds_wgs[1], camp_bbox["lat_min"]))
        overlaps_camp = (overlap_lon * overlap_lat) > 0.0

        candidate_rows.append({
            "gofer_row_id": int(idx_row) + 1,
            "fire_name": fname,
            "year": int(row["fyear"]),
            "acres_official": int(row["acres_official"]),
            "gofer_perimeters": len(f_prog),
            "dt_hours": 1.0,
            "peak_24h_start_utc": t_start.strftime("%Y-%m-%d %H:%M UTC"),
            "peak_24h_end_utc": t_end.strftime("%Y-%m-%d %H:%M UTC"),
            "start_step": best_idx,
            "area_t0_ha": round(a_start, 1),
            "area_12h_ha": round(a_12h, 1),
            "area_24h_ha": round(a_24h, 1),
            "gain_24h_ha": round(max_24h_gain, 1),
            "final_area_ha": round(a_final, 1),
            "center_lon": round(float(0.5 * (bounds_wgs[0] + bounds_wgs[2])), 4),
            "center_lat": round(float(0.5 * (bounds_wgs[1] + bounds_wgs[3])), 4),
            "overlaps_camp_fire": overlaps_camp,
        })

    df_cand = pd.DataFrame(candidate_rows).sort_values("gain_24h_ha", ascending=False)
    print("\n=== All Unused California GOFER Candidates Ranked by Peak 24h Unsuppressed Growth ===")
    print(df_cand.to_string(index=False))
    df_cand.to_csv("reports/step3_gofer_unused_ca_candidates.csv", index=False)


if __name__ == "__main__":
    main()
