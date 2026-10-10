"""Analyze GOFERC_summary.csv and GOFERC_fireProg.shp for the 7 approved training fires."""

from pathlib import Path

import geopandas as gpd
import pandas as pd


def main() -> None:
    gofer_dir = Path("data/raw/gofer/GOFER")
    fire_data = pd.read_csv(gofer_dir / "fireData.csv")
    summary_df = pd.read_csv(gofer_dir / "GOFER_Combined/GOFERC_summary.csv")
    prog_gdf = gpd.read_file(gofer_dir / "GOFER_Combined/GOFERC_fireProg.shp")
    ig_gdf = gpd.read_file(gofer_dir / "GOFER_Combined/GOFERC_fireIg.shp")

    print("GOFERC_fireProg columns:", list(prog_gdf.columns))
    print("GOFERC_fireProg CRS:", prog_gdf.crs)
    print("GOFERC_fireIg columns:", list(ig_gdf.columns))

    target_fires = ["Kincade", "Bobcat", "Glass", "Creek", "Zogg", "Caldor", "Tamarack"]

    for fname in target_fires:
        f_meta = fire_data[fire_data["fname"] == fname].iloc[0]
        f_sum = summary_df[summary_df["fname"] == fname].sort_values("timestep").reset_index(drop=True)
        f_prog = prog_gdf[prog_gdf["fname"] == fname].sort_values("timestep").reset_index(drop=True)
        f_ig = ig_gdf[ig_gdf["fname"] == fname]

        # Compute time step in hours
        t_utc = pd.to_datetime(f_sum["tUTC"])
        dt_hours = t_utc.diff().dt.total_seconds() / 3600.0
        median_dt = float(dt_hours.dropna().median())

        # Find rapid 24h and 48h growth windows
        f_sum["area_ha"] = f_sum["farea"] * 100.0  # km^2 to ha
        f_sum["d_area_ha"] = f_sum["dfarea"] * 100.0

        # Check initial 24h and 48h from timestep 1
        area_t1 = float(f_sum.iloc[0]["area_ha"])
        area_24h = float(f_sum.iloc[min(23, len(f_sum) - 1)]["area_ha"])
        area_48h = float(f_sum.iloc[min(47, len(f_sum) - 1)]["area_ha"])
        max_area_ha = float(f_sum["area_ha"].max())

        # Rolling 24h and 48h area gain
        gain_24h = f_sum["area_ha"].diff(24).fillna(f_sum["area_ha"])
        best_end_24 = int(gain_24h.idxmax())
        best_start_24 = max(0, best_end_24 - 24)

        gain_48h = f_sum["area_ha"].diff(48).fillna(f_sum["area_ha"])
        best_end_48 = int(gain_48h.idxmax())
        best_start_48 = max(0, best_end_48 - 48)

        bounds = f_prog.total_bounds  # minx, miny, maxx, maxy
        ig_coords = (
            (float(f_ig.geometry.centroid.x.iloc[0]), float(f_ig.geometry.centroid.y.iloc[0]))
            if len(f_ig) > 0
            else None
        )

        print(f"\n==================== {fname} ({f_meta['fyear']}) ====================")
        print(f"Official Acres: {f_meta['acres_official']} ac ({f_meta['acres_official']*0.404686:.1f} ha)")
        print(f"GOESIg_UTC: {f_meta['GOESIg_UTC']} | Ig Point: {ig_coords}")
        print(f"Summary rows: {len(f_sum)} | Shapefile perimeters: {len(f_prog)} | Median dt: {median_dt:.1f} h")
        print(f"Total GOFER Max Area: {max_area_ha:.1f} ha ({f_sum['farea'].max():.2f} km2)")
        print(f"Prog Total Bounds (lon/lat): [{bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f}]")
        print(
            f"From t=1 ({f_sum.iloc[0]['tUTC']}): "
            f"t1={area_t1:.1f} ha -> +24h={area_24h:.1f} ha -> +48h={area_48h:.1f} ha"
        )
        print(
            f"Peak 24h growth: step {f_sum.iloc[best_start_24]['timestep']} ({f_sum.iloc[best_start_24]['tUTC']}) "
            f"to step {f_sum.iloc[best_end_24]['timestep']} ({f_sum.iloc[best_end_24]['tUTC']}): "
            f"{f_sum.iloc[best_start_24]['area_ha']:.1f} ha -> {f_sum.iloc[best_end_24]['area_ha']:.1f} ha "
            f"(+{gain_24h.iloc[best_end_24]:.1f} ha)"
        )
        print(
            f"Peak 48h growth: step {f_sum.iloc[best_start_48]['timestep']} ({f_sum.iloc[best_start_48]['tUTC']}) "
            f"to step {f_sum.iloc[best_end_48]['timestep']} ({f_sum.iloc[best_end_48]['tUTC']}): "
            f"{f_sum.iloc[best_start_48]['area_ha']:.1f} ha -> {f_sum.iloc[best_end_48]['area_ha']:.1f} ha "
            f"(+{gain_48h.iloc[best_end_48]:.1f} ha)"
        )
        # Also print first 5 timesteps where area > 0
        non_zero = f_sum[f_sum["area_ha"] > 0]
        if len(non_zero) > 0:
            first_nz = non_zero.iloc[0]
            idx_nz = int(first_nz.name)
            idx_nz_24 = min(idx_nz + 24, len(f_sum) - 1)
            idx_nz_48 = min(idx_nz + 48, len(f_sum) - 1)
            print(
                f"First non-zero detection: step {first_nz['timestep']} ({first_nz['tUTC']}, {first_nz['area_ha']:.1f} ha) "
                f"-> +24h ({f_sum.iloc[idx_nz_24]['tUTC']}): {f_sum.iloc[idx_nz_24]['area_ha']:.1f} ha "
                f"-> +48h ({f_sum.iloc[idx_nz_48]['tUTC']}): {f_sum.iloc[idx_nz_48]['area_ha']:.1f} ha"
            )


if __name__ == "__main__":
    main()
