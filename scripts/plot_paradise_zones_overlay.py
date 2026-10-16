"""Generate official 14 Paradise Evacuation Zones, overlay map image, and reconciliation table (Phase 4e).

Methodology:
1. Polygon Source: US Census Bureau 2010 Places boundary for Town of Paradise (tl_2010_06_place10.shp,
   Paradise town, LSAD 43, ALAND 47.42 km2).
2. Zone Centroids & Timeline: Butte County GIS Evacuation Plan Maps (NIST TN 2252 sup Appendix C, p. 124-125)
   and NIST TN 2252 Table 11 (p. 94).
3. Tessellation: Dirichlet-Voronoi cellular decomposition around official zone centroids, clipped
   strictly to the Town of Paradise municipal boundary polygon.
4. Demographic Assignment: Spatial containment of 332 Census 2010 decennial blocks (tl_2010_06007_tabblock10).
   Total Town of Paradise population strictly reconciles to 26,218 persons.
5. Saves:
   - reports/paradise_zones_overlay.png: High-resolution cartographic map with zones, blocks, corridors, and exits.
   - reports/paradise_14_zones_reconciliation.csv: Detailed block and demographic table with reconciled TOTAL row.
   - data/tiles/paradise_14_zones.parquet: GeoParquet spatial polygons for the 14 evacuation zones.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry
import shapely.ops
import yaml
from scipy.spatial import Voronoi


def build_and_save_14_zones() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    with open("configs/evacuation_demand.yaml") as f:
        cfg = yaml.safe_load(f)

    occupancy = float(cfg.get("occupancy", {}).get("persons_per_vehicle", 1.6))
    zone_configs = {k: v for k, v in cfg["zones"].items() if k.startswith("Zone_")}

    # 1. Load Town Boundary (EPSG:32610 UTM Zone 10N)
    places = gpd.read_file("data/raw/census2010_places/tl_2010_06_place10.shp").to_crs("EPSG:32610")
    town_poly = places[places["NAMELSAD10"] == "Paradise town"].iloc[0].geometry

    # 2. Load Census 2010 Blocks
    blocks = gpd.read_parquet("data/tiles/census2010_butte_blocks_pophu.parquet").to_crs(
        "EPSG:32610"
    )
    # Spatial filter for blocks with centroids inside Town of Paradise boundary
    p_blocks = blocks[blocks.geometry.centroid.within(town_poly)].copy()
    assert len(p_blocks) == 332, f"Expected 332 blocks, got {len(p_blocks)}"
    assert p_blocks["POP10"].sum() == 26218, (
        f"Expected 26,218 population, got {p_blocks['POP10'].sum()}"
    )

    # 3. Voronoi Tessellation of Town Polygon around 14 Official Centroids
    zone_names = [f"Zone_{i}" for i in range(1, 15)]
    coords = np.array([zone_configs[z]["centroid_utm"] for z in zone_names])

    minx, miny, maxx, maxy = town_poly.bounds
    far_pts = np.array(
        [
            [minx - 100000, miny - 100000],
            [maxx + 100000, miny - 100000],
            [maxx + 100000, maxy + 100000],
            [minx - 100000, maxy + 100000],
        ]
    )
    all_pts = np.vstack([coords, far_pts])
    vor = Voronoi(all_pts)

    lines = [
        shapely.geometry.LineString(vor.vertices[line])
        for line in vor.ridge_vertices
        if -1 not in line
    ]
    polys = list(shapely.ops.polygonize(lines))

    zone_geoms = {}
    for p in polys:
        p_clipped = p.intersection(town_poly)
        if not p_clipped.is_empty:
            for z_name, pt in zip(zone_names, coords, strict=True):
                if p.contains(shapely.geometry.Point(pt)):
                    zone_geoms[z_name] = p_clipped
                    break

    # 4. Map Census Blocks into Zones by Centroid Containment
    block_records = []
    zone_block_counts = {z: 0 for z in zone_names}
    zone_populations = {z: 0 for z in zone_names}

    for _, blk in p_blocks.iterrows():
        b_pt = blk.geometry.centroid
        assigned_z = None
        for z_name, z_poly in zone_geoms.items():
            if z_poly.contains(b_pt):
                assigned_z = z_name
                break
        if assigned_z is None:
            # Fallback to nearest centroid distance if precisely on edge boundary
            dists = [np.hypot(b_pt.x - pt[0], b_pt.y - pt[1]) for pt in coords]
            assigned_z = zone_names[int(np.argmin(dists))]

        zone_block_counts[assigned_z] += 1
        zone_populations[assigned_z] += int(blk["POP10"])
        block_records.append(
            {
                "geoid10": blk["GEOID10"],
                "pop10": int(blk["POP10"]),
                "vehicles": int(np.round(blk["POP10"] / occupancy)),
                "assigned_zone": assigned_z,
                "centroid_utm_x": b_pt.x,
                "centroid_utm_y": b_pt.y,
                "geometry": blk.geometry,
            }
        )

    gdf_p_blocks = gpd.GeoDataFrame(block_records, geometry="geometry", crs="EPSG:32610")

    # 5. Compile Zone Polygon GeoDataFrame
    zone_rows = []
    recon_rows = []
    for z in zone_names:
        z_cfg = zone_configs[z]
        pop = zone_populations[z]
        veh = int(np.round(pop / occupancy))
        geom = zone_geoms[z]
        zone_rows.append(
            {
                "zone_name": z,
                "population": pop,
                "vehicles": veh,
                "block_count": zone_block_counts[z],
                "order_clock_pst": z_cfg["order_clock_pst"],
                "warning_clock_pst": z_cfg.get("warning_clock_pst", "08:01:00"),
                "notification_elapsed_min": z_cfg["notification_elapsed_min"],
                "departure_duration_min": z_cfg["departure_duration_min"],
                "centroid_utm_x": coords[zone_names.index(z)][0],
                "centroid_utm_y": coords[zone_names.index(z)][1],
                "geometry": geom,
            }
        )
        recon_rows.append(
            {
                "Zone": z,
                "Assigned Census Blocks": zone_block_counts[z],
                "Population (2010)": pop,
                "Vehicles (1.6 occ)": veh,
                "Order Clock (PST)": z_cfg["order_clock_pst"],
                "Elapsed Notif (min)": z_cfg["notification_elapsed_min"],
                "Duration (min)": z_cfg["departure_duration_min"],
                "Centroid UTM Easting": coords[zone_names.index(z)][0],
                "Centroid UTM Northing": coords[zone_names.index(z)][1],
                "Source Document": "NIST TN 2252 sup App C & Table 11",
            }
        )

    total_blocks = sum(r["Assigned Census Blocks"] for r in recon_rows)
    total_pop = sum(r["Population (2010)"] for r in recon_rows)
    total_veh = sum(r["Vehicles (1.6 occ)"] for r in recon_rows)

    recon_rows.append(
        {
            "Zone": "TOTAL (Town of Paradise)",
            "Assigned Census Blocks": total_blocks,
            "Population (2010)": total_pop,
            "Vehicles (1.6 occ)": total_veh,
            "Order Clock (PST)": "07:46 - 08:49",
            "Elapsed Notif (min)": "73 - 136 min",
            "Duration (min)": "60 - 90 min",
            "Centroid UTM Easting": float(np.mean(coords[:, 0])),
            "Centroid UTM Northing": float(np.mean(coords[:, 1])),
            "Source Document": "US Census 2010 Decennial SF1 Table P1 (26,218 exactly)",
        }
    )

    df_recon = pd.DataFrame(recon_rows)
    out_csv = Path("reports/paradise_14_zones_reconciliation.csv")
    df_recon.to_csv(out_csv, index=False)

    gdf_zones = gpd.GeoDataFrame(zone_rows, geometry="geometry", crs="EPSG:32610")
    gdf_zones.to_parquet("data/tiles/paradise_14_zones.parquet")

    return gdf_zones, gdf_p_blocks, df_recon


def plot_zones_overlay(gdf_zones: gpd.GeoDataFrame, gdf_blocks: gpd.GeoDataFrame) -> None:
    """Generate high-resolution cartographic map with zones, blocks, corridors, and exits."""
    fig, ax = plt.subplots(figsize=(12, 11), dpi=200)

    # Palette for 14 zones
    import matplotlib as mpl

    cmap = mpl.colormaps["tab20"]

    # 1. Plot Zone Polygons
    for idx, row in gdf_zones.iterrows():
        color = cmap(idx)
        gpd.GeoSeries([row.geometry]).plot(
            ax=ax, color=color, alpha=0.45, edgecolor="black", linewidth=1.2, zorder=2
        )
        c_x, c_y = row.centroid_utm_x, row.centroid_utm_y
        ax.annotate(
            f"{row.zone_name}\n({row.population:,} p / {row.vehicles:,} v)\nOrder {row.order_clock_pst}",
            xy=(c_x, c_y),
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.85, lw=0.7),
            zorder=6,
        )

    # 2. Plot Census Blocks
    gdf_blocks.plot(ax=ax, facecolor="none", edgecolor="gray", linewidth=0.35, alpha=0.6, zorder=3)

    # 3. Plot Road Network
    from wf052.routing.graph import load_camp_fire_road_graph

    graph = load_camp_fire_road_graph(full_detail=True)

    # Filter edges in Paradise bounding box
    minx, miny, maxx, maxy = gdf_zones.total_bounds
    pad = 1500.0

    primary_lines = []
    secondary_lines = []
    for e in graph.edges:
        u_pt = graph.nodes[e.u]
        v_pt = graph.nodes[e.v]
        if (minx - pad <= u_pt[0] <= maxx + pad) and (miny - pad <= u_pt[1] <= maxy + pad):
            line = shapely.geometry.LineString([u_pt, v_pt])
            hw = e.highway.lower()
            if any(k in hw for k in ["primary", "trunk", "motorway"]):
                primary_lines.append(line)
            elif any(k in hw for k in ["secondary", "tertiary"]):
                secondary_lines.append(line)

    if secondary_lines:
        gpd.GeoSeries(secondary_lines).plot(
            ax=ax, color="darkgoldenrod", linewidth=0.9, alpha=0.7, zorder=4
        )
    if primary_lines:
        gpd.GeoSeries(primary_lines).plot(
            ax=ax, color="darkred", linewidth=1.6, alpha=0.9, zorder=5
        )

    # 4. Annotate Key Egress Corridors & Boundary Destinations
    corridors = [
        ("Skyway (West/South to Chico)", 616000.0, 4404000.0, -25.0, 15.0),
        ("Clark Road (South to Oroville)", 621000.0, 4399000.0, 15.0, -20.0),
        ("Pentz Road (South to Hwy 70)", 622800.0, 4401000.0, 20.0, -15.0),
        ("Neal Road (Southwest to Hwy 99)", 615500.0, 4398500.0, -30.0, -20.0),
    ]
    for c_name, cx, cy, dx, dy in corridors:
        ax.annotate(
            c_name,
            xy=(cx, cy),
            xytext=(cx + dx * 60, cy + dy * 60),
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="navy",
            bbox=dict(boxstyle="square,pad=0.2", fc="linen", ec="navy", lw=1.0),
            arrowprops=dict(arrowstyle="->", color="navy", lw=1.5),
            zorder=7,
        )

    ax.set_xlim(minx - 1200, maxx + 1200)
    ax.set_ylim(miny - 1200, maxy + 1200)
    ax.set_title(
        "Town of Paradise Official Evacuation Zones (1-14) & Census 2010 Blocks\n"
        "Butte County GIS Evacuation Plan / NIST TN 2252 sup App C (Reconciled Population: 26,218)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("UTM Zone 10N Easting (m)", fontsize=9)
    ax.set_ylabel("UTM Zone 10N Northing (m)", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    out_png = Path("reports/paradise_zones_overlay.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved high-resolution zone overlay image to {out_png}")


def main():
    print("Building 14 official Paradise evacuation zone polygons from Census 2010 & NIST App C...")
    gdf_zones, gdf_blocks, df_recon = build_and_save_14_zones()
    print(f"Constructed {len(gdf_zones)} zones across {len(gdf_blocks)} census blocks.")
    print("\n" + "=" * 115)
    print("TOWN OF PARADISE 14 EVACUATION ZONES DEMOGRAPHIC RECONCILIATION TABLE")
    print("=" * 115)
    print(df_recon.to_string(index=False))

    print("\nGenerating cartographic overlay map...")
    plot_zones_overlay(gdf_zones, gdf_blocks)


if __name__ == "__main__":
    main()
