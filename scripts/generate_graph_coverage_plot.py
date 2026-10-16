"""Generate coverage plot of road graph extent vs Camp Fire domain bounding box."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

# Domain BBox from configs/phase0.yaml
DOMAIN_BBOX_WGS84 = {
    "west": -121.750163,
    "south": 39.650017,
    "east": -121.349998,
    "north": 39.949900,
}

def main():
    p_edges = Path("data/tiles/roads_edges_full.parquet")
    if not p_edges.exists():
        p_edges = Path("data/tiles/roads_edges.parquet")

    gdf_edges = gpd.read_parquet(p_edges)
    gdf_wgs84 = gdf_edges.to_crs(epsg=4326)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Plot sample of edges for fast rendering
    sample_edges = gdf_wgs84.sample(min(4000, len(gdf_wgs84)), random_state=42)
    sample_edges.plot(ax=ax, color="gray", linewidth=0.6, alpha=0.6, label="Road Edges (Full Detail)")

    # Plot domain bounding box
    d_box = box(
        DOMAIN_BBOX_WGS84["west"],
        DOMAIN_BBOX_WGS84["south"],
        DOMAIN_BBOX_WGS84["east"],
        DOMAIN_BBOX_WGS84["north"],
    )
    gpd.GeoSeries([d_box], crs="EPSG:4326").boundary.plot(
        ax=ax, color="red", linestyle="--", linewidth=2.0, label="Camp Fire Domain BBox"
    )

    # Plot graph extent bounding box
    gbounds = gdf_wgs84.total_bounds
    g_box = box(gbounds[0], gbounds[1], gbounds[2], gbounds[3])
    gpd.GeoSeries([g_box], crs="EPSG:4326").boundary.plot(
        ax=ax, color="blue", linestyle="-", linewidth=1.5, label="Graph Total Extent"
    )

    ax.set_title("Road Network Extent vs Camp Fire Domain Bounding Box", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude (deg WGS84)")
    ax.set_ylabel("Latitude (deg WGS84)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")

    out_path = Path("reports/graph_domain_coverage.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved coverage plot to {out_path}")
    print(f"Graph Bounds: [{gbounds[0]:.4f}, {gbounds[1]:.4f}, {gbounds[2]:.4f}, {gbounds[3]:.4f}]")
    print(f"Domain Bounds: [{DOMAIN_BBOX_WGS84['west']:.4f}, {DOMAIN_BBOX_WGS84['south']:.4f}, {DOMAIN_BBOX_WGS84['east']:.4f}, {DOMAIN_BBOX_WGS84['north']:.4f}]")

if __name__ == "__main__":
    main()
