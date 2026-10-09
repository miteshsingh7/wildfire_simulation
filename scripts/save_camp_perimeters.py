"""Save and inspect official GeoMAC perimeters for California Camp Fire (2018)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd


def main() -> None:
    path = Path("data/raw/camp_fire_2018_perimeter.geojson")
    with open(path) as f:
        data = json.load(f)

    ca_features = []
    rows = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        if props.get("state") == "CA" and props.get("incidentname") == "CAMP":
            ca_features.append(feat)
            ts_ms = props.get("perimeterdatetime")
            dt_str = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC") if ts_ms else "N/A"
            acres = props.get("gisacres", 0.0)
            ha = acres * 0.40468564224
            rows.append({
                "objectid": props.get("OBJECTID"),
                "datetime_utc": dt_str,
                "timestamp_ms": ts_ms,
                "gis_acres": acres,
                "area_ha": ha,
                "agency": props.get("agency"),
                "comments": props.get("comments"),
            })

    out_geojson = {
        "type": "FeatureCollection",
        "features": ca_features
    }
    out_path = Path("data/raw/geomac_camp_fire_2018_all_perimeters.geojson")
    with open(out_path, "w") as f:
        json.dump(out_geojson, f)

    gdf = gpd.read_file(out_path)
    gdf["datetime_utc"] = [r["datetime_utc"] for r in rows]
    gdf["area_ha"] = [r["area_ha"] for r in rows]
    gdf = gdf.sort_values(by="area_ha")
    print("=== GeoMAC Camp Fire (2018) Perimeters ===")
    for _, r in gdf.iterrows():
        print(f"Time: {r['datetime_utc']} | Acres: {r['gisacres']:.1f} | Ha: {r['area_ha']:.1f} | Agency: {r['agency']}")

    # Save sorted perimeters summary
    gdf[["datetime_utc", "gisacres", "area_ha", "agency", "mapmethod"]].to_csv("reports/geomac_perimeters_summary.csv", index=False)
    print("Saved summary to reports/geomac_perimeters_summary.csv")

if __name__ == "__main__":
    main()
