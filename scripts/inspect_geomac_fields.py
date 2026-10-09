"""Inspect all fields of the 28 GeoMAC perimeters."""

import json
from pathlib import Path


def main() -> None:
    data_path = Path("data/raw/camp_fire_2018_perimeter.geojson")
    with open(data_path) as f:
        data = json.load(f)

    ca_perims = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        if props.get("state") == "CA" and props.get("incidentname") == "CAMP":
            ca_perims.append(props)

    print(f"Total perimeters found: {len(ca_perims)}")
    # Print all keys present in the first record
    print("Keys:", sorted(list(ca_perims[0].keys())))

    # Print relevant columns for each record
    for idx, p in enumerate(ca_perims):
        print(f"\n--- Perimeter #{idx + 1} (OBJECTID: {p.get('OBJECTID')}) ---")
        for k, v in sorted(p.items()):
            if "date" in k.lower() or "time" in k.lower() or "acre" in k.lower() or "comment" in k.lower() or "method" in k.lower():
                print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
