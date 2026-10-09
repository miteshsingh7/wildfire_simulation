"""Parse and filter GeoMAC 2018 perimeters for California Camp Fire."""

import json
from pathlib import Path


def main() -> None:
    path = Path("data/raw/camp_fire_2018_perimeter.geojson")
    with open(path) as f:
        data = json.load(f)

    ca_camp = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        if props.get("state") == "CA" or "CAMP" in str(props.get("incidentname")).upper():
            print(f"Name: {props.get('incidentname')}, State: {props.get('state')}, Acres: {props.get('gisacres')}, Date: {props.get('perimeterdatetime')}")
            if props.get("state") == "CA":
                ca_camp.append(feat)

    print(f"Total matching CA: {len(ca_camp)}")

if __name__ == "__main__":
    main()
