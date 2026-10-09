"""Search FRAP or NIFC open data for 2018 Camp fire."""

import json
from pathlib import Path

import requests


def main() -> None:
    # Try NIFC / Interagency Fire Perimeters or CAL FIRE FRAP
    urls = [
        ("https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/California_Fire_Perimeters/FeatureServer/0/query",
         {"where": "YEAR_ = 2018 AND ALARM_DATE LIKE '%2018-11%'", "outFields": "FIRE_NAME,YEAR_,GIS_ACRES,ALARM_DATE,CONT_DATE", "f": "json"}),
        ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/Historic_Geomac_Perimeters_2018/FeatureServer/0/query",
         {"where": "incidentname LIKE '%CAMP%'", "outFields": "*", "f": "geojson"}),
        ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/InteragencyFirePerimeterHistory_All_Years_View/FeatureServer/0/query",
         {"where": "poly_IncidentName LIKE '%CAMP%' AND fireyear = 2018", "outFields": "*", "f": "geojson"}),
    ]
    for url, params in urls:
        print(f"Trying {url}...")
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
                features = data.get("features", [])
                print(f"  Returned {len(features)} features.")
                for feat in features[:3]:
                    props = feat.get("properties") or feat.get("attributes")
                    print(f"    Sample: {props}")
                if len(features) > 0 and "geometry" in features[0]:
                    out_path = Path("data/raw/camp_fire_2018_perimeter.geojson")
                    with open(out_path, "w") as f:
                        json.dump(data, f)
                    print(f"Saved to {out_path}")
                    return
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
