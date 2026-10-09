"""Deep analysis of all 28 GeoMAC perimeter timestamps."""

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


def main() -> None:
    data_path = Path("data/raw/camp_fire_2018_perimeter.geojson")
    with open(data_path) as f:
        data = json.load(f)

    # Camp fire ignition: 2018-11-08 06:30 PST = 2018-11-08 14:30 UTC
    ign_utc = datetime(2018, 11, 8, 14, 30, tzinfo=UTC)
    pst = timezone(timedelta(hours=-8))
    mst = timezone(timedelta(hours=-7))

    records = []
    for feat in data.get("features", []):
        p = feat.get("properties", {})
        if p.get("state") != "CA" or p.get("incidentname") != "CAMP":
            continue

        raw_perim_dt = p.get("perimeterdatetime")
        raw_datecurr = p.get("datecurrent")
        acres = p.get("gisacres", 0.0)
        ha = acres * 0.40468564224

        dt_perim_utc = datetime.fromtimestamp(raw_perim_dt / 1000.0, tz=UTC) if raw_perim_dt else None
        dt_curr_utc = datetime.fromtimestamp(raw_datecurr / 1000.0, tz=UTC) if raw_datecurr else None

        hours_since_ign = (dt_perim_utc - ign_utc).total_seconds() / 3600.0 if dt_perim_utc else None

        records.append({
            "objectid": p.get("OBJECTID"),
            "raw_perimeterdatetime": raw_perim_dt,
            "raw_datecurrent": raw_datecurr,
            "parsed_utc": dt_perim_utc.strftime("%Y-%m-%d %H:%M:%S UTC") if dt_perim_utc else "None",
            "parsed_pst": dt_perim_utc.astimezone(pst).strftime("%Y-%m-%d %H:%M:%S PST") if dt_perim_utc else "None",
            "parsed_mst": dt_perim_utc.astimezone(mst).strftime("%Y-%m-%d %H:%M:%S MST") if dt_perim_utc else "None",
            "curr_utc": dt_curr_utc.strftime("%Y-%m-%d %H:%M:%S UTC") if dt_curr_utc else "None",
            "curr_pst": dt_curr_utc.astimezone(pst).strftime("%Y-%m-%d %H:%M:%S PST") if dt_curr_utc else "None",
            "hours_since_ign": hours_since_ign,
            "gisacres": acres,
            "area_ha": ha,
            "mapmethod": p.get("mapmethod"),
            "comments": p.get("comments", "").strip(),
        })

    df = pd.DataFrame(records)
    # Sort by raw perimeterdatetime
    df = df.sort_values(by="raw_perimeterdatetime").reset_index(drop=True)

    print("=== All 28 GeoMAC Camp Fire Perimeters (Sorted by perimeterdatetime) ===")
    for idx, r in df.iterrows():
        print(
            f"#{idx+1:02d} [ID {r['objectid']:4d}] UTC: {r['parsed_utc']} | PST: {r['parsed_pst']} | "
            f"+{r['hours_since_ign']:6.2f}h | {r['gisacres']:9.1f} ac ({r['area_ha']:8.1f} ha) | "
            f"Method: {r['mapmethod']:14s} | Comments: {r['comments']}"
        )

    df.to_csv("reports/geomac_perimeters_audit_raw.csv", index=False)
    print("\nSaved to reports/geomac_perimeters_audit_raw.csv")

if __name__ == "__main__":
    main()
