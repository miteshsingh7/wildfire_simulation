"""Parse reports/gofer_large_fires_raw.js into a clean dataframe of California fires."""

import re
from pathlib import Path

import pandas as pd


def extract_val(body: str, key: str, default: str = "") -> str:
    m = re.search(rf"{key}:\s*['\"]([^'\"]+)['\"]", body)
    if m:
        return m.group(1)
    m_num = re.search(rf"{key}:\s*([0-9\.\-]+)", body)
    if m_num:
        return m_num.group(1)
    return default


def main() -> None:
    path = Path("reports/gofer_large_fires_raw.js")
    with open(path) as f:
        text = f.read()

    # Pattern matching JavaScript objects:
    # '2019_Kincade': { ... }
    fire_blocks = re.findall(r"'(\d{4}_[^']+)':\s*\{([^}]+)\}", text)
    print(f"Total fire blocks found: {len(fire_blocks)}")

    records = []
    for fire_id, body in fire_blocks:
        fire_name = extract_val(body, "name", fire_id)
        year = extract_val(body, "year", fire_id[:4])
        start_date = extract_val(body, "startDate")
        end_date = extract_val(body, "endDate")
        lon = extract_val(body, "lon")
        lat = extract_val(body, "lat")
        acres = extract_val(body, "acres", "0")
        frap_acres = extract_val(body, "frap_acres", "0")

        records.append({
            "fire_id": fire_id,
            "fire_name": fire_name,
            "year": year,
            "start_date": start_date,
            "end_date": end_date,
            "lon": float(lon) if lon else None,
            "lat": float(lat) if lat else None,
            "acres": float(acres) if acres else (float(frap_acres) if frap_acres else 0.0),
        })

    df = pd.DataFrame(records)
    df["area_ha"] = df["acres"] * 0.40468564224
    df = df.sort_values(by=["year", "acres"], ascending=[True, False]).reset_index(drop=True)

    print("=== All Fires in GOFER (2019-2021) ===")
    for _, r in df.iterrows():
        print(
            f"[{r['year']}] {r['fire_name']:20s} ({r['fire_id']:22s}) | "
            f"Dates: {r['start_date']} to {r['end_date']} | "
            f"Coords: ({r['lat']}, {r['lon']}) | {r['acres']:9.1f} ac ({r['area_ha']:8.1f} ha)"
        )

    out_csv = Path("reports/gofer_all_fires.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved all fires to {out_csv}")

if __name__ == "__main__":
    main()
