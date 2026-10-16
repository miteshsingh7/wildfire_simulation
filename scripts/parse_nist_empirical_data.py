"""Parse NIST TN 2252 sup and TN 2135 empirical field observations into GeoParquet (Phase 4d).

Extracts and persists:
1. Appendix F / Appendix B: 23 life-threatening burnover events with geolocations and time intervals.
2. Appendix H (Table H-2): Documented abandoned vehicles in/along roadways (235 vehicles).
3. Appendix A (Table A-2): Road-block and road-closure incident rows with coordinates and timestamps.
"""

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pypdf
import shapely.geometry


def parse_abandoned_vehicles() -> gpd.GeoDataFrame:
    pdf_path = Path("data/raw/NIST.TN.2252sup.pdf")
    reader = pypdf.PdfReader(pdf_path)

    # Table H-2 spans pages 171 to 176 (0-indexed 170 to 175)
    records = []
    for p in range(170, 176):
        text = reader.pages[p].extract_text()
        for line in text.split("\n"):
            m = re.match(
                r"^(AV\d+)\s+(.+?)\s+([NSEW]B|WB|EB|SB|NB)\s+([A-Z0-9]+)\s+([A-Z0-9]+)\s+([A-Z0-9]+)\s+([A-Z0-9]+)\s+([YN\s]?)\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)",
                line.strip(),
            )
            if m:
                av_id, street, direction, lane, flow, color, vtype, burned, lat, lon = m.groups()
                records.append(
                    {
                        "av_id": av_id,
                        "street": street.strip(),
                        "direction": direction,
                        "lane": lane,
                        "flow": flow,
                        "color": color,
                        "vehicle_type": vtype,
                        "burned": (burned.strip() == "Y"),
                        "latitude": float(lat),
                        "longitude": float(lon),
                        "source": "NIST TN 2252 sup Table H-2",
                    }
                )

    df = pd.DataFrame(records)
    geometry = [
        shapely.geometry.Point(xy) for xy in zip(df["longitude"], df["latitude"], strict=True)
    ]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def parse_23_burnovers() -> gpd.GeoDataFrame:
    burnovers = []

    # 1. Events 20-23 from NIST TN 2252 sup (Appendix F, pages 151-162)
    reader_sup = pypdf.PdfReader("data/raw/NIST.TN.2252sup.pdf")
    for p in range(150, 163):
        text = reader_sup.pages[p].extract_text()
        m_title = re.search(
            r"(\d+)\.\s+([A-Za-z0-9\s/]+)\nDate/Time:\s*(.+?)\nLocation:\s*(.+?)\nCoordinates:\s*\[(-?\d+\.\d+),\s*(-?\d+\.\d+)\]",
            text,
        )
        if m_title:
            num, name, dt, loc, lat, lon = m_title.groups()
            burnovers.append(
                {
                    "burnover_id": int(num),
                    "name": name.strip(),
                    "datetime_str": dt.strip(),
                    "location": loc.strip(),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "source_doc": "NIST TN 2252 sup Appendix F",
                }
            )

    # 2. Events 1-19 from NIST TN 2135 (Appendix B, pages 197-260)
    reader_2135 = pypdf.PdfReader("data/raw/NIST.TN.2135.pdf")
    for p in range(197, 260):
        text = reader_2135.pages[p].extract_text()
        m_title = re.search(
            r"(\d+)\.\s+([A-Za-z0-9\s/(),\-]+)\nDate/Time:\s*(.+?)\nLocation:\s*(.+?)\nCoordinates:\s*\[(-?\d+\.\d+),\s*(-?\d+\.\d+)\]",
            text,
        )
        if m_title:
            num, name, dt, loc, lat, lon = m_title.groups()
            b_id = int(num)
            if b_id not in [b["burnover_id"] for b in burnovers]:
                burnovers.append(
                    {
                        "burnover_id": b_id,
                        "name": name.strip(),
                        "datetime_str": dt.strip(),
                        "location": loc.strip(),
                        "latitude": float(lat),
                        "longitude": float(lon),
                        "source_doc": "NIST TN 2135 Appendix B",
                    }
                )

    df = pd.DataFrame(burnovers).sort_values("burnover_id").reset_index(drop=True)
    geometry = [
        shapely.geometry.Point(xy) for xy in zip(df["longitude"], df["latitude"], strict=True)
    ]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def parse_roadblocks_table_a2() -> gpd.GeoDataFrame:
    pdf_path = Path("data/raw/NIST.TN.2252sup.pdf")
    reader = pypdf.PdfReader(pdf_path)

    # Table A-2 spans pages 15 to 112 (0-indexed 14 to 111)
    roadblock_records = []

    # Header format: Date, Time, Obs Window, Time Source, Lat, Lon, Location Source, NETTRA Observations, Rd Block, AV, Traffic, Notif, Evac, Rescue, TRA, Source #, Info Source
    for p in range(14, 112):
        text = reader.pages[p].extract_text()
        lines = text.split("\n")
        for line in lines:
            line_str = line.strip()
            # Match rows starting with 11/8 or 11/9 and a time HH:MM
            # Example: "11/8 07:29 0 Radio Log 39.649847 -121.559463 Radio Log LE is shutting down traffic at Pentz Rd [northbound from CA-70] ×"
            if re.match(r"^11/[89]\s+\d{2}:\d{2}", line_str):
                # Check for road block indication
                is_roadblock = False
                if (
                    "roadblock" in line_str.lower()
                    or "shutting down traffic" in line_str.lower()
                    or "blocked" in line_str.lower()
                    or "closed" in line_str.lower()
                ):
                    is_roadblock = True
                elif "Rd Block" in line_str:
                    is_roadblock = True

                # Check if 'x' appears in the line position corresponding to Rd Block
                # In Table A-2, "×" marks categories. If line contains "×" and description relates to roadblock or first category
                if "×" in line_str and any(
                    kw in line_str.lower()
                    for kw in [
                        "block",
                        "close",
                        "shut",
                        "impassable",
                        "lines down",
                        "powerline",
                        "turn around",
                        "stuck",
                    ]
                ):
                    is_roadblock = True

                if is_roadblock:
                    # Extract coordinates if present
                    coord_match = re.search(r"(-?\d{2}\.\d{4,7})\s+(-?121\.\d{4,7})", line_str)
                    time_match = re.search(r"(11/[89])\s+(\d{2}:\d{2})", line_str)
                    if time_match:
                        date_str, time_str = time_match.groups()
                        lat = float(coord_match.group(1)) if coord_match else None
                        lon = float(coord_match.group(2)) if coord_match else None

                        # Find source id (TD-xxx, VTD-xx, 911-xxx, AVL, Radio Log)
                        source_match = re.search(
                            r"(TD-\d+|VTD-\d+|911-[\d\-]+|Radio Log|PPD|BCSO)", line_str
                        )
                        source_id = source_match.group(1) if source_match else "Table A-2"

                        roadblock_records.append(
                            {
                                "date": date_str,
                                "time_pst": time_str,
                                "datetime_utc": f"2018-{date_str.replace('/', '-')[-2:]}T{int(time_str[:2]) + 8:02d}:{time_str[3:]}:00Z",
                                "latitude": lat,
                                "longitude": lon,
                                "description": line_str,
                                "source_id": source_id,
                                "table_ref": f"NIST TN 2252 sup Table A-2 (p. {p + 1})",
                            }
                        )

    df = pd.DataFrame(roadblock_records)
    # Filter rows that have coordinates
    df_coords = df.dropna(subset=["latitude", "longitude"]).copy()
    geometry = [
        shapely.geometry.Point(xy)
        for xy in zip(df_coords["longitude"], df_coords["latitude"], strict=True)
    ]
    gdf = gpd.GeoDataFrame(df_coords, geometry=geometry, crs="EPSG:4326")
    return gdf


def main():
    p_out = Path("data/tiles")
    p_out.mkdir(parents=True, exist_ok=True)

    print("Parsing Appendix H (Table H-2) Abandoned Vehicles...")
    gdf_av = parse_abandoned_vehicles()
    out_av = p_out / "nist_abandoned_vehicles_h2.parquet"
    gdf_av.to_parquet(out_av)
    print(f"Saved {len(gdf_av)} abandoned vehicles to {out_av}")

    print("\nParsing Appendix F & B (23 Burnovers)...")
    gdf_bo = parse_23_burnovers()
    out_bo = p_out / "nist_burnovers_23.parquet"
    gdf_bo.to_parquet(out_bo)
    print(f"Saved {len(gdf_bo)} burnovers to {out_bo}")

    print("\nParsing Appendix A (Table A-2) Road Blocks...")
    gdf_rb = parse_roadblocks_table_a2()
    out_rb = p_out / "nist_roadblocks_a2.parquet"
    gdf_rb.to_parquet(out_rb)
    print(f"Saved {len(gdf_rb)} road-block incidents to {out_rb}")


if __name__ == "__main__":
    main()
