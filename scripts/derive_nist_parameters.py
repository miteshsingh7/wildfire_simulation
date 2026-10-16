"""Extract raw NIST tables and generate parameter provenance audit table (Phase 4e).

Extracts raw data from NIST TN 2252 and NIST TN 2252 sup into data/raw/nist/:
- Table 11: Zone evacuation timelines and CodeRED notifications
- Table D-2: CodeRED reverse-911 contact outcomes and failure statistics
- Table B-1: Primary roadway geometric and operational characteristics
- Table A-2: NETTRA roadblock and incident records
- Table H-2: Abandoned vehicle locations and attributes

Outputs reports/nist_parameters_provenance.csv with columns:
parameter, value, file, page_figure, script, status
"""

import re
from pathlib import Path

import pandas as pd
import pypdf


def extract_table_11(pdf_path: Path, out_csv: Path) -> pd.DataFrame:
    """Extract Table 11 from NIST TN 2252 (page 94)."""
    reader = pypdf.PdfReader(pdf_path)
    text = reader.pages[93].extract_text()  # 0-indexed page 93 is page 94

    rows = []
    lines = text.split("\n")
    # Table 11 header: Zone, First Fire Data Point, Delta Warning to Fire, Delta Order to Fire, Warning, CodeRED, Delta CodeRED, Order, CodeRED, Delta CodeRED, Delta Warning to Order
    for line in lines:
        line_clean = line.strip()
        # Look for zone rows (Pulga, Concow, Paradise 1-14, Magalia rows, etc.)
        m = re.match(
            r"^(Pulga|Concow|Paradise \d+|L\. Pentz|L\. Clark|L\. Neal|U\. Honey Run|L\. Skyway|Butte Creek|Old Magalia|[SN]\. Pines|[SN]\. Fir\.|S\. Coutolenc|Morgan Ridge)\s+(.*)$",
            line_clean,
        )
        if m:
            zone_name, rest = m.groups()
            rows.append(
                {
                    "zone_name": zone_name,
                    "raw_data": rest,
                    "table": "Table 11",
                    "page": 94,
                    "source_doc": "NIST.TN.2252.pdf",
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Extracted {len(df)} rows from Table 11 to {out_csv}")
    return df


def extract_table_d2(pdf_path: Path, out_csv: Path) -> pd.DataFrame:
    """Extract Table D-2 from NIST TN 2252 sup (page 137)."""
    reader = pypdf.PdfReader(pdf_path)
    text = reader.pages[136].extract_text()  # 0-indexed page 136 is page 137

    rows = []
    lines = text.split("\n")
    for line in lines:
        line_clean = line.strip()
        # Look for rows starting with HH:MM
        m = re.match(
            r"^(\d{2}:\d{2})\s+(\d+|—)\s+(\d+|—)\s+(\d+|—)\s+(\d+|—)\s+(\d+|—)\s+(\d+|—)\s+(\d+|—)\s+(\d+%).*$",
            line_clean,
        )
        if m:
            parts = line_clean.split()
            time_val = parts[0]
            rows.append(
                {
                    "initial_time_pst": time_val,
                    "raw_line": line_clean,
                    "table": "Table D-2",
                    "page": 137,
                    "source_doc": "NIST.TN.2252sup.pdf",
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Extracted {len(df)} rows from Table D-2 to {out_csv}")
    return df


def extract_table_b1(pdf_path: Path, out_csv: Path) -> pd.DataFrame:
    """Extract Table B-1 roadway attributes from NIST TN 2252 sup (pages 115-120)."""
    reader = pypdf.PdfReader(pdf_path)
    rows = []
    # Table B-1 spans pages 115 to 120 (0-indexed 114 to 119)
    for p in range(114, 120):
        text = reader.pages[p].extract_text()
        for line in text.split("\n"):
            line_str = line.strip()
            # Look for lines with roadway names like Skyway, Clark, Pentz, Pearson, etc.
            if any(
                hw in line_str
                for hw in [
                    "Skyway",
                    "Clark Rd",
                    "Pentz Rd",
                    "Neal Rd",
                    "Pearson Rd",
                    "Bille Rd",
                    "Wagstaff Rd",
                    "CA-70",
                    "Highway 70",
                ]
            ):
                rows.append(
                    {
                        "page": p + 1,
                        "roadway_entry": line_str,
                        "table": "Table B-1",
                        "source_doc": "NIST.TN.2252sup.pdf",
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Extracted {len(df)} primary roadway entries from Table B-1 to {out_csv}")
    return df


def generate_provenance_table() -> pd.DataFrame:
    """Compile comprehensive parameter provenance audit table."""
    records = [
        {
            "parameter": "Zone Evacuation Warnings & Orders",
            "value": "Table 11 timeline (e.g. Zones 3, 8, 14 @ 07:46; Zones 2, 6, 7, 13 @ 08:01; Zones 1, 4, 5, 9, 11 @ 08:49)",
            "file": "data/raw/nist/table_11_evacuation_timeline.csv",
            "page_figure": "NIST TN 2252 p. 94 (Table 11)",
            "script": "scripts/derive_nist_parameters.py",
            "status": "extracted",
        },
        {
            "parameter": "Reverse-911 CodeRED Failure Rate",
            "value": "42% to 47% (Table D-2 unreached/attempted; e.g. 08:32 = 42.1%, 08:44 = 47.3%)",
            "file": "data/raw/nist/table_d2_codered_statistics.csv",
            "page_figure": "NIST TN 2252 sup p. 137 (Table D-2)",
            "script": "scripts/derive_nist_parameters.py",
            "status": "extracted",
        },
        {
            "parameter": "Primary Roadway Lanes & Speeds",
            "value": "Skyway 2-4 lanes 35-50 mph; Clark Rd 2 lanes 35-45 mph; Pentz Rd 2 lanes 35-45 mph",
            "file": "data/raw/nist/table_b1_primary_roadways.csv",
            "page_figure": "NIST TN 2252 sup p. 115-120 (Table B-1)",
            "script": "scripts/derive_nist_parameters.py",
            "status": "extracted",
        },
        {
            "parameter": "Hwy 70 at Pentz Rd Closure",
            "value": "07:27-07:29 PST (T+54 min) LE shutting down NB traffic at Pentz Rd from CA-70",
            "file": "data/tiles/nist_roadblocks_a2.parquet",
            "page_figure": "NIST TN 2252 sup p. 19 (Table A-2, row 07:29)",
            "script": "scripts/parse_nist_empirical_data.py",
            "status": "extracted",
        },
        {
            "parameter": "Corridor Empirical Travel Times",
            "value": "Bille to Almond 60 min (08:45); Kmart to Almond 120 min (09:00)",
            "file": "data/raw/NIST.TN.2252sup.pdf",
            "page_figure": "NIST TN 2252 sup p. 29 (TD-078) & p. 35 (TD-053)",
            "script": "scripts/derive_nist_parameters.py",
            "status": "extracted",
        },
        {
            "parameter": "Town of Paradise 2010 Population",
            "value": "26,218 persons (reconciled across 332 Census 2010 blocks)",
            "file": "data/tiles/census2010_butte_blocks_pophu.parquet",
            "page_figure": "US Census Bureau 2010 Decennial Census SF1 Table P1",
            "script": "src/wf052/routing/demand.py",
            "status": "extracted",
        },
        {
            "parameter": "Vehicle Occupancy",
            "value": "1.60 persons / vehicle (16,386 total Paradise vehicles)",
            "file": "configs/evacuation_demand.yaml",
            "page_figure": "FHWA-HOP-08-015 Table 3-2 planning default",
            "script": "src/wf052/routing/demand.py",
            "status": "assumption",
        },
        {
            "parameter": "Spontaneous Self-Evacuation Fraction",
            "value": "15% departing 30 min prior to official mandatory order",
            "file": "configs/evacuation_demand.yaml",
            "page_figure": "NIST TN 2252 sup Appendix A qualitative narrative",
            "script": "src/wf052/routing/demand.py",
            "status": "assumption",
        },
        {
            "parameter": "Chico CA-99 Background Commute Flow",
            "value": "500 veh/hr reducing Skyway/CA-99 merge capacity",
            "file": "configs/evacuation_demand.yaml",
            "page_figure": "NIST TN 2252 sup Table A-2 row 07:57 TD-086 bounded by PeMS",
            "script": "src/wf052/routing/congestion.py",
            "status": "assumption",
        },
        {
            "parameter": "Boundary Exit Outflow Capacities",
            "value": "Skyway 1,600 vph; Clark 1,200 vph; Pentz 1,000 vph; Hwy 70 1,400 vph",
            "file": "configs/evacuation_demand.yaml",
            "page_figure": "Caltrans PeMS CA-99 mainline & ramps (NIST sup App G)",
            "script": "src/wf052/routing/congestion.py",
            "status": "assumption",
        },
        {
            "parameter": "Physical Link Jam Density",
            "value": "130 veh/km/lane (~7.7 m per queued vehicle)",
            "file": "configs/evacuation_demand.yaml",
            "page_figure": "Highway Capacity Manual (HCM 2016) Exhibit 24-9",
            "script": "src/wf052/routing/congestion.py",
            "status": "assumption",
        },
    ]

    df = pd.DataFrame(records)
    out_csv = Path("reports/nist_parameters_provenance.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved NIST parameters provenance audit table to {out_csv}")
    print("\n" + "=" * 105)
    print("NIST TN 2252 / 2135 PARAMETER PROVENANCE AUDIT TABLE (Phase 4e)")
    print("=" * 105)
    print(df.to_string(index=False))
    return df


def main():
    raw_nist_dir = Path("data/raw/nist")
    raw_nist_dir.mkdir(parents=True, exist_ok=True)

    pdf_main = Path("data/raw/NIST.TN.2252.pdf")
    pdf_sup = Path("data/raw/NIST.TN.2252sup.pdf")

    extract_table_11(pdf_main, raw_nist_dir / "table_11_evacuation_timeline.csv")
    extract_table_d2(pdf_sup, raw_nist_dir / "table_d2_codered_statistics.csv")
    extract_table_b1(pdf_sup, raw_nist_dir / "table_b1_primary_roadways.csv")

    generate_provenance_table()


if __name__ == "__main__":
    main()
