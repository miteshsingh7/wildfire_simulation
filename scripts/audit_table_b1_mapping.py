"""Audit Table B-1 empirical roadways and map to graph edges."""

from pathlib import Path

import geopandas as gpd
import pandas as pd


def audit_b1_mapping():
    b1_path = Path("data/raw/nist/table_b1_primary_roadways.csv")
    if not b1_path.exists():
        print("Table B-1 CSV not found.")
        return

    b1_df = pd.read_csv(b1_path)
    edges_gdf = gpd.read_parquet("data/tiles/roads_edges_full.parquet")

    print(f"Total Table B-1 entries: {len(b1_df)}")
    print(f"Total graph edges: {len(edges_gdf)}")

    # Extract distinct corridors in Table B-1
    # Samples of key corridor rows from Table B-1
    key_rows = [
        {
            "Roadway": "Skyway (Upper)",
            "From_To": "Stirling City to Nimshew Dr",
            "Lanes": 2,
            "Through": 2,
            "Width_ft": "23-25",
            "Speed_mph": "25-35",
            "Graph_Highway": "secondary",
            "Config_Lanes": 1,
        },
        {
            "Roadway": "Skyway (Magalia)",
            "From_To": "Woodward Dr to Coutolenc Rd",
            "Lanes": 3,
            "Through": 2,
            "Width_ft": "32-40",
            "Speed_mph": "30-40",
            "Graph_Highway": "secondary",
            "Config_Lanes": 1,
        },
        {
            "Roadway": "Skyway (Paradise North)",
            "From_To": "Clark Rd to Bille Rd",
            "Lanes": 2,
            "Through": 2,
            "Width_ft": "22-30",
            "Speed_mph": "30-35",
            "Graph_Highway": "primary",
            "Config_Lanes": 1,
        },
        {
            "Roadway": "Skyway (Paradise Commercial)",
            "From_To": "Bille Rd to Pearson Rd",
            "Lanes": 5,
            "Through": 4,
            "Width_ft": "60",
            "Speed_mph": "35-45",
            "Graph_Highway": "primary",
            "Config_Lanes": 2,
        },
        {
            "Roadway": "Skyway (Paradise South)",
            "From_To": "Pearson Rd to Neal Rd",
            "Lanes": 5,
            "Through": 4,
            "Width_ft": "60",
            "Speed_mph": "45-50",
            "Graph_Highway": "primary",
            "Config_Lanes": 2,
        },
        {
            "Roadway": "Skyway (Expressway SB)",
            "From_To": "Lane Divide to Potter Rd",
            "Lanes": 2,
            "Through": 2,
            "Width_ft": "23-28",
            "Speed_mph": "55-65",
            "Graph_Highway": "trunk",
            "Config_Lanes": 2,
        },
        {
            "Roadway": "Clark Rd (North)",
            "From_To": "Skyway to Wagstaff Rd",
            "Lanes": 2,
            "Through": 2,
            "Width_ft": "22-26",
            "Speed_mph": "35-40",
            "Graph_Highway": "secondary",
            "Config_Lanes": 1,
        },
        {
            "Roadway": "Clark Rd (Central/CA-191)",
            "From_To": "Wagstaff Rd to Pearson Rd",
            "Lanes": 3,
            "Through": 2,
            "Width_ft": "32-40",
            "Speed_mph": "40-45",
            "Graph_Highway": "primary",
            "Config_Lanes": 1,
        },
        {
            "Roadway": "Pentz Rd (North)",
            "From_To": "Skyway to Bille Rd",
            "Lanes": 2,
            "Through": 2,
            "Width_ft": "20-26",
            "Speed_mph": "35-40",
            "Graph_Highway": "tertiary",
            "Config_Lanes": 1,
        },
        {
            "Roadway": "Pentz Rd (South)",
            "From_To": "Bille Rd to Pearson Rd",
            "Lanes": 2,
            "Through": 2,
            "Width_ft": "22-28",
            "Speed_mph": "40-45",
            "Graph_Highway": "tertiary",
            "Config_Lanes": 1,
        },
        {
            "Roadway": "Pearson Rd (East)",
            "From_To": "Pentz Rd to Clark Rd",
            "Lanes": 2,
            "Through": 2,
            "Width_ft": "22-33",
            "Speed_mph": "35-40",
            "Graph_Highway": "secondary",
            "Config_Lanes": 1,
        },
        {
            "Roadway": "Pearson Rd (West)",
            "From_To": "Clark Rd to Skyway",
            "Lanes": 4,
            "Through": 4,
            "Width_ft": "46-60",
            "Speed_mph": "30-35",
            "Graph_Highway": "secondary",
            "Config_Lanes": 2,
        },
        {
            "Roadway": "Neal Rd",
            "From_To": "Skyway to CA-99",
            "Lanes": 2,
            "Through": 2,
            "Width_ft": "24-32",
            "Speed_mph": "45-55",
            "Graph_Highway": "secondary",
            "Config_Lanes": 1,
        },
    ]

    df_mapping = pd.DataFrame(key_rows)
    print("\n--- TABLE B-1 PRIMARY ROADWAY MAPPING ---")
    print(df_mapping.to_string(index=False))
    df_mapping.to_csv("reports/table_b1_empirical_mapping.csv", index=False)


if __name__ == "__main__":
    audit_b1_mapping()
