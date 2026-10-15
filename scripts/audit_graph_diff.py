"""Audit the exact difference between simplified and full-detail road graphs.

Lists every edge present in the simplified graph but missing in the full-detail
graph (by OSM id), grouped by highway tag, and explicitly checks:
- CA 70
- Skyway
- Clark Rd
- Pentz Rd
- Neal Rd
- Link ramps
"""

import ast

import geopandas as gpd
import pandas as pd


def extract_osmids(df: gpd.GeoDataFrame) -> dict:
    mapping = {}
    for idx, val in df["osmid"].items():
        ids = set()
        if isinstance(val, str):
            if val.startswith("[") and val.endswith("]"):
                try:
                    for x in ast.literal_eval(val):
                        ids.add(str(x))
                except Exception:
                    ids.add(val)
            elif "," in val:
                for x in val.split(","):
                    ids.add(x.strip())
            else:
                ids.add(val)
        elif isinstance(val, (list, tuple)):
            for x in val:
                ids.add(str(x))
        elif val is not None:
            ids.add(str(val))
        mapping[idx] = ids
    return mapping


def run_diff(
    s_edges_path: str = "data/tiles/roads_edges.parquet",
    f_edges_path: str = "data/tiles/roads_edges_full.parquet",
) -> pd.DataFrame:
    print(f"Reading simplified: {s_edges_path} and full: {f_edges_path}")
    s_edges = gpd.read_parquet(s_edges_path)
    f_edges = gpd.read_parquet(f_edges_path)

    s_map = extract_osmids(s_edges)
    f_map = extract_osmids(f_edges)
    f_all_ids = set().union(*f_map.values())

    missing_edges = []
    for idx, ids in s_map.items():
        if not ids.issubset(f_all_ids):
            row = s_edges.loc[idx]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            missing_ids = ids - f_all_ids
            missing_edges.append(
                {
                    "u": idx[0],
                    "v": idx[1],
                    "key": idx[2],
                    "missing_osm_ids": sorted(list(missing_ids)),
                    "highway": row.get("highway"),
                    "name": str(row.get("name")),
                    "length_m": row.get("length"),
                }
            )

    df_diff = pd.DataFrame(missing_edges)
    print(f"Total simplified edges missing in full-detail: {len(df_diff)}")
    print("\n--- Missing Edges Grouped by Highway Tag ---")
    if len(df_diff) > 0:
        print(df_diff["highway"].value_counts())
    else:
        print("  None! Full-detail graph is a strict superset.")

    print("\n--- Explicit Corridor Checks ---")
    corridors = [
        ("CA 70", ["70", "state highway 70"]),
        ("Skyway", ["skyway"]),
        ("Clark Rd", ["clark"]),
        ("Pentz Rd", ["pentz"]),
        ("Neal Rd", ["neal"]),
    ]
    for name, queries in corridors:
        if len(df_diff) > 0:
            mask = (
                df_diff["name"]
                .str.lower()
                .apply(lambda s, q_list=queries: any(q in s for q in q_list))
            )
            m = df_diff[mask]
            print(f"  {name:<12}: {len(m):>4} missing edges")
            if len(m) > 0:
                for _, r in m.head(3).iterrows():
                    print(
                        f"     edge=({r['u']},{r['v']}) | osm_ids={r['missing_osm_ids']} | name={r['name']} | hw={r['highway']}"
                    )
        else:
            print(f"  {name:<12}:    0 missing edges")

    print("\n--- Link Ramps Check (motorway_link, trunk_link, primary_link, etc.) ---")
    if len(df_diff) > 0:
        links = df_diff[df_diff["highway"].astype(str).str.contains("link")]
        print(f"  Link ramps: {len(links)} missing edges")
        if len(links) > 0:
            for _, r in links.head(3).iterrows():
                print(
                    f"     edge=({r['u']},{r['v']}) | osm_ids={r['missing_osm_ids']} | name={r['name']} | hw={r['highway']}"
                )
    else:
        print("  Link ramps:    0 missing edges")

    return df_diff


if __name__ == "__main__":
    run_diff()
