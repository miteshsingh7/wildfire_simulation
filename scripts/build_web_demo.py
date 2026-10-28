#!/usr/bin/env python3
"""Phase 5A-UI: Build standalone offline web dashboard.

Reads out/demo/ assets, simplifies geometries (Douglas-Peucker tolerance 0.0001 deg,
coordinates rounded to 5 decimals), inlines all simulation and routing data into
web/dist/index.html, and reports bundle sizes before and after.
"""

import base64
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any
import pandas as pd
import shapely.geometry
import shapely.ops


TOLERANCE_DEG = 0.0001  # ~11.1 meters at latitude 39.8° N
COORD_PRECISION = 5     # 5 decimal places (~1.1 meter resolution)


def round_coord(pt: tuple[float, float]) -> list[float]:
    """Round (x, y) coordinate pair to 5 decimal places."""
    return [round(float(pt[0]), COORD_PRECISION), round(float(pt[1]), COORD_PRECISION)]


def simplify_geometry(geom_dict: dict[str, Any], tolerance: float = TOLERANCE_DEG) -> dict[str, Any]:
    """Simplify shapely geometry via Douglas-Peucker and round coordinates."""
    if not geom_dict:
        return geom_dict
    try:
        shp = shapely.geometry.shape(geom_dict)
        if shp.is_empty:
            return geom_dict
        if shp.geom_type in ("Polygon", "MultiPolygon", "LineString", "MultiLineString"):
            shp = shp.simplify(tolerance, preserve_topology=True)

        def _round_geom(g: Any) -> Any:
            if g.is_empty:
                return g
            if g.geom_type == "Point":
                return shapely.geometry.Point(round_coord(g.coords[0]))
            elif g.geom_type == "LineString":
                return shapely.geometry.LineString([round_coord(c) for c in g.coords])
            elif g.geom_type == "MultiLineString":
                return shapely.geometry.MultiLineString([[round_coord(c) for c in ls.coords] for ls in g.geoms])
            elif g.geom_type == "Polygon":
                ext = [round_coord(c) for c in g.exterior.coords]
                rings = [[round_coord(c) for c in ring.coords] for ring in g.interiors]
                return shapely.geometry.Polygon(ext, rings)
            elif g.geom_type == "MultiPolygon":
                return shapely.geometry.MultiPolygon([_round_geom(p) for p in g.geoms])
            return g

        rounded_shp = _round_geom(shp)
        return shapely.geometry.mapping(rounded_shp)
    except Exception:
        return geom_dict


def compute_path_length_km(coords: list[list[float]]) -> float:
    """Compute approximate path length in kilometers using spherical equirectangular distance."""
    if not coords or len(coords) < 2:
        return 0.0
    total_m = 0.0
    lat_mid = math.radians(39.76)
    m_per_deg_lat = 111132.954
    m_per_deg_lon = 111412.84 * math.cos(lat_mid)
    for i in range(len(coords) - 1):
        dx = (coords[i + 1][0] - coords[i][0]) * m_per_deg_lon
        dy = (coords[i + 1][1] - coords[i][1]) * m_per_deg_lat
        total_m += math.hypot(dx, dy)
    return round(total_m / 1000.0, 2)


def simplify_geojson(geojson_dict: dict[str, Any], tolerance: float = TOLERANCE_DEG) -> dict[str, Any]:
    """Simplify all features in a GeoJSON FeatureCollection."""
    features_out = []
    for feat in geojson_dict.get("features", []):
        f = dict(feat)
        if "geometry" in f and f["geometry"]:
            f["geometry"] = simplify_geometry(f["geometry"], tolerance=tolerance)
        features_out.append(f)
    return {
        "type": "FeatureCollection",
        "features": features_out,
    }


def simplify_routes(routes_dict: dict[str, Any], tolerance: float = TOLERANCE_DEG) -> dict[str, Any]:
    """Simplify polyline paths and calculate path distances in routes.json."""
    routes_out = {}
    for orig_id, orig_data in routes_dict.items():
        o = dict(orig_data)
        for m_key in ("static", "agent_a2", "agent_a2_assim", "agent_a2_pure", "agent_a3", "agent_b_on_a3"):
            m = o.get(m_key)
            if m and isinstance(m, dict):
                m_copy = dict(m)
                path = m.get("path", [])
                if path and len(path) > 1:
                    ls = shapely.geometry.LineString(path)
                    ls_sim = ls.simplify(tolerance, preserve_topology=True)
                    sim_coords = [round_coord(c) for c in ls_sim.coords]
                    m_copy["path"] = sim_coords
                    m_copy["distance_km"] = compute_path_length_km(sim_coords)
                else:
                    m_copy["distance_km"] = 0.0
                # Sanitize margin_min for JSON encoding (inf -> null, or handle in JS)
                margin = m_copy.get("margin_min")
                if margin is not None:
                    if math.isinf(margin):
                        m_copy["margin_min"] = None if margin > 0 else -9999.0
                        m_copy["margin_state"] = "no_hazard" if margin > 0 else "trapped"
                    elif math.isnan(margin):
                        m_copy["margin_min"] = None
                        m_copy["margin_state"] = "unknown"
                    else:
                        m_copy["margin_min"] = round(float(margin), 1)
                        m_copy["margin_state"] = "positive" if margin >= 0 else "negative"
                else:
                    m_copy["margin_state"] = "no_hazard"
                o[m_key] = m_copy
        routes_out[orig_id] = o
    return routes_out


def build() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    demo_dir = repo_root / "out" / "demo"
    web_dir = repo_root / "web"
    dist_dir = web_dir / "dist"
    dist_data_dir = dist_dir / "data"

    assert demo_dir.exists(), f"Missing {demo_dir}!"
    dist_dir.mkdir(parents=True, exist_ok=True)
    dist_data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 5A-UI: BUILDING STANDALONE OFFLINE DASHBOARD")
    print(f"Douglas-Peucker simplification tolerance: {TOLERANCE_DEG} degrees (~11.1 m)")
    print(f"Coordinate rounding precision: {COORD_PRECISION} decimal places (~1.1 m)")
    print("=" * 80)

    # Calculate raw uncompressed size of out/demo/
    raw_size_bytes = sum(f.stat().st_size for f in demo_dir.glob("**/*") if f.is_file())
    print(f"Raw out/demo/ assets size: {raw_size_bytes / (1024 * 1024):.2f} MB ({raw_size_bytes:,} bytes)")

    # 1. Load and simplify assets
    print("\n[1/5] Processing and simplifying geometries...")
    with open(demo_dir / "metadata.json") as f:
        meta = json.load(f)

    with open(demo_dir / "hazards.geojson") as f:
        hazards = simplify_geojson(json.load(f), tolerance=TOLERANCE_DEG)

    with open(demo_dir / "updates.geojson") as f:
        updates = simplify_geojson(json.load(f), tolerance=TOLERANCE_DEG)

    with open(demo_dir / "zones.geojson") as f:
        zones = simplify_geojson(json.load(f), tolerance=TOLERANCE_DEG)

    # Zones summary CSV: load 14 zone rows and 3 corridor rows for A3 stand-in field
    step5d_csv = repo_root / "reports" / "step5d_per_zone_metrics.csv"
    if step5d_csv.exists():
        df_zones = pd.read_csv(step5d_csv)
        if "method" in df_zones.columns:
            df_zones = df_zones[df_zones["method"] == "A3 (Surrogate)"]
    else:
        df_zones = pd.read_csv(demo_dir / "zones_summary.csv")
        if "method" in df_zones.columns:
            df_zones = df_zones[df_zones["method"] == "A3 (Surrogate)"]
    zones_summary = df_zones.to_dict(orient="records")

    # Isochrones
    with open(demo_dir / "isochrones" / "teacher_isochrones.geojson") as f:
        teacher_iso = simplify_geojson(json.load(f), tolerance=TOLERANCE_DEG)
    with open(demo_dir / "isochrones" / "surrogate_isochrones.geojson") as f:
        surrogate_iso = simplify_geojson(json.load(f), tolerance=TOLERANCE_DEG)

    # Perimeters
    with open(demo_dir / "perimeters" / "observed_perimeter_11_4h.geojson") as f:
        obs_perim = simplify_geojson(json.load(f), tolerance=TOLERANCE_DEG)

    teacher_perims = {}
    surrogate_perims = {}
    for h in (1, 2, 4, 6):
        tp_path = demo_dir / "perimeters" / f"teacher_perimeter_{h}h.geojson"
        sp_path = demo_dir / "perimeters" / f"surrogate_perimeter_{h}h.geojson"
        if tp_path.exists():
            with open(tp_path) as f:
                teacher_perims[str(h)] = simplify_geojson(json.load(f), tolerance=TOLERANCE_DEG)
        if sp_path.exists():
            with open(sp_path) as f:
                surrogate_perims[str(h)] = simplify_geojson(json.load(f), tolerance=TOLERANCE_DEG)

    # Routes
    with open(demo_dir / "routes.json") as f:
        routes_raw = json.load(f)
    routes = simplify_routes(routes_raw, tolerance=TOLERANCE_DEG)

    # Bundle payload
    demo_bundle = {
        "metadata": meta,
        "hazards": hazards,
        "updates": updates,
        "zones": zones,
        "zones_summary": zones_summary,
        "isochrones": {
            "teacher": teacher_iso,
            "surrogate": surrogate_iso,
        },
        "perimeters": {
            "observed_11_4h": obs_perim,
            "teacher": teacher_perims,
            "surrogate": surrogate_perims,
        },
        "routes": routes,
    }

    # 2. Write individual data files to web/dist/data/
    print("\n[2/5] Writing modular data files to web/dist/data/...")
    with open(dist_data_dir / "metadata.json", "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    with open(dist_data_dir / "hazards.geojson", "w") as f:
        json.dump(hazards, f, separators=(",", ":"))
    with open(dist_data_dir / "updates.geojson", "w") as f:
        json.dump(updates, f, separators=(",", ":"))
    with open(dist_data_dir / "zones.geojson", "w") as f:
        json.dump(zones, f, separators=(",", ":"))
    with open(dist_data_dir / "zones_summary.json", "w") as f:
        json.dump(zones_summary, f, separators=(",", ":"))
    with open(dist_data_dir / "teacher_isochrones.geojson", "w") as f:
        json.dump(teacher_iso, f, separators=(",", ":"))
    with open(dist_data_dir / "surrogate_isochrones.geojson", "w") as f:
        json.dump(surrogate_iso, f, separators=(",", ":"))
    with open(dist_data_dir / "observed_perimeter_11_4h.geojson", "w") as f:
        json.dump(obs_perim, f, separators=(",", ":"))
    with open(dist_data_dir / "routes.json", "w") as f:
        json.dump(routes, f, separators=(",", ":"))

    # Copy fonts to web/dist/fonts/
    dist_fonts_dir = dist_dir / "fonts"
    dist_fonts_dir.mkdir(parents=True, exist_ok=True)
    if (web_dir / "fonts").exists():
        for font_file in (web_dir / "fonts").glob("*.woff2"):
            shutil.copy2(font_file, dist_fonts_dir / font_file.name)

    # 3. Read CSS and inlining fonts as base64
    print("\n[3/5] Bundling fonts, CSS, and JS...")
    with open(web_dir / "style.css") as f:
        css_content = f.read()

    # Inline fonts as base64 into CSS for 100% offline file:// execution
    if (web_dir / "fonts").exists():
        for font_file in (web_dir / "fonts").glob("*.woff2"):
            with open(font_file, "rb") as ff:
                b64_font = base64.b64encode(ff.read()).decode("ascii")
            css_content = css_content.replace(
                f'url("fonts/{font_file.name}")',
                f'url("data:font/woff2;base64,{b64_font}")',
            )

    # Read JS
    with open(web_dir / "app.js") as f:
        js_content = f.read()

    # Read HTML template
    with open(web_dir / "index.html") as f:
        html_template = f.read()

    # 4. Construct standalone inlined HTML
    print("\n[4/5] Assembling self-contained web/dist/index.html...")
    data_json_str = json.dumps(demo_bundle, separators=(",", ":"))

    # Inlined script block containing DEMO_DATA and app.js
    inlined_script = f"""
    <script>
      window.DEMO_DATA = {data_json_str};
    </script>
    <script>
      {js_content}
    </script>
    """

    inlined_style = f"""
    <style>
      {css_content}
    </style>
    """

    final_html = html_template.replace("<!-- INLINE_CSS -->", inlined_style)
    final_html = final_html.replace("<!-- INLINE_JS -->", inlined_script)

    out_index = dist_dir / "index.html"
    with open(out_index, "w", encoding="utf-8") as f:
        f.write(final_html)

    final_size_bytes = out_index.stat().st_size
    print(f"\n[5/5] Final Verification:")
    print(f"  Input raw size:  {raw_size_bytes / (1024 * 1024):.2f} MB ({raw_size_bytes:,} bytes)")
    print(f"  Final HTML size: {final_size_bytes / (1024 * 1024):.2f} MB ({final_size_bytes:,} bytes)")
    print(f"  Target size:     < 10.00 MB")

    assert final_size_bytes < 10 * 1024 * 1024, f"Final HTML exceeds 10MB! ({final_size_bytes} bytes)"
    print(f"  SUCCESS: Standalone web/dist/index.html generated ({final_size_bytes / (1024 * 1024):.2f} MB)!")


if __name__ == "__main__":
    build()
