"""Precomputes offline data assets for the Streamlit/pydeck demo application.

Generates into out/demo/:
- perimeters/: Surrogate and Teacher v2 perimeters at +1, +2, +4, +6 h, plus GeoMAC 1052 (+11.4 h) in WGS84 (EPSG:4326).
- isochrones/: Hourly isochrone bands (1h..6h) for Surrogate and Teacher v2 in WGS84.
- hazards.geojson: Truth hazards (burnovers, roadblocks, abandoned vehicles) with onset elapsed minutes in WGS84.
- routes.json: Origin evacuation routes for static, A2, A3, and B-on-A3 in WGS84 path format.
- updates.geojson: 30-min update points and event metrics in WGS84.
- zones.geojson: Paradise 14 zone boundary polygons in WGS84.
- zones_summary.csv: Per-zone route change and margin summaries.
- metadata.json: Map viewport coordinates, bounds, and STAND-IN WEIGHTS disclaimer.
"""

import json
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize, shapes
import shapely.geometry
import shapely.ops
import torch

from evaluate_heldout_camp_fire import build_camp_fire_dataset_60m
from wf052.replay.harness import CausalReplayHarness
from wf052.replay.methods import (
    run_agent_simulation,
    run_static_baseline,
    sample_block_origins,
)
from wf052.replay.providers import (
    ObservationInflatedProvider,
    SurrogateFieldProvider,
    TeacherFieldProvider,
    TeacherV2FieldProvider,
)
from wf052.routing.graph import load_camp_fire_road_graph
from wf052.surrogate.adapter import TilePrediction, stitch_tiles_to_arrival_field
from wf052.surrogate.dataset import rebuild_spatial_inputs
from wf052.surrogate.model import WindConditionedUNet


def main() -> None:
    print("=" * 80)
    print("STAND-IN WEIGHTS (Phase 3A Smoke Checkpoint)")
    print("BUILDING DEMO ASSETS FOR app/demo.py")
    print("=" * 80)

    out_dir = Path("out/demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    perim_dir = out_dir / "perimeters"
    perim_dir.mkdir(parents=True, exist_ok=True)
    iso_dir = out_dir / "isochrones"
    iso_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ds_camp, cfg, transform_60m = build_camp_fire_dataset_60m(device)
    h_full, w_full = ds_camp.fbfm_arr.shape

    tf_base = TeacherFieldProvider().get_field(0.0)
    gx = tf_base["x"].values
    gy = tf_base["y"].values
    origin_x = float(gx[0])
    origin_y = float(gy[0])

    road_graph = load_camp_fire_road_graph(
        full_detail=True,
        grid_x_coords=gx,
        grid_y_coords=gy,
    )
    destinations = set(road_graph.boundary_destinations.values())
    origins = sample_block_origins(road_graph=road_graph, num_samples=200, seed=42)

    harness_base = CausalReplayHarness(road_graph=road_graph, provider=TeacherFieldProvider(), latency_min=0.0)
    truth_hazards = harness_base.truth_hazards

    # 1. Observed Perimeter (GeoMAC 1052 at +11.4 h)
    print("1. Exporting GeoMAC ID 1052 observed perimeter (WGS84)...")
    gdf_geomac = gpd.read_file("data/raw/geomac_camp_fire_2018_all_perimeters.geojson")
    sub_1052 = gdf_geomac[gdf_geomac["OBJECTID"] == 1052].copy()
    sub_1052_wgs84 = sub_1052.to_crs("EPSG:4326")
    sub_1052_wgs84.to_file(perim_dir / "observed_perimeter_11_4h.geojson", driver="GeoJSON")

    # 2. Zones (Paradise 14 Zones in WGS84)
    print("2. Exporting Paradise 14 evacuation zones (WGS84)...")
    zones_df = gpd.read_parquet("data/tiles/paradise_14_zones.parquet")
    zones_wgs84 = zones_df.to_crs("EPSG:4326")
    zones_wgs84.to_file(out_dir / "zones.geojson", driver="GeoJSON")

    # 3. Truth Hazards in WGS84
    print("3. Exporting Truth Hazards (WGS84)...")
    haz_records = []
    for h in truth_hazards:
        haz_records.append({
            "hazard_id": h.hazard_id,
            "hazard_type": h.hazard_type,
            "onset_elapsed_min": h.onset_elapsed_min,
            "description": h.description,
            "source_ref": h.source_ref,
            "geometry": h.geometry_utm,
        })
    gdf_haz = gpd.GeoDataFrame(haz_records, crs="EPSG:32610").to_crs("EPSG:4326")
    gdf_haz.to_file(out_dir / "hazards.geojson", driver="GeoJSON")

    # 4. Teacher v2 Field & Perimeters / Isochrones
    print("4. Computing Teacher v2 arrival field & perimeters/isochrones...")
    prov_tv2 = TeacherV2FieldProvider(assimilate=False)
    af_tv2 = prov_tv2.get_field(360.0)
    arr_tv2 = af_tv2["arrival_q50"].values

    hours = [1.0, 2.0, 4.0, 6.0]
    for h_val in hours:
        t_limit = h_val * 60.0
        mask_t = (np.isfinite(arr_tv2)) & (arr_tv2 <= t_limit)
        mask_uint8 = mask_t.astype(np.uint8)
        poly_geoms = []
        for s_geom, val in shapes(mask_uint8, mask=mask_uint8, transform=transform_60m):
            if val == 1:
                poly_geoms.append(shapely.geometry.shape(s_geom))
        unary_p = shapely.ops.unary_union(poly_geoms) if poly_geoms else shapely.geometry.Polygon()
        gdf_p = gpd.GeoDataFrame([{"horizon_hours": h_val, "horizon_min": t_limit}], geometry=[unary_p], crs="EPSG:32610")
        gdf_p.to_crs("EPSG:4326").to_file(perim_dir / f"teacher_perimeter_{int(h_val)}h.geojson", driver="GeoJSON")

    # Teacher Isochrones (1h to 6h bands)
    iso_records_tv2 = []
    for h_idx in range(1, 7):
        t_low = (h_idx - 1) * 60.0
        t_high = h_idx * 60.0
        mask_iso = (arr_tv2 > t_low) & (arr_tv2 <= t_high)
        mask_uint8 = mask_iso.astype(np.uint8)
        poly_geoms = []
        for s_geom, val in shapes(mask_uint8, mask=mask_uint8, transform=transform_60m):
            if val == 1:
                poly_geoms.append(shapely.geometry.shape(s_geom))
        if poly_geoms:
            u_p = shapely.ops.unary_union(poly_geoms)
            iso_records_tv2.append({"band_label": f"{h_idx-1}h-{h_idx}h", "t_low": t_low, "t_high": t_high, "geometry": u_p})
    if iso_records_tv2:
        gpd.GeoDataFrame(iso_records_tv2, crs="EPSG:32610").to_crs("EPSG:4326").to_file(iso_dir / "teacher_isochrones.geojson", driver="GeoJSON")

    # 5. Surrogate Model & Perimeters / Isochrones
    print("5. Generating Surrogate arrival field & perimeters/isochrones...")
    model = WindConditionedUNet(in_spatial_channels=13, base_channels=32, emb_dim=64, max_horizon_min=720.0)
    ckpt = torch.load("checkpoints/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    weather_df = pd.read_parquet("data/tiles/hrrr_weather_hourly.parquet")
    u_vals = weather_df["u10_ms"].values[:12].astype(np.float32)
    v_vals = weather_df["v10_ms"].values[:12].astype(np.float32)
    wind_series = np.column_stack([u_vals, v_vals])
    wind_batch = torch.from_numpy(wind_series).unsqueeze(0).to(device)

    r_origins = [0, 154, 308]
    c_origins = [0, 153, 306]
    tile_preds = []
    for r0 in r_origins:
        for c0 in c_origins:
            rel_ign_r = 232 - r0
            rel_ign_c = 452 - c0
            tile_tensor = rebuild_spatial_inputs(
                fire_id="camp_fire",
                crop_origin=(r0, c0),
                ignition_cell=(rel_ign_r, rel_ign_c),
                fires_base_dir="out/camp_fire_60m",
                tile_shape=(256, 256),
            )
            with torch.no_grad():
                out = model(tile_tensor.unsqueeze(0).to(device), wind_batch)
                q_t = out.quantiles[0].cpu().numpy()
                p_t = torch.sigmoid(out.burn_logits[0, 0]).cpu().numpy()
            tile_preds.append(TilePrediction(quantiles=q_t, burn_prob=p_t, r0=r0, c0=c0))

    full_af = stitch_tiles_to_arrival_field(
        tile_predictions=tile_preds,
        domain_shape=(h_full, w_full),
        origin_x=origin_x,
        origin_y=origin_y,
        resolution_m=60.0,
        crs="EPSG:32610",
        ignition_time="2018-11-08T14:33:00Z",
        p_thr=0.1,
    )
    arr_surr = full_af["arrival_q50"].values

    # Convert surrogate perimeters to WGS84
    for h_val in hours:
        src_path = perim_dir / f"surrogate_perimeter_{int(h_val)}h.geojson"
        if src_path.exists():
            gdf_surr_p = gpd.read_file(src_path).to_crs("EPSG:4326")
            gdf_surr_p.to_file(src_path, driver="GeoJSON")

    # Surrogate Isochrones (1h to 6h bands)
    iso_records_surr = []
    for h_idx in range(1, 7):
        t_low = (h_idx - 1) * 60.0
        t_high = h_idx * 60.0
        mask_iso = (arr_surr > t_low) & (arr_surr <= t_high)
        mask_uint8 = mask_iso.astype(np.uint8)
        poly_geoms = []
        for s_geom, val in shapes(mask_uint8, mask=mask_uint8, transform=transform_60m):
            if val == 1:
                poly_geoms.append(shapely.geometry.shape(s_geom))
        if poly_geoms:
            u_p = shapely.ops.unary_union(poly_geoms)
            iso_records_surr.append({"band_label": f"{h_idx-1}h-{h_idx}h", "t_low": t_low, "t_high": t_high, "geometry": u_p})
    if iso_records_surr:
        gpd.GeoDataFrame(iso_records_surr, crs="EPSG:32610").to_crs("EPSG:4326").to_file(iso_dir / "surrogate_isochrones.geojson", driver="GeoJSON")

    # 6. Update Markers in WGS84
    print("6. Exporting 30-min update markers (WGS84)...")
    upd_df = pd.read_csv("reports/step5a_update_loop_log.csv")
    center_utm_x, center_utm_y = 622000.0, 4402000.0
    upd_records = []
    for _, row in upd_df.iterrows():
        upd_records.append({
            "update_step_min": row["update_step_min"],
            "hour_index": int(row["hour_index"]),
            "frac_cells_changed": row["frac_cells_q50_changed_gt_10m"],
            "origins_route_changed": int(row["origins_route_changed"]),
            "surrogate_inf_ms": row["surrogate_inf_ms"],
            "geometry": shapely.geometry.Point(center_utm_x, center_utm_y),
        })
    gpd.GeoDataFrame(upd_records, crs="EPSG:32610").to_crs("EPSG:4326").to_file(out_dir / "updates.geojson", driver="GeoJSON")

    # 7. Routes across Static, A2, A3, and B-on-A3 in WGS84
    print("7. Exporting Evacuation Routes for Demo (WGS84)...")
    # Load 30-min update fields dictionary for surrogate
    update_steps_min = list(np.arange(0.0, 360.0 + 1e-6, 30.0))
    surrogate_update_fields = {}
    u_all = weather_df["u10_ms"].values.astype(np.float32)
    v_all = weather_df["v10_ms"].values.astype(np.float32)

    for t_upd in update_steps_min:
        h_idx = min(len(u_all) - 1, int(t_upd // 60.0))
        u_sl = u_all[h_idx : h_idx + 12]
        v_sl = v_all[h_idx : h_idx + 12]
        if len(u_sl) < 12:
            pad = 12 - len(u_sl)
            u_sl = np.pad(u_sl, (0, pad), mode="edge")
            v_sl = np.pad(v_sl, (0, pad), mode="edge")
        w_b = torch.from_numpy(np.column_stack([u_sl, v_sl])).unsqueeze(0).to(device)

        t_preds = []
        for r0 in r_origins:
            for c0 in c_origins:
                rel_ign_r = 232 - r0
                rel_ign_c = 452 - c0
                tt = rebuild_spatial_inputs(
                    fire_id="camp_fire",
                    crop_origin=(r0, c0),
                    ignition_cell=(rel_ign_r, rel_ign_c),
                    fires_base_dir="out/camp_fire_60m",
                    tile_shape=(256, 256),
                )
                with torch.no_grad():
                    out = model(tt.unsqueeze(0).to(device), w_b)
                    q_t = out.quantiles[0].cpu().numpy()
                    p_t = torch.sigmoid(out.burn_logits[0, 0]).cpu().numpy()
                t_preds.append(TilePrediction(quantiles=q_t, burn_prob=p_t, r0=r0, c0=c0))

        surrogate_update_fields[t_upd] = stitch_tiles_to_arrival_field(
            tile_predictions=t_preds,
            domain_shape=(h_full, w_full),
            origin_x=origin_x,
            origin_y=origin_y,
            resolution_m=60.0,
            crs="EPSG:32610",
            ignition_time="2018-11-08T14:33:00Z",
            p_thr=0.1,
        )

    # Simulate methods
    res_static, _ = run_static_baseline(origins, road_graph, destinations, truth_hazards)

    prov_a2_assim = TeacherV2FieldProvider(assimilate=True)
    harness_a2_assim = CausalReplayHarness(road_graph=road_graph, provider=prov_a2_assim, latency_min=0.0)
    res_a2_assim, _ = run_agent_simulation("agent_a2_assim", origins, road_graph, destinations, truth_hazards, harness_a2_assim)

    prov_a2_pure = TeacherV2FieldProvider(assimilate=False)
    harness_a2_pure = CausalReplayHarness(road_graph=road_graph, provider=prov_a2_pure, latency_min=0.0)
    res_a2_pure, _ = run_agent_simulation("agent_a2_pure", origins, road_graph, destinations, truth_hazards, harness_a2_pure)

    prov_a3 = SurrogateFieldProvider(update_fields=surrogate_update_fields)
    harness_a3 = CausalReplayHarness(road_graph=road_graph, provider=prov_a3, latency_min=0.0)
    res_a3, _ = run_agent_simulation("agent_a3", origins, road_graph, destinations, truth_hazards, harness_a3)

    prov_b_a3 = ObservationInflatedProvider(base_teacher_provider=prov_a3, r_spot_km=4.0)
    harness_b_a3 = CausalReplayHarness(road_graph=road_graph, provider=prov_b_a3, latency_min=0.0)
    res_b_a3, _ = run_agent_simulation("agent_b_on_a3", origins, road_graph, destinations, truth_hazards, harness_b_a3)

    import pyproj
    transformer = pyproj.Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
    edge_geom_map = {(ge.u, ge.v, ge.key): ge.geometry for ge in road_graph.edges}
    node_coords_map = {n: (geom.x, geom.y) for n, geom in zip(road_graph.nodes, road_graph.nodes_gdf.geometry)}

    def route_to_wgs84_coords(route) -> list[list[float]]:
        if route is None or not route.edges:
            return []
        coords_utm = []
        for e in route.edges:
            geom = edge_geom_map.get((e.u, e.v, e.key))
            if geom is not None and hasattr(geom, "coords"):
                coords_utm.extend(list(geom.coords))
            else:
                p_u = node_coords_map.get(e.u)
                p_v = node_coords_map.get(e.v)
                if p_u and p_v:
                    coords_utm.extend([p_u, p_v])
        # Unique preserving order
        dedup_utm = []
        prev = None
        for pt in coords_utm:
            if pt != prev:
                dedup_utm.append(pt)
                prev = pt
        coords_wgs84 = []
        for x, y in dedup_utm:
            lon, lat = transformer.transform(x, y)
            coords_wgs84.append([round(lon, 6), round(lat, 6)])
        return coords_wgs84

    demo_routes = {}
    for orig in origins:
        o_id = orig.origin_id
        demo_routes[str(o_id)] = {
            "origin_id": o_id,
            "assigned_zone": orig.assigned_zone,
            "departure_time_min": orig.departure_time_min,
            "static": {
                "hazard_miss": bool(res_static[origins.index(orig)].hazard_miss),
                "margin_min": float(res_static[origins.index(orig)].min_margin_min),
                "trapped": res_static[origins.index(orig)].route is None or not res_static[origins.index(orig)].route.edges,
                "path": route_to_wgs84_coords(res_static[origins.index(orig)].route),
            },
            "agent_a2": {
                "hazard_miss": bool(res_a2_assim[origins.index(orig)].hazard_miss),
                "margin_min": float(res_a2_assim[origins.index(orig)].min_margin_min),
                "trapped": res_a2_assim[origins.index(orig)].route is None or not res_a2_assim[origins.index(orig)].route.edges,
                "path": route_to_wgs84_coords(res_a2_assim[origins.index(orig)].route),
            },
            "agent_a2_assim": {
                "hazard_miss": bool(res_a2_assim[origins.index(orig)].hazard_miss),
                "margin_min": float(res_a2_assim[origins.index(orig)].min_margin_min),
                "trapped": res_a2_assim[origins.index(orig)].route is None or not res_a2_assim[origins.index(orig)].route.edges,
                "path": route_to_wgs84_coords(res_a2_assim[origins.index(orig)].route),
            },
            "agent_a2_pure": {
                "hazard_miss": bool(res_a2_pure[origins.index(orig)].hazard_miss),
                "margin_min": float(res_a2_pure[origins.index(orig)].min_margin_min),
                "trapped": res_a2_pure[origins.index(orig)].route is None or not res_a2_pure[origins.index(orig)].route.edges,
                "path": route_to_wgs84_coords(res_a2_pure[origins.index(orig)].route),
            },
            "agent_a3": {
                "hazard_miss": bool(res_a3[origins.index(orig)].hazard_miss),
                "margin_min": float(res_a3[origins.index(orig)].min_margin_min),
                "trapped": res_a3[origins.index(orig)].route is None or not res_a3[origins.index(orig)].route.edges,
                "path": route_to_wgs84_coords(res_a3[origins.index(orig)].route),
            },
            "agent_b_on_a3": {
                "hazard_miss": bool(res_b_a3[origins.index(orig)].hazard_miss),
                "margin_min": float(res_b_a3[origins.index(orig)].min_margin_min),
                "trapped": res_b_a3[origins.index(orig)].route is None or not res_b_a3[origins.index(orig)].route.edges,
                "path": route_to_wgs84_coords(res_b_a3[origins.index(orig)].route),
            },
        }

    with open(out_dir / "routes.json", "w") as f:
        json.dump(demo_routes, f)
    print(f"Exported routes for {len(demo_routes)} origins to out/demo/routes.json")

    # 8. Copy per-zone metrics summary
    if Path("reports/step5d_per_zone_metrics.csv").exists():
        df_z = pd.read_csv("reports/step5d_per_zone_metrics.csv")
        df_z.to_csv(out_dir / "zones_summary.csv", index=False)

    # 9. Metadata
    metadata = {
        "title": "Camp Fire Evacuation Replay Demo",
        "weights_status": "STAND-IN WEIGHTS (Phase 3A Smoke Checkpoint)",
        "development_case": "Camp Fire 2018 (NEVER in surrogate training)",
        "center_lon": -121.60,
        "center_lat": 39.759,
        "zoom": 11.2,
        "origins_count": 200,
        "hazards_count": len(truth_hazards),
        "horizon_min": 360.0,
        "update_interval_min": 30.0,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("DEMO ASSETS BUILD COMPLETE in out/demo/")
    print("=" * 80)


if __name__ == "__main__":
    main()
