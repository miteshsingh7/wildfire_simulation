"""Phase 6D: Close-Out of Replay and Teacher v2 Attribution Audit Script.

Computes:
STEP 2: Replay Robustness:
  - 2a: Teacher seed sensitivity (5 seeds from Arm c: 389, 117, 304, 100, 236).
        Combined misses, trapped count, abandoned vehicles met per routed origin (mean, median, p90).
  - 2b: Offset sensitivity (offsets 0, 4, 8, 12 min).
        Combined misses, trapped count, abandoned vehicles met per routed origin (mean, median, p90).

STEP 3: Metric Cleanup:
  - Trapped origins (out of 200).
  - Burnover encounters among routed origins, unique burnovers met (out of 23).
  - Roadblock encounters (labelled OBSERVED-INPUT).
  - Abandoned vehicles met per routed origin (mean, median, p90), origin-level >=1, unique met (out of 235).
  - Cluster bootstrap vs Static AND A2 vs A (corridors: Clark_Rd, Skyway, Pentz_Hwy70; K=3 clusters).
  - Plain origin-level bootstrap beside cluster bootstrap with interval width comparison.

STEP 4: Latency per actual re-plan event in Agent B replay:
  - Provider update + router + validator.
  - Event count, mean, p50, p95.
"""

import copy
import hashlib
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
import torch
import xarray as xr
from rasterio.features import rasterize

from wf052.replay.contracts import MethodMetrics, ReplayOriginResult, TruthHazard
from wf052.replay.harness import CausalReplayHarness
from wf052.replay.methods import (
    EvacuationOrigin,
    _find_vehicle_position_on_route,
    run_agent_simulation,
    run_reactive_baseline,
    run_static_baseline,
    sample_block_origins,
)
from wf052.replay.metrics import (
    check_route_hazard_miss_by_type,
    compute_warning_lead_time,
)
from wf052.replay.providers import (
    FieldProvider,
    ObservationInflatedProvider,
    TeacherFieldProvider,
    TeacherV2FieldProvider,
)
from wf052.routing.contracts import EvacuationRoute, validate_arrival_field
from wf052.routing.graph import RoadGraph, load_camp_fire_road_graph
from wf052.routing.router import TimeDependentRouter
from wf052.routing.validator import IndependentRouteValidator
from wf052.sim.ember_kernel import EmberParams, simulate_teacher_with_embers_60m
from wf052.sim.teacher_calibrator import TeacherParams


def count_abandoned_vehicles_for_route(
    route: EvacuationRoute | None,
    hazards: Sequence[TruthHazard],
    road_graph: RoadGraph,
    spatial_radius_m: float = 120.0,
) -> tuple[int, set[str]]:
    """Count how many abandoned vehicle hazards an evacuation route encountered at or after onset."""
    if route is None or not route.edges:
        return 0, set()

    av_hazards = [h for h in hazards if h.hazard_type == "abandoned_vehicle"]
    edge_map = {(e.u, e.v, e.key): e for e in road_graph.edges}
    met_ids = set()

    for re in route.edges:
        ed = edge_map.get((re.u, re.v, re.key))
        geom = ed.geometry if ed is not None else None

        for h in av_hazards:
            if re.exit_time_min < h.onset_elapsed_min:
                continue

            if geom is not None and not geom.is_empty:
                dist = float(geom.distance(h.geometry_utm))
            else:
                u_coord = road_graph.nodes.get(re.u)
                v_coord = road_graph.nodes.get(re.v)
                dist = float(shapely.geometry.LineString([u_coord, v_coord]).distance(h.geometry_utm)) if (u_coord and v_coord) else float("inf")

            if dist <= spatial_radius_m:
                met_ids.add(h.hazard_id)

    return len(met_ids), met_ids


def count_burnovers_for_route(
    route: EvacuationRoute | None,
    hazards: Sequence[TruthHazard],
    road_graph: RoadGraph,
    spatial_radius_m: float = 120.0,
) -> tuple[int, set[str]]:
    """Count burnover hazards encountered by a routed origin."""
    if route is None or not route.edges:
        return 0, set()

    bo_hazards = [h for h in hazards if h.hazard_type == "burnover"]
    edge_map = {(e.u, e.v, e.key): e for e in road_graph.edges}
    met_ids = set()

    for re in route.edges:
        ed = edge_map.get((re.u, re.v, re.key))
        geom = ed.geometry if ed is not None else None

        for h in bo_hazards:
            if re.exit_time_min < h.onset_elapsed_min:
                continue

            if geom is not None and not geom.is_empty:
                dist = float(geom.distance(h.geometry_utm))
            else:
                u_coord = road_graph.nodes.get(re.u)
                v_coord = road_graph.nodes.get(re.v)
                dist = float(shapely.geometry.LineString([u_coord, v_coord]).distance(h.geometry_utm)) if (u_coord and v_coord) else float("inf")

            if dist <= spatial_radius_m:
                met_ids.add(h.hazard_id)

    return len(met_ids), met_ids


def count_roadblocks_for_route(
    route: EvacuationRoute | None,
    hazards: Sequence[TruthHazard],
    road_graph: RoadGraph,
    spatial_radius_m: float = 120.0,
) -> tuple[int, set[str]]:
    """Count roadblock hazards encountered by a routed origin."""
    if route is None or not route.edges:
        return 0, set()

    rb_hazards = [h for h in hazards if h.hazard_type == "roadblock"]
    edge_map = {(e.u, e.v, e.key): e for e in road_graph.edges}
    met_ids = set()

    for re in route.edges:
        ed = edge_map.get((re.u, re.v, re.key))
        geom = ed.geometry if ed is not None else None

        for h in rb_hazards:
            if re.exit_time_min < h.onset_elapsed_min:
                continue

            if geom is not None and not geom.is_empty:
                dist = float(geom.distance(h.geometry_utm))
            else:
                u_coord = road_graph.nodes.get(re.u)
                v_coord = road_graph.nodes.get(re.v)
                dist = float(shapely.geometry.LineString([u_coord, v_coord]).distance(h.geometry_utm)) if (u_coord and v_coord) else float("inf")

            if dist <= spatial_radius_m:
                met_ids.add(h.hazard_id)

    return len(met_ids), met_ids


class DynamicTeacherV2Provider(FieldProvider):
    """Teacher v2 provider initialized directly from simulated arrival raster."""

    def __init__(self, arr_min: np.ndarray, gx: np.ndarray, gy: np.ndarray, patch_radius_m: float = 120.0) -> None:
        self.arr_min = arr_min
        self.gx = gx
        self.gy = gy
        self.patch_radius_m = patch_radius_m

        arr_clean = np.where(np.isfinite(arr_min) & (arr_min >= 0.0), arr_min, np.inf)
        self._cached_base_field = xr.Dataset(
            data_vars={
                "arrival_q10": (["y", "x"], arr_clean),
                "arrival_q50": (["y", "x"], arr_clean),
                "arrival_q90": (["y", "x"], arr_clean),
            },
            coords={"x": gx, "y": gy},
            attrs={"provider": "teacher_v2_dynamic", "crs": "EPSG:32610", "ignition_time": "2018-11-08T14:33:00Z"},
        )
        validate_arrival_field(self._cached_base_field)

    def get_field(self, current_step_min: float, visible_observations: gpd.GeoDataFrame | None = None) -> xr.Dataset:
        if visible_observations is None or len(visible_observations) == 0:
            return self._cached_base_field

        gx, gy = self.gx, self.gy
        ny, nx = len(gy), len(gx)
        dx = float(gx[1] - gx[0])
        dy = float(gy[0] - gy[1])

        base_arr = self._cached_base_field["arrival_q50"].values.copy()
        assim_arr = base_arr.copy()

        for _, obs in visible_observations.iterrows():
            t_obs = float(obs.get("elapsed_minutes_adj", obs.get("elapsed_minutes", 0.0)))
            if t_obs > current_step_min:
                continue
            geom = obs.geometry
            if geom is None:
                continue
            ox = float(geom.x) if hasattr(geom, "x") else float(geom.centroid.x)
            oy = float(geom.y) if hasattr(geom, "y") else float(geom.centroid.y)

            c_min = max(0, int((ox - self.patch_radius_m - gx[0]) / dx))
            c_max = min(nx, int((ox + self.patch_radius_m - gx[0]) / dx) + 1)
            r_min = max(0, int((gy[0] - (oy + self.patch_radius_m)) / dy))
            r_max = min(ny, int((gy[0] - (oy - self.patch_radius_m)) / dy) + 1)

            if c_min >= c_max or r_min >= r_max:
                continue

            sub = assim_arr[r_min:r_max, c_min:c_max]
            sub[:] = np.minimum(sub, t_obs)

        ds = xr.Dataset(
            data_vars={"arrival_q10": (["y", "x"], assim_arr), "arrival_q50": (["y", "x"], assim_arr), "arrival_q90": (["y", "x"], assim_arr)},
            coords={"x": gx, "y": gy},
            attrs={"provider": "teacher_v2_dynamic_assim", "crs": "EPSG:32610", "ignition_time": "2018-11-08T14:33:00Z"},
        )
        validate_arrival_field(ds)
        return ds


class OffsetObservationInflatedProvider(ObservationInflatedProvider):
    """Observation inflated provider with configurable elapsed_minutes offset."""

    def __init__(self, base_teacher_provider: FieldProvider, r_spot_km: float = 4.0, offset_min: float = 8.0) -> None:
        super().__init__(base_teacher_provider=base_teacher_provider, r_spot_km=r_spot_km)
        self.offset_min = offset_min

    def get_field(self, current_step_min: float, visible_observations: gpd.GeoDataFrame | None = None) -> xr.Dataset:
        base_field = self.base_provider.get_field(current_step_min, visible_observations)
        if visible_observations is None or len(visible_observations) == 0:
            return base_field

        gx, gy = self._gx, self._gy
        if gx is None or gy is None:
            gx = base_field["x"].values
            gy = base_field["y"].values
            self._gx = gx
            self._gy = gy
            self._dx = float(gx[1] - gx[0])
            self._dy = float(gy[0] - gy[1])
            self._xx, self._yy = np.meshgrid(gx, gy)

        ny, nx = len(gy), len(gx)
        dx, dy = self._dx, self._dy
        base_arr = base_field["arrival_q50"].values.copy()
        inflated_arr = np.full((ny, nx), np.inf, dtype=np.float64)

        for obs_idx, obs in visible_observations.iterrows():
            # Use configured offset: t_obs = elapsed_minutes - offset_min
            raw_t = float(obs.get("elapsed_minutes", 0.0))
            t_obs = raw_t - self.offset_min
            if t_obs > current_step_min:
                continue

            geom = obs.geometry
            if geom is None:
                continue
            ox = float(geom.x) if hasattr(geom, "x") else float(geom.centroid.x)
            oy = float(geom.y) if hasattr(geom, "y") else float(geom.centroid.y)

            c_min = max(0, int((ox - self.r_spot_m - gx[0]) / dx))
            c_max = min(nx, int((ox + self.r_spot_m - gx[0]) / dx) + 1)
            r_min = max(0, int((gy[0] - (oy + self.r_spot_m)) / dy))
            r_max = min(ny, int((gy[0] - (oy - self.r_spot_m)) / dy) + 1)

            if c_min >= c_max or r_min >= r_max:
                continue

            sub_x = gx[c_min:c_max]
            sub_y = gy[r_min:r_max]
            mesh_x, mesh_y = np.meshgrid(sub_x, sub_y)
            vec_x, vec_y = mesh_x - ox, mesh_y - oy
            dist = np.hypot(vec_x, vec_y)

            dot = vec_x * self.wind_u + vec_y * self.wind_v
            dot_norm = np.clip(dot / np.maximum(dist, 1e-6), -1.0, 1.0)
            angle = np.arccos(dot_norm)

            cone_mask = (dist <= self.r_spot_m) & (angle <= self.wind_cone_rad)
            t_arrival_cone = t_obs + dist / self.spread_speed_m_per_min

            sub_arr = inflated_arr[r_min:r_max, c_min:c_max]
            sub_arr[cone_mask] = np.minimum(sub_arr[cone_mask], t_arrival_cone[cone_mask])

        combined = np.minimum(base_arr, inflated_arr)
        ds = xr.Dataset(
            data_vars={"arrival_q10": (["y", "x"], combined), "arrival_q50": (["y", "x"], combined), "arrival_q90": (["y", "x"], combined)},
            coords={"x": gx, "y": gy},
            attrs={"provider": f"prov_b_offset_{self.offset_min:.0f}", "crs": "EPSG:32610", "ignition_time": "2018-11-08T14:33:00Z"},
        )
        return ds


def run_agent_simulation_with_latency_events(
    method_name: str,
    origins: Sequence[EvacuationOrigin],
    road_graph: RoadGraph,
    destinations: Sequence[int] | set[int],
    truth_hazards: Sequence[TruthHazard],
    harness: CausalReplayHarness,
    buffer_minutes: float = 15.0,
    step_duration_min: float = 10.0,
    horizon_min: float = 360.0,
) -> tuple[list[ReplayOriginResult], list[float]]:
    """Runs agent simulation while recording wall-clock ms for each individual re-plan event."""
    router = TimeDependentRouter(road_graph=road_graph)
    validator = IndependentRouteValidator()
    dest_set = set(destinations)

    env0 = harness.step(0.0)
    edge_arrivals = road_graph.compute_edge_fire_arrivals(env0.arrival_field)

    routes: dict[int, EvacuationRoute | None] = {}
    for orig in origins:
        routes[orig.origin_id] = router.find_route(
            orig.origin_id, dest_set, orig.departure_time_min, buffer_minutes, edge_arrivals
        )

    initial_routes = {orig.origin_id: routes[orig.origin_id] for orig in origins}
    replan_histories: dict[int, list[tuple[float, EvacuationRoute | None]]] = {
        orig.origin_id: [(0.0, routes[orig.origin_id])] for orig in origins
    }
    replan_counts: dict[int, int] = {orig.origin_id: 0 for orig in origins}

    steps = np.arange(step_duration_min, horizon_min + 1e-6, step_duration_min)
    replan_event_latencies_ms: list[float] = []

    for step_t in steps:
        env = harness.step(step_t)
        edge_arrivals = road_graph.compute_edge_fire_arrivals(env.arrival_field)
        closed_rb = harness.get_closed_edges_for_roadblocks(env.active_roadblocks)

        dynamic_times = None
        if closed_rb:
            dynamic_times = np.array([e.traversal_time_min for e in road_graph.edges], dtype=np.float64)
            for idx, ge in enumerate(road_graph.edges):
                if (ge.u, ge.v) in closed_rb or (ge.v, ge.u) in closed_rb:
                    dynamic_times[idx] = float("inf")

        for orig in origins:
            cur_r = routes[orig.origin_id]
            if cur_r is None or cur_r.arrival_time_min <= step_t:
                continue

            severed = any((e.u, e.v) in closed_rb or (e.v, e.u) in closed_rb for e in cur_r.edges)

            edge_lookup = {(e.u, e.v): e for e in road_graph.edges}
            min_cur_margin = float("inf")
            for re in cur_r.edges:
                ed = edge_lookup.get((re.u, re.v))
                if ed is not None:
                    fire_t = float(edge_arrivals[ed.edge_idx])
                    m = fire_t - re.exit_time_min
                    if m < min_cur_margin:
                        min_cur_margin = m

            margin_eroded = min_cur_margin < buffer_minutes
            if not (severed or margin_eroded):
                continue

            # Measured Re-plan EVENT: Router + Validator
            t_event_start = time.perf_counter()
            replan_counts[orig.origin_id] += 1
            if step_t <= orig.departure_time_min:
                new_r = router.find_route(
                    orig.origin_id, dest_set, orig.departure_time_min, buffer_minutes,
                    edge_arrivals, dynamic_times
                )
                if new_r is not None:
                    _ = validator.validate(new_r, env.arrival_field, road_graph, buffer_minutes)
                routes[orig.origin_id] = new_r
                replan_histories[orig.origin_id].append((step_t, new_r))
            else:
                pos = _find_vehicle_position_on_route(cur_r, step_t)
                if pos is not None:
                    down_r = router.find_route(
                        pos.edge_v, dest_set, step_t, buffer_minutes,
                        edge_arrivals, dynamic_times
                    )
                    if down_r is not None:
                        _ = validator.validate(down_r, env.arrival_field, road_graph, buffer_minutes)
                    routes[orig.origin_id] = down_r
                    replan_histories[orig.origin_id].append((step_t, down_r))
                else:
                    routes[orig.origin_id] = None
                    replan_histories[orig.origin_id].append((step_t, None))

            event_elapsed_ms = (time.perf_counter() - t_event_start) * 1000.0
            replan_event_latencies_ms.append(event_elapsed_ms)

    # Compile results
    results: list[ReplayOriginResult] = []
    for orig in origins:
        final_r = routes[orig.origin_id]
        by_type = check_route_hazard_miss_by_type(final_r, truth_hazards, road_graph)
        is_miss, _ = by_type["total"]
        lead_time, flagged_before = compute_warning_lead_time(
            initial_routes[orig.origin_id], replan_histories[orig.origin_id], truth_hazards, road_graph
        )
        tt = final_r.total_travel_time_min if final_r is not None else float("inf")
        margin = final_r.min_margin_min if final_r is not None else -float("inf")

        results.append(
            ReplayOriginResult(
                origin_id=orig.origin_id,
                method_name=method_name,
                departure_time_min=orig.departure_time_min,
                route=final_r,
                travel_time_min=tt,
                hazard_miss=is_miss,
                min_margin_min=margin,
                replan_count=replan_counts[orig.origin_id],
                warning_lead_time_min=lead_time,
                flagged_before_onset=flagged_before,
                hazard_misses_by_type={k: v[0] for k, v in by_type.items()},
            )
        )
    return results, replan_event_latencies_ms


def cluster_bootstrap_difference(
    static_misses: np.ndarray,
    method_misses: np.ndarray,
    cluster_labels: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Cluster bootstrap for difference in miss rates (method - static).

    Resamples clusters with replacement; each cluster contributes all its constituent origins.
    """
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    diff_observed = float(np.mean(method_misses) - np.mean(static_misses)) * 100.0

    cluster_indices = {c: np.where(cluster_labels == c)[0] for c in unique_clusters}

    rng = np.random.default_rng(seed)
    diffs = np.zeros(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        sampled_c = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx_list = [cluster_indices[c] for c in sampled_c]
        sampled_indices = np.concatenate(idx_list)

        diffs[b] = (np.mean(method_misses[sampled_indices]) - np.mean(static_misses[sampled_indices])) * 100.0

    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    return diff_observed, (ci_low, ci_high)


def main() -> None:
    t_start = time.time()
    print("=" * 80)
    print("PHASE 6D: REPLAY ATTRIBUTION AND ROBUSTNESS CLOSE-OUT AUDIT")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading simulation domain on {device}...")
    from evaluate_heldout_camp_fire import build_camp_fire_dataset_60m
    ds_camp, bbox, transform = build_camp_fire_dataset_60m(device)
    h, w = ds_camp.fbfm_arr.shape

    # Road Graph and Origins
    teacher_field = TeacherFieldProvider().get_field(0.0)
    road_graph = load_camp_fire_road_graph(
        full_detail=True,
        grid_x_coords=teacher_field["x"].values,
        grid_y_coords=teacher_field["y"].values,
    )
    destinations = set(road_graph.boundary_destinations.values())
    origins = sample_block_origins(road_graph=road_graph, num_samples=200, seed=42)

    harness_base = CausalReplayHarness(road_graph=road_graph, provider=TeacherFieldProvider(), latency_min=0.0)
    truth_hazards = harness_base.truth_hazards

    # Map static egress corridors for cluster bootstrap
    res_static, _ = run_static_baseline(origins, road_graph, destinations, truth_hazards)
    corridor_clusters = []
    for r in res_static:
        if r.route and r.route.edges:
            dest_node = r.route.edges[-1].v
            if dest_node == 86429619:
                c = "Clark_Rd"
            elif dest_node == 2611113946:
                c = "Skyway"
            else:
                c = "Pentz_Hwy70"
        else:
            c = "Unknown"
        corridor_clusters.append(c)
    corridor_clusters = np.array(corridor_clusters)
    u_clusters, counts_c = np.unique(corridor_clusters, return_counts=True)
    print(f"\nCorridor clusters (K={len(u_clusters)}): {dict(zip(u_clusters, counts_c))}")

    # =========================================================================
    # STEP 2a: Teacher Seed Sensitivity (5 representative seeds from Arm c)
    # Seeds: 389 (min), 117 (low-mid), 304 (median), 100 (mid-high), 236 (max)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2a: TEACHER SEED SENSITIVITY (5 seeds: 389, 117, 304, 100, 236)")
    print("=" * 80)
    test_seeds = [389, 117, 304, 100, 236]

    t_params = TeacherParams(
        p_base=0.22, a=0.06, c_1=0.045, c_2=0.15, p_continue=0.30,
        fuel_scale_grass_shrub=1.0, fuel_scale_timber=1.0, canopy_den_scale=1.0, wind_moisture_damping=0.0
    )
    e_params = EmberParams(alpha_launch=0.0040, beta_dist_m_per_ms=110.0, p_urban_wui=0.15)

    seed_sens_rows = []
    gx = teacher_field["x"].values
    gy = teacher_field["y"].values

    for s in test_seeds:
        print(f"\nSimulating Teacher v2 with seed {s}...")
        sim_s = simulate_teacher_with_embers_60m(
            ds_camp, t_params, ember_params=e_params, use_ember_kernel=True,
            duration_hours=12, steps_per_hour=30, seed=s
        )
        arr_s = sim_s["arrival_time_min"]
        prov_dyn_a2 = DynamicTeacherV2Provider(arr_s, gx, gy)
        prov_dyn_b = ObservationInflatedProvider(base_teacher_provider=prov_dyn_a2, r_spot_km=4.0)

        # Run A2 on this seed
        h_a2_s = CausalReplayHarness(road_graph=road_graph, provider=prov_dyn_a2, latency_min=0.0)
        res_a2_s, _ = run_agent_simulation(f"A2_seed_{s}", origins, road_graph, destinations, truth_hazards, h_a2_s)
        miss_a2 = sum(1 for r in res_a2_s if r.hazard_miss)
        trap_a2 = sum(1 for r in res_a2_s if r.route is None or not r.route.edges)
        av_routed_a2 = [count_abandoned_vehicles_for_route(r.route, truth_hazards, road_graph)[0] for r in res_a2_s if r.route is not None and r.route.edges]

        # Run B-on-A2 on this seed
        h_b_s = CausalReplayHarness(road_graph=road_graph, provider=prov_dyn_b, latency_min=0.0)
        res_b_s, _ = run_agent_simulation(f"B_on_A2_seed_{s}", origins, road_graph, destinations, truth_hazards, h_b_s)
        miss_b = sum(1 for r in res_b_s if r.hazard_miss)
        trap_b = sum(1 for r in res_b_s if r.route is None or not r.route.edges)
        av_routed_b = [count_abandoned_vehicles_for_route(r.route, truth_hazards, road_graph)[0] for r in res_b_s if r.route is not None and r.route.edges]

        seed_sens_rows.append({
            "teacher_seed": s,
            "method": "A2 (Teacher v2 Embers)",
            "combined_misses": f"{miss_a2}/200 ({miss_a2/2.0:.1f}%)",
            "trapped_count": f"{trap_a2}/200 ({trap_a2/2.0:.1f}%)",
            "av_routed_mean": float(np.mean(av_routed_a2)) if av_routed_a2 else 0.0,
            "av_routed_median": float(np.median(av_routed_a2)) if av_routed_a2 else 0.0,
            "av_routed_p90": float(np.percentile(av_routed_a2, 90.0)) if av_routed_a2 else 0.0,
        })
        seed_sens_rows.append({
            "teacher_seed": s,
            "method": "B-on-A2 (Obs on v2)",
            "combined_misses": f"{miss_b}/200 ({miss_b/2.0:.1f}%)",
            "trapped_count": f"{trap_b}/200 ({trap_b/2.0:.1f}%)",
            "av_routed_mean": float(np.mean(av_routed_b)) if av_routed_b else 0.0,
            "av_routed_median": float(np.median(av_routed_b)) if av_routed_b else 0.0,
            "av_routed_p90": float(np.percentile(av_routed_b, 90.0)) if av_routed_b else 0.0,
        })

    df_seed_sens = pd.DataFrame(seed_sens_rows)
    df_seed_sens.to_csv("reports/replay_seed_robustness.csv", index=False)
    print("\n=== SEED SENSITIVITY TABLE SAVED to reports/replay_seed_robustness.csv ===")
    print(df_seed_sens.to_string(index=False))

    # =========================================================================
    # STEP 2b: Offset Sensitivity (0, 4, 8, 12 min) on Teacher v2 (frozen seed 42)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2b: OFFSET SENSITIVITY (0, 4, 8, 12 min)")
    print("=" * 80)
    prov_a2_frozen = TeacherV2FieldProvider(assimilate=True)

    offset_rows = []
    for off in [0.0, 4.0, 8.0, 12.0]:
        prov_b_off = OffsetObservationInflatedProvider(base_teacher_provider=prov_a2_frozen, r_spot_km=4.0, offset_min=off)
        h_off = CausalReplayHarness(road_graph=road_graph, provider=prov_b_off, latency_min=0.0)
        res_off, _ = run_agent_simulation(f"B_offset_{int(off)}", origins, road_graph, destinations, truth_hazards, h_off)

        miss_cnt = sum(1 for r in res_off if r.hazard_miss)
        trap_cnt = sum(1 for r in res_off if r.route is None or not r.route.edges)
        av_routed = [count_abandoned_vehicles_for_route(r.route, truth_hazards, road_graph)[0] for r in res_off if r.route is not None and r.route.edges]

        offset_rows.append({
            "offset_min": off,
            "combined_misses": f"{miss_cnt}/200 ({miss_cnt/2.0:.1f}%)",
            "trapped_count": f"{trap_cnt}/200 ({trap_cnt/2.0:.1f}%)",
            "av_routed_mean": float(np.mean(av_routed)) if av_routed else 0.0,
            "av_routed_median": float(np.median(av_routed)) if av_routed else 0.0,
            "av_routed_p90": float(np.percentile(av_routed, 90.0)) if av_routed else 0.0,
        })

    df_offset = pd.DataFrame(offset_rows)
    df_offset.to_csv("reports/replay_offset_sensitivity.csv", index=False)
    print("\n=== OFFSET SENSITIVITY TABLE SAVED to reports/replay_offset_sensitivity.csv ===")
    print(df_offset.to_string(index=False))

    # =========================================================================
    # STEP 3: Metric Cleanup & Cluster Bootstrap
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: METRIC CLEANUP & CLUSTER BOOTSTRAP")
    print("=" * 80)

    # Core providers
    prov_a1 = TeacherFieldProvider()
    prov_a2 = TeacherV2FieldProvider(assimilate=True)
    prov_b_on_a1 = ObservationInflatedProvider(base_teacher_provider=prov_a1, r_spot_km=4.0)
    prov_b_on_a2 = ObservationInflatedProvider(base_teacher_provider=prov_a2, r_spot_km=4.0)

    # Leakage exclusion
    raw_obs = harness_base._raw_observations
    obs_coords = np.array([(row.geometry.x, row.geometry.y) for _, row in raw_obs.iterrows()])
    obs_times = raw_obs["elapsed_minutes_adj"].values
    overlap_indices = set()
    for h in truth_hazards:
        hx, hy = h.geometry_utm.x, h.geometry_utm.y
        ht = h.onset_elapsed_min
        dists = np.hypot(obs_coords[:, 0] - hx, obs_coords[:, 1] - hy)
        dt = np.abs(obs_times - ht)
        matched = np.where((dists <= 500.0) & (dt <= 30.0))[0]
        overlap_indices.update(matched)

    prov_b_clean = ObservationInflatedProvider(
        base_teacher_provider=prov_a1, r_spot_km=4.0, filter_leakage=True, leakage_exclusion_indices=overlap_indices
    )

    methods = {
        "static": res_static,
        "reactive": run_reactive_baseline(origins, road_graph, destinations, truth_hazards, harness_base)[0],
        "agent_a": run_agent_simulation("agent_a", origins, road_graph, destinations, truth_hazards, CausalReplayHarness(road_graph, prov_a1, 0.0))[0],
        "agent_a2": run_agent_simulation("agent_a2", origins, road_graph, destinations, truth_hazards, CausalReplayHarness(road_graph, prov_a2, 0.0))[0],
        "agent_b_on_a": run_agent_simulation("agent_b_on_a", origins, road_graph, destinations, truth_hazards, CausalReplayHarness(road_graph, prov_b_on_a1, 0.0))[0],
        "agent_b_on_a2": run_agent_simulation("agent_b_on_a2", origins, road_graph, destinations, truth_hazards, CausalReplayHarness(road_graph, prov_b_on_a2, 0.0))[0],
        "agent_b_clean": run_agent_simulation("agent_b_clean", origins, road_graph, destinations, truth_hazards, CausalReplayHarness(road_graph, prov_b_clean, 0.0))[0],
    }

    cleaned_metrics_rows = []
    for m_label, res_m in methods.items():
        trapped_count = sum(1 for r in res_m if r.route is None or not r.route.edges)
        routed_results = [r for r in res_m if r.route is not None and r.route.edges]

        # Burnover encounters among routed origins
        bo_counts = [count_burnovers_for_route(r.route, truth_hazards, road_graph)[0] for r in routed_results]
        bo_routed_encounters = sum(bo_counts)
        bo_all_met_ids = set()
        for r in routed_results:
            bo_all_met_ids.update(count_burnovers_for_route(r.route, truth_hazards, road_graph)[1])
        unique_bo_met = len(bo_all_met_ids)

        # Roadblocks (OBSERVED-INPUT)
        rb_counts = [count_roadblocks_for_route(r.route, truth_hazards, road_graph)[0] for r in routed_results]
        rb_routed_encounters = sum(rb_counts)
        rb_all_met_ids = set()
        for r in routed_results:
            rb_all_met_ids.update(count_roadblocks_for_route(r.route, truth_hazards, road_graph)[1])
        unique_rb_met = len(rb_all_met_ids)

        # Abandoned vehicles (UNOBSERVED)
        av_counts = [count_abandoned_vehicles_for_route(r.route, truth_hazards, road_graph)[0] for r in routed_results]
        av_mean = float(np.mean(av_counts)) if av_counts else 0.0
        av_med = float(np.median(av_counts)) if av_counts else 0.0
        av_p90 = float(np.percentile(av_counts, 90.0)) if av_counts else 0.0
        av_at_least_one = sum(1 for c in av_counts if c > 0)
        av_all_met_ids = set()
        for r in routed_results:
            av_all_met_ids.update(count_abandoned_vehicles_for_route(r.route, truth_hazards, road_graph)[1])
        unique_av_met = len(av_all_met_ids)

        cleaned_metrics_rows.append({
            "method": m_label,
            "trapped_origins": f"{trapped_count}/200 ({trapped_count/2.0:.1f}%)",
            "bo_routed_encounters": f"{bo_routed_encounters} ({unique_bo_met}/23 unique met)",
            "rb_encounters_OBSERVED_INPUT": f"{rb_routed_encounters} ({unique_rb_met}/178 unique met)",
            "av_routed_mean": av_mean,
            "av_routed_median": av_med,
            "av_routed_p90": av_p90,
            "av_origin_at_least_one": f"{av_at_least_one}/{len(routed_results)} ({av_at_least_one/max(len(routed_results),1)*100:.1f}%)",
            "av_unique_met": f"{unique_av_met}/235 ({unique_av_met/235.0*100:.1f}%)",
        })

    df_cleaned = pd.DataFrame(cleaned_metrics_rows)
    df_cleaned.to_csv("reports/replay_cleaned_metrics.csv", index=False)
    print("\n=== CLEANED METRICS TABLE SAVED to reports/replay_cleaned_metrics.csv ===")
    print(df_cleaned.to_string(index=False))

    # Cluster Bootstrap & Plain Bootstrap
    print("\n--- Running 10,000 Cluster and Plain Bootstrap Resamples ---")
    bootstrap_comparison_rows = []

    # Comparisons vs Static
    stat_miss = np.array([r.hazard_miss for r in methods["static"]], dtype=int)
    for m_label in ["reactive", "agent_a", "agent_a2", "agent_b_on_a", "agent_b_on_a2", "agent_b_clean"]:
        meth_miss = np.array([r.hazard_miss for r in methods[m_label]], dtype=int)

        # Discordant counts
        n10 = int(np.sum((stat_miss == 1) & (meth_miss == 0)))
        n01 = int(np.sum((stat_miss == 0) & (meth_miss == 1)))

        # Plain unclustered bootstrap
        from eval_replay_6c_audit import paired_bootstrap_difference
        diff_p, ci_plain, _, _ = paired_bootstrap_difference(stat_miss, meth_miss, 10000, 42)
        width_plain = ci_plain[1] - ci_plain[0]

        # Cluster bootstrap (K=3)
        diff_c, ci_clust = cluster_bootstrap_difference(stat_miss, meth_miss, corridor_clusters, 10000, 42)
        width_clust = ci_clust[1] - ci_clust[0]

        ratio_width = width_clust / max(width_plain, 1e-6)

        bootstrap_comparison_rows.append({
            "comparison": f"{m_label} vs static",
            "diff_pct": diff_p,
            "discordant_n10_n01": f"n10={n10}, n01={n01}",
            "plain_ci_95": f"[{ci_plain[0]:+.1f}%, {ci_plain[1]:+.1f}%]",
            "plain_width": width_plain,
            "cluster_ci_95": f"[{ci_clust[0]:+.1f}%, {ci_clust[1]:+.1f}%]",
            "cluster_width": width_clust,
            "width_ratio_clust_to_plain": ratio_width,
            "cluster_survives": not (ci_clust[0] <= 0 <= ci_clust[1]),
        })

    # Comparison A2 vs A
    a_miss = np.array([r.hazard_miss for r in methods["agent_a"]], dtype=int)
    a2_miss = np.array([r.hazard_miss for r in methods["agent_a2"]], dtype=int)
    n10_a2 = int(np.sum((a_miss == 1) & (a2_miss == 0)))
    n01_a2 = int(np.sum((a_miss == 0) & (a2_miss == 1)))
    diff_a2_p, ci_a2_plain, _, _ = paired_bootstrap_difference(a_miss, a2_miss, 10000, 42)
    diff_a2_c, ci_a2_clust = cluster_bootstrap_difference(a_miss, a2_miss, corridor_clusters, 10000, 42)
    w_plain_a2 = ci_a2_plain[1] - ci_a2_plain[0]
    w_clust_a2 = ci_a2_clust[1] - ci_a2_clust[0]

    bootstrap_comparison_rows.append({
        "comparison": "agent_a2 vs agent_a",
        "diff_pct": diff_a2_p,
        "discordant_n10_n01": f"n10={n10_a2}, n01={n01_a2}",
        "plain_ci_95": f"[{ci_a2_plain[0]:+.1f}%, {ci_a2_plain[1]:+.1f}%]",
        "plain_width": w_plain_a2,
        "cluster_ci_95": f"[{ci_a2_clust[0]:+.1f}%, {ci_a2_clust[1]:+.1f}%]",
        "cluster_width": w_clust_a2,
        "width_ratio_clust_to_plain": w_clust_a2 / max(w_plain_a2, 1e-6),
        "cluster_survives": not (ci_a2_clust[0] <= 0 <= ci_a2_clust[1]),
    })

    df_boot = pd.DataFrame(bootstrap_comparison_rows)
    df_boot.to_csv("reports/replay_cluster_bootstrap.csv", index=False)
    print("\n=== CLUSTER BOOTSTRAP SUMMARY SAVED to reports/replay_cluster_bootstrap.csv ===")
    print(df_boot.to_string(index=False))

    # =========================================================================
    # STEP 4: Latency per actual Re-plan EVENT in Agent B
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: LATENCY PER ACTUAL RE-PLAN EVENT IN AGENT B")
    print("=" * 80)
    h_b_prof = CausalReplayHarness(road_graph=road_graph, provider=prov_b_on_a1, latency_min=0.0)
    _, event_latencies = run_agent_simulation_with_latency_events(
        "agent_b_prof", origins, road_graph, destinations, truth_hazards, h_b_prof, buffer_minutes=15.0
    )

    n_events = len(event_latencies)
    mean_ev = float(np.mean(event_latencies))
    p50_ev = float(np.median(event_latencies))
    p95_ev = float(np.percentile(event_latencies, 95.0))
    print(f"Agent B actual re-plan events: {n_events} events")
    print(f"Latency per re-plan event (Router + Validator): Mean = {mean_ev:.2f} ms | P50 = {p50_ev:.2f} ms | P95 = {p95_ev:.2f} ms")

    total_time = time.time() - t_start
    print(f"\nPhase 6D audit script completed in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
