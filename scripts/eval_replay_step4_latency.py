"""Step 4: Latency Measurement per actual Re-Plan Event in Agent B Replay.

Measures:
- Provider update latency (field generation / assimilation + edge arrival projection)
- Router find_route latency
- Validator validate latency
- Total end-to-end latency per actual re-plan event:
  - Event count
  - Mean, Median (p50), P95

Reconciles with:
- 2.88 ms pure Dijkstra search
- 17.2 ms static routing
- 67 ms benchmark
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from wf052.routing.graph import load_camp_fire_road_graph, RoadGraph
from wf052.routing.router import TimeDependentRouter, RoutingMode
from wf052.routing.validator import IndependentRouteValidator
from wf052.routing.contracts import EvacuationRoute, validate_arrival_field
from wf052.replay.contracts import TruthHazard
from wf052.replay.harness import CausalReplayHarness
from wf052.replay.providers import (
    FieldProvider,
    TeacherFieldProvider,
    TeacherV2FieldProvider,
    ObservationInflatedProvider,
)
from wf052.replay.methods import (
    EvacuationOrigin,
    sample_block_origins,
    _find_vehicle_position_on_route,
)


def run_latency_profile(
    harness: CausalReplayHarness,
    origins: list[EvacuationOrigin],
    road_graph: RoadGraph,
    destinations: list[int],
    buffer_minutes: float = 15.0,
    horizon_min: float = 360.0,
    step_duration_min: float = 10.0,
):
    router = TimeDependentRouter(road_graph)
    validator = IndependentRouteValidator()
    dest_set = set(destinations)

    # Initial routing at t=0
    t0_prov_start = time.perf_counter()
    env0 = harness.step(0.0)
    edge_arrivals = road_graph.compute_edge_fire_arrivals(env0.arrival_field)
    t0_prov_ms = (time.perf_counter() - t0_prov_start) * 1000.0

    t_init_routes = []
    routes: dict[int, EvacuationRoute | None] = {}
    for orig in origins:
        t_r_start = time.perf_counter()
        r = router.find_route(
            origin_node=orig.origin_id,
            destination_nodes=dest_set,
            t0=orig.departure_time_min,
            buffer_minutes=buffer_minutes,
            edge_fire_arrivals=edge_arrivals,
        )
        t_r_ms = (time.perf_counter() - t_r_start) * 1000.0
        routes[orig.origin_id] = r
        t_init_routes.append(t_r_ms)

    steps = np.arange(step_duration_min, horizon_min + 1e-6, step_duration_min)

    # Tracking per-event latencies
    events: list[dict] = []
    step_provider_times: list[float] = []

    for step_t in steps:
        # 1. Provider update + edge fire arrivals
        t_step_prov_start = time.perf_counter()
        env = harness.step(step_t)
        edge_arrivals = road_graph.compute_edge_fire_arrivals(env.arrival_field)
        closed_rb = harness.get_closed_edges_for_roadblocks(env.active_roadblocks)
        t_step_prov_ms = (time.perf_counter() - t_step_prov_start) * 1000.0
        step_provider_times.append(t_step_prov_ms)

        dynamic_times = None
        if closed_rb:
            dynamic_times = np.array([e.traversal_time_min for e in road_graph.edges], dtype=np.float64)
            for idx, ge in enumerate(road_graph.edges):
                if (ge.u, ge.v) in closed_rb or (ge.v, ge.u) in closed_rb:
                    dynamic_times[idx] = float("inf")

        # Step-level origins re-planning
        step_events: list[dict] = []
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

            # This is an actual Re-plan EVENT
            # Time Router
            t_router_start = time.perf_counter()
            if step_t <= orig.departure_time_min:
                new_r = router.find_route(
                    origin_node=orig.origin_id,
                    destination_nodes=dest_set,
                    t0=orig.departure_time_min,
                    buffer_minutes=buffer_minutes,
                    edge_fire_arrivals=edge_arrivals,
                    dynamic_edge_traversal_times=dynamic_times,
                )
            else:
                pos = _find_vehicle_position_on_route(cur_r, step_t)
                if pos is not None:
                    new_r = router.find_route(
                        origin_node=pos.edge_v,
                        destination_nodes=dest_set,
                        t0=step_t,
                        buffer_minutes=buffer_minutes,
                        edge_fire_arrivals=edge_arrivals,
                        dynamic_edge_traversal_times=dynamic_times,
                    )
                else:
                    new_r = None
            t_router_ms = (time.perf_counter() - t_router_start) * 1000.0

            # Time Validator
            t_val_start = time.perf_counter()
            if new_r is not None:
                _ = validator.validate(new_r, env.arrival_field, road_graph, buffer_minutes)
            t_val_ms = (time.perf_counter() - t_val_start) * 1000.0

            routes[orig.origin_id] = new_r

            step_events.append({
                "step_t": step_t,
                "origin_id": orig.origin_id,
                "reason": "severed" if severed else "margin_eroded",
                "t_provider_step_ms": t_step_prov_ms,
                "t_router_ms": t_router_ms,
                "t_validator_ms": t_val_ms,
                "t_router_plus_validator_ms": t_router_ms + t_val_ms,
            })

        n_ev_at_step = len(step_events)
        for ev in step_events:
            ev["n_events_at_step"] = n_ev_at_step
            ev["t_provider_amortized_ms"] = ev["t_provider_step_ms"] / max(n_ev_at_step, 1)
            ev["t_e2e_direct_ms"] = ev["t_provider_step_ms"] + ev["t_router_plus_validator_ms"]
            ev["t_e2e_amortized_ms"] = ev["t_provider_amortized_ms"] + ev["t_router_plus_validator_ms"]
            events.append(ev)

    return pd.DataFrame(events), pd.Series(step_provider_times), pd.Series(t_init_routes)


def main():
    print("=" * 80)
    print("STEP 4: DETAILED LATENCY MEASUREMENT PER ACTUAL RE-PLAN EVENT IN AGENT B")
    print("=" * 80)

    # 1. Load Road Graph
    print("Loading Camp Fire road graph (14,691 edges)...")
    prov_a2 = TeacherV2FieldProvider(assimilate=True)
    teacher_field = prov_a2.get_field(current_step_min=0.0)

    road_graph = load_camp_fire_road_graph(
        full_detail=True,
        grid_x_coords=teacher_field["x"].values,
        grid_y_coords=teacher_field["y"].values,
    )
    destinations = set(road_graph.boundary_destinations.values())
    origins = sample_block_origins(road_graph=road_graph, num_samples=200, seed=42)

    # 2. Providers
    # Provider B on Teacher v2
    prov_b_on_a2 = ObservationInflatedProvider(base_teacher_provider=prov_a2, r_spot_km=4.0)
    harness_b_on_a2 = CausalReplayHarness(road_graph=road_graph, provider=prov_b_on_a2, latency_min=0.0)

    # Provider B on Teacher v1
    prov_a1 = TeacherFieldProvider()
    prov_b_on_a1 = ObservationInflatedProvider(base_teacher_provider=prov_a1, r_spot_km=4.0)
    harness_b_on_a1 = CausalReplayHarness(road_graph=road_graph, provider=prov_b_on_a1, latency_min=0.0)

    print("\n--- Profiling Agent B on Teacher v2 (Primary Arm) ---")
    df_events_a2, prov_times_a2, init_times = run_latency_profile(
        harness=harness_b_on_a2,
        origins=origins,
        road_graph=road_graph,
        destinations=destinations,
        buffer_minutes=15.0,
    )

    df_events_a2.to_csv("reports/replay_latency_events_agent_b_v2.csv", index=False)
    print(f"Saved {len(df_events_a2)} events to reports/replay_latency_events_agent_b_v2.csv")

    n_events = len(df_events_a2)
    print(f"\nAgent B on Teacher v2 Total Re-Plan Events: {n_events}")
    
    # Provider Update Latency across 36 steps
    print("\n1. Provider Update (Per Step, including edge arrival projection):")
    print(f"   Count: {len(prov_times_a2)} steps")
    print(f"   Mean:   {prov_times_a2.mean():.2f} ms")
    print(f"   Median: {prov_times_a2.median():.2f} ms")
    print(f"   P95:    {np.percentile(prov_times_a2, 95.0):.2f} ms")

    # Router Latency per event
    r_times = df_events_a2["t_router_ms"]
    print("\n2. Router find_route Latency (Per Event):")
    print(f"   Mean:   {r_times.mean():.2f} ms")
    print(f"   Median: {r_times.median():.2f} ms")
    print(f"   P95:    {np.percentile(r_times, 95.0):.2f} ms")

    # Validator Latency per event
    v_times = df_events_a2["t_validator_ms"]
    print("\n3. Validator validate Latency (Per Event):")
    print(f"   Mean:   {v_times.mean():.2f} ms")
    print(f"   Median: {v_times.median():.2f} ms")
    print(f"   P95:    {np.percentile(v_times, 95.0):.2f} ms")

    # Router + Validator per event
    rv_times = df_events_a2["t_router_plus_validator_ms"]
    print("\n4. Router + Validator Latency (Per Event):")
    print(f"   Mean:   {rv_times.mean():.2f} ms")
    print(f"   Median: {rv_times.median():.2f} ms")
    print(f"   P95:    {np.percentile(rv_times, 95.0):.2f} ms")

    # Amortized End-to-End Latency per event
    amort_times = df_events_a2["t_e2e_amortized_ms"]
    print("\n5. Amortized End-to-End Latency per event (Provider/N_events + Router + Validator):")
    print(f"   Mean:   {amort_times.mean():.2f} ms")
    print(f"   Median: {amort_times.median():.2f} ms")
    print(f"   P95:    {np.percentile(amort_times, 95.0):.2f} ms")

    # Direct End-to-End Latency per event (Step Provider + Router + Validator)
    direct_times = df_events_a2["t_e2e_direct_ms"]
    print("\n6. Direct End-to-End Latency per event (Full Step Provider + Router + Validator):")
    print(f"   Mean:   {direct_times.mean():.2f} ms")
    print(f"   Median: {direct_times.median():.2f} ms")
    print(f"   P95:    {np.percentile(direct_times, 95.0):.2f} ms")

    # Static Initial Route Latency per origin
    print("\n7. Static Baseline Initial Route Latency (Per Origin at t=0):")
    print(f"   Mean:   {init_times.mean():.2f} ms")
    print(f"   Median: {init_times.median():.2f} ms")
    print(f"   P95:    {np.percentile(init_times, 95.0):.2f} ms")

    # Summary table for reports
    summary_df = pd.DataFrame([
        {
            "metric": "replan_event_count",
            "mean": float(n_events),
            "median": float(n_events),
            "p95": float(n_events),
        },
        {
            "metric": "provider_update_step_ms",
            "mean": float(prov_times_a2.mean()),
            "median": float(prov_times_a2.median()),
            "p95": float(np.percentile(prov_times_a2, 95.0)),
        },
        {
            "metric": "router_find_route_ms",
            "mean": float(r_times.mean()),
            "median": float(r_times.median()),
            "p95": float(np.percentile(r_times, 95.0)),
        },
        {
            "metric": "validator_validate_ms",
            "mean": float(v_times.mean()),
            "median": float(v_times.median()),
            "p95": float(np.percentile(v_times, 95.0)),
        },
        {
            "metric": "router_plus_validator_ms",
            "mean": float(rv_times.mean()),
            "median": float(rv_times.median()),
            "p95": float(np.percentile(rv_times, 95.0)),
        },
        {
            "metric": "e2e_replan_event_amortized_ms",
            "mean": float(amort_times.mean()),
            "median": float(amort_times.median()),
            "p95": float(np.percentile(amort_times, 95.0)),
        },
        {
            "metric": "e2e_replan_event_direct_ms",
            "mean": float(direct_times.mean()),
            "median": float(direct_times.median()),
            "p95": float(np.percentile(direct_times, 95.0)),
        },
        {
            "metric": "static_initial_route_ms",
            "mean": float(init_times.mean()),
            "median": float(init_times.median()),
            "p95": float(np.percentile(init_times, 95.0)),
        },
    ])
    summary_df.to_csv("reports/replay_latency_summary.csv", index=False)
    print("\n=== LATENCY SUMMARY SAVED to reports/replay_latency_summary.csv ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
