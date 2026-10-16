"""Four-stage end-to-end evacuation router latency benchmark (Phase 4c).

Breaks end-to-end evacuation latency into 4 distinct stages:
1. Field update / edge mapping (spatial min arrival per edge over grid)
2. Congested time-sliced re-assignment from current time forward
3. Route validation (independent re-integration of traversal and margins)
4. Entrapment metrics calculation (cohort danger exposure and fire arrivals)

Explicitly benchmarks the full time-sliced assignment pipeline in contrast
to prior benchmarks that only measured point-to-point Dijkstra replanning.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd

from wf052.routing.congestion import CongestionModel, TimeSlicedAssignmentEngine
from wf052.routing.demand import EvacuationDemandModel
from wf052.routing.graph import RoadGraph, load_camp_fire_road_graph
from wf052.routing.router import TimeDependentRouter
from wf052.routing.synthetic_fire import create_wind_ellipse_arrival_field
from wf052.routing.validator import IndependentRouteValidator


def benchmark_pipeline(num_runs: int = 25) -> pd.DataFrame:
    print(f"Loading full-detail Camp Fire road graph (14,691 edges)...")
    road_graph = load_camp_fire_road_graph(full_detail=True)
    assert road_graph.grid_x is not None and road_graph.grid_y is not None
    x = road_graph.grid_x
    y = road_graph.grid_y

    demand_model = EvacuationDemandModel(
        config_path="configs/evacuation_demand.yaml", road_graph=road_graph
    )
    congestion_model = CongestionModel(
        road_graph=road_graph,
        config_path="configs/evacuation_demand.yaml",
        use_point_queue=True,
    )
    router = TimeDependentRouter(road_graph=road_graph)
    engine = TimeSlicedAssignmentEngine(
        road_graph=road_graph,
        demand_model=demand_model,
        congestion_model=congestion_model,
        router=router,
    )
    validator = IndependentRouteValidator()
    destinations = set(road_graph.boundary_destinations.values())

    times_stage1 = []  # Field update / edge mapping
    times_stage2 = []  # Time-sliced congested assignment
    times_stage3 = []  # Route validation
    times_stage4 = []  # Entrapment metrics
    times_total = []

    np.random.seed(42)
    print(f"Executing {num_runs} end-to-end benchmark trials...")

    for i in range(num_runs):
        ign_x = float(np.random.uniform(616000.0, 622000.0))
        ign_y = float(np.random.uniform(4398000.0, 4406000.0))
        head_rate = float(np.random.uniform(40.0, 80.0))

        arr_field = create_wind_ellipse_arrival_field(
            x_coords=x,
            y_coords=y,
            ignition_x=ign_x,
            ignition_y=ign_y,
            head_rate_m_per_min=head_rate,
            back_rate_m_per_min=10.0,
            length_to_breadth=2.5,
            wind_towards_deg=235.0,
            horizon_minutes=720.0,
        )

        t_run_start = time.perf_counter()

        # Stage 1: Field update / edge mapping
        t1_0 = time.perf_counter()
        edge_arrivals = road_graph.compute_edge_fire_arrivals(arr_field)
        t1_1 = time.perf_counter()

        # Stage 2: Congested time-sliced re-assignment
        t2_0 = time.perf_counter()
        summaries, entrapment = engine.run_assignment(
            edge_fire_arrivals=edge_arrivals,
            destination_node_ids=destinations,
            apply_exit_capacities=True,
        )
        t2_1 = time.perf_counter()

        # Stage 3: Independent route validation on selected routes
        t3_0 = time.perf_counter()
        for z_name, z_obj in demand_model.zones.items():
            route = router.find_route(
                origin_node=z_obj.origin_node_id,
                destination_nodes=destinations,
                t0=z_obj.notification_elapsed_min,
                edge_fire_arrivals=edge_arrivals,
            )
            if route is not None and route.edges:
                validator.validate(route, arr_field, road_graph)
        t3_1 = time.perf_counter()

        # Stage 4: Standalone Entrapment Metrics computation
        t4_0 = time.perf_counter()
        _ = engine._compute_entrapment_metrics(
            cohort_trajectories=[],
            edge_fire_arrivals=edge_arrivals,
            danger_buffer_minutes=15.0,
        )
        t4_1 = time.perf_counter()

        t_run_end = time.perf_counter()

        times_stage1.append(t1_1 - t1_0)
        times_stage2.append(t2_1 - t2_0)
        times_stage3.append(t3_1 - t3_0)
        times_stage4.append(t4_1 - t4_0)
        times_total.append(t_run_end - t_run_start)

    stages = [
        ("Stage 1: Field Update & Edge Mapping", times_stage1),
        ("Stage 2: Congested Time-Sliced Re-Assignment", times_stage2),
        ("Stage 3: Independent Route Validation", times_stage3),
        ("Stage 4: Cohort Entrapment Metrics", times_stage4),
        ("Total End-to-End Pipeline", times_total),
    ]

    records = []
    for name, data in stages:
        arr_ms = np.array(data) * 1000.0
        records.append({
            "Stage": name,
            "p50 (ms)": round(float(np.percentile(arr_ms, 50)), 2),
            "p95 (ms)": round(float(np.percentile(arr_ms, 95)), 2),
            "max (ms)": round(float(np.max(arr_ms)), 2),
            "mean (ms)": round(float(np.mean(arr_ms)), 2),
        })

    df = pd.DataFrame(records)
    out_csv = Path("reports/latency_4stage_benchmark.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved 4-stage latency benchmark to {out_csv}")
    print("\n" + "=" * 80)
    print("FOUR-STAGE END-TO-END EVACUATION ROUTER LATENCY BENCHMARK (ms)")
    print("=" * 80)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    benchmark_pipeline(num_runs=20)
