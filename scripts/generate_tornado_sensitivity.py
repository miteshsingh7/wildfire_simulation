"""Tornado sensitivity analysis for Paradise evacuation clearance time (Phase 4c).

Evaluates sensitivity of Town of Paradise clearance time to:
1. capacity_scale (1.0, 0.7, 0.5)
2. inbound_traffic (0.0 vs 200.0 veh/hr)
3. Skyway south exit capacity (nominal 2,200 vs 1,540 / -30%)
4. Clark south exit capacity (nominal 1,400 vs 980 / -30%)
5. Pentz south exit capacity (nominal 1,000 vs 700 / -30%)
6. Skyway contraflow (ON vs OFF)

Identifies the dominant bottleneck parameter and compares against NETTRA / BCAG observations.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from wf052.routing.congestion import CongestionModel, TimeSlicedAssignmentEngine
from wf052.routing.demand import EvacuationDemandModel
from wf052.routing.graph import RoadGraph, load_camp_fire_road_graph
from wf052.routing.router import TimeDependentRouter


def run_single_scenario(
    road_graph: RoadGraph,
    capacity_scale: float = 1.0,
    inbound_traffic: float = 0.0,
    exit_capacities: dict[str, float] | None = None,
    skyway_contraflow: bool = False,
) -> tuple[float, dict[str, float]]:
    demand_model = EvacuationDemandModel(
        config_path="configs/evacuation_demand.yaml", road_graph=road_graph
    )
    congestion_model = CongestionModel(
        road_graph=road_graph,
        config_path="configs/evacuation_demand.yaml",
        use_point_queue=True,
        capacity_scale=capacity_scale,
        inbound_traffic_veh_per_hr=inbound_traffic,
    )

    if exit_capacities is not None:
        congestion_model.exit_outflow_capacities.update(exit_capacities)

    if skyway_contraflow:
        # Find Skyway edges and activate contraflow (doubling outbound capacity)
        skyway_edges = [
            i for i, e in enumerate(road_graph.edges)
            if "skyway" in e.name.lower()
        ]
        congestion_model.apply_contraflow(skyway_edges, capacity_multiplier=2.0)

    router = TimeDependentRouter(road_graph=road_graph)
    engine = TimeSlicedAssignmentEngine(
        road_graph=road_graph,
        demand_model=demand_model,
        congestion_model=congestion_model,
        router=router,
    )

    # Free fire arrival (no road cutoffs to measure pure traffic clearance)
    edge_arrivals = np.full(len(road_graph.edges), np.inf, dtype=np.float64)
    destinations = set(road_graph.boundary_destinations.values())

    summaries, _ = engine.run_assignment(
        edge_fire_arrivals=edge_arrivals,
        destination_node_ids=destinations,
        apply_exit_capacities=True,
    )

    paradise_zones = ["Paradise_North", "Paradise_Central", "Paradise_East", "Paradise_South"]
    paradise_clearance_min = max(summaries[z].clearance_time_min for z in paradise_zones)
    paradise_clearance_hrs = paradise_clearance_min / 60.0

    zone_hours = {z: summaries[z].clearance_time_hours for z in paradise_zones}
    return paradise_clearance_hrs, zone_hours


def main():
    print("Loading RoadGraph for Camp Fire domain...")
    road_graph = load_camp_fire_road_graph(full_detail=True)
    print(f"Loaded graph: {len(road_graph.nodes)} nodes, {len(road_graph.edges)} edges.")

    # 1. Baseline scenario
    base_hrs, base_zones = run_single_scenario(road_graph)
    print(f"\nBaseline Clearance (scale=1.0, inbound=0, nominal exits, contraflow=OFF): {base_hrs:.2f} h")

    scenarios = [
        ("Baseline (Nominal)", 1.0, 0.0, None, False),
        # Capacity scale
        ("Capacity Scale = 0.7 (-30% link cap)", 0.7, 0.0, None, False),
        ("Capacity Scale = 0.5 (-50% link cap)", 0.5, 0.0, None, False),
        # Inbound traffic
        ("Inbound Traffic = 200 veh/hr", 1.0, 200.0, None, False),
        # Exit capacities (-30% individual reductions)
        ("Skyway Exit Cap -30% (1,540 veh/hr)", 1.0, 0.0, {"skyway_south_exit": 1540.0, "fallback_exit": 700.0}, False),
        ("Clark Exit Cap -30% (980 veh/hr)", 1.0, 0.0, {"clark_south_exit": 980.0}, False),
        ("Pentz Exit Cap -30% (700 veh/hr)", 1.0, 0.0, {"pentz_south_exit": 700.0}, False),
        ("All Exits -30% Outflow", 1.0, 0.0, {"skyway_south_exit": 1540.0, "clark_south_exit": 980.0, "pentz_south_exit": 700.0, "fallback_exit": 700.0}, False),
        # Contraflow
        ("Skyway Contraflow = ON (2x capacity)", 1.0, 0.0, None, True),
        # Extreme Congestion Compound
        ("Compound Worst-Case (Scale 0.5 + Inbound 200 + Exits -30%)", 0.5, 200.0, {"skyway_south_exit": 1540.0, "clark_south_exit": 980.0, "pentz_south_exit": 700.0, "fallback_exit": 700.0}, False),
    ]

    results = []
    for label, cap_scale, inbound, exits, contra in scenarios:
        hrs, z_hrs = run_single_scenario(
            road_graph,
            capacity_scale=cap_scale,
            inbound_traffic=inbound,
            exit_capacities=exits,
            skyway_contraflow=contra,
        )
        delta_hrs = hrs - base_hrs
        results.append({
            "Scenario / Parameter": label,
            "Clearance (h)": round(hrs, 2),
            "Delta vs Baseline (h)": round(delta_hrs, 2),
            "Paradise North (h)": round(z_hrs["Paradise_North"], 2),
            "Paradise Central (h)": round(z_hrs["Paradise_Central"], 2),
            "Paradise East (h)": round(z_hrs["Paradise_East"], 2),
            "Paradise South (h)": round(z_hrs["Paradise_South"], 2),
        })

    df = pd.DataFrame(results)
    out_csv = Path("reports/paradise_clearance_tornado_sensitivity.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved tornado sensitivity table to {out_csv}")
    print("\n" + "=" * 95)
    print("PARADISE EVACUATION CLEARANCE TIME TORNADO SENSITIVITY TABLE")
    print("=" * 95)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
