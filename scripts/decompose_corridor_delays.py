"""Decompose corridor travel times into free-flow, queue delay, and spillback penalty."""

import numpy as np
import pandas as pd

from wf052.routing.congestion import CongestionModel, TimeSlicedAssignmentEngine
from wf052.routing.demand import EvacuationDemandModel
from wf052.routing.graph import load_camp_fire_road_graph
from wf052.routing.router import TimeDependentRouter


def decompose_delays() -> None:
    print("Loading road graph...")
    road_graph = load_camp_fire_road_graph(full_detail=True)
    demand_model = EvacuationDemandModel(road_graph=road_graph)
    congestion_model = CongestionModel(road_graph=road_graph, use_point_queue=True)
    router = TimeDependentRouter(road_graph=road_graph)
    engine = TimeSlicedAssignmentEngine(
        road_graph=road_graph,
        demand_model=demand_model,
        congestion_model=congestion_model,
        router=router,
    )

    arrivals = np.full(len(road_graph.edges), np.inf, dtype=np.float64)
    dests = set(road_graph.boundary_destinations.values())

    print("Running baseline assignment...")
    summaries, _ = engine.run_assignment(
        edge_fire_arrivals=arrivals,
        destination_node_ids=dests,
        apply_exit_capacities=True,
    )

    corridors = ["Skyway", "Clark", "Pentz", "Neal", "Highway 70"]

    # Analyze recorded cohort trajectories and edge queues
    # For each corridor, aggregate vehicle-minutes of free-flow, pure queue delay, and spillback penalty
    corridor_stats = {
        c: {"free_veh_min": 0.0, "queue_veh_min": 0.0, "spill_veh_min": 0.0, "total_veh": 0}
        for c in corridors
    }

    # Map edges to corridor
    edge_corridor: dict[int, str] = {}
    for e in road_graph.edges:
        name_lower = str(e.name or "").lower()
        for c in corridors:
            c_low = c.lower()
            if c_low in name_lower or (
                c == "Highway 70" and ("ca-70" in name_lower or "ca 70" in name_lower)
            ):
                edge_corridor[e.edge_idx] = c
                break

    print(f"Mapped {len(edge_corridor)} edges to key corridors.")

    # Inspect cohort trajectories
    for cohort in engine.last_cohort_trajectories:
        cnt = cohort["num_vehicles"]
        for e_idx, t_in, t_out in cohort["intervals"]:
            if e_idx in edge_corridor:
                corr = edge_corridor[e_idx]
                e_info = congestion_model.edge_capacities[e_idx]
                t_free = e_info.free_traversal_time_min
                t_total = t_out - t_in
                t_delay = max(0.0, t_total - t_free)

                # From queue and storage at that link
                q = engine.last_edge_queues[e_idx]
                storage = e_info.storage_veh
                cap = max(10.0, e_info.capacity_veh_per_hr)

                # Proportions
                if q > storage:
                    # In our model: q_delay = (q / cap)*60; spill = ((q - storage)/cap)*60
                    # Ratio of spill to (q_delay + spill)
                    pure_q = (q / cap) * 60.0
                    spill = ((q - storage) / cap) * 60.0
                    tot_d = pure_q + spill
                    frac_spill = spill / tot_d if tot_d > 0 else 0.0
                    spill_del = t_delay * frac_spill
                    pure_q_del = t_delay * (1.0 - frac_spill)
                else:
                    spill_del = 0.0
                    pure_q_del = t_delay

                corridor_stats[corr]["free_veh_min"] += cnt * t_free
                corridor_stats[corr]["queue_veh_min"] += cnt * pure_q_del
                corridor_stats[corr]["spill_veh_min"] += cnt * spill_del
                corridor_stats[corr]["total_veh"] += cnt

    rows = []
    for c, s in corridor_stats.items():
        tot_time = s["free_veh_min"] + s["queue_veh_min"] + s["spill_veh_min"]
        free_pct = (s["free_veh_min"] / tot_time * 100.0) if tot_time > 0 else 0.0
        q_pct = (s["queue_veh_min"] / tot_time * 100.0) if tot_time > 0 else 0.0
        spill_pct = (s["spill_veh_min"] / tot_time * 100.0) if tot_time > 0 else 0.0
        rows.append(
            {
                "Corridor": c,
                "Free-Flow (veh-hr)": s["free_veh_min"] / 60.0,
                "Free-Flow (%)": free_pct,
                "Queue Delay (veh-hr)": s["queue_veh_min"] / 60.0,
                "Queue Delay (%)": q_pct,
                "Spillback Penalty (veh-hr)": s["spill_veh_min"] / 60.0,
                "Spillback Penalty (%)": spill_pct,
                "Total Travel Time (veh-hr)": tot_time / 60.0,
            }
        )

    df = pd.DataFrame(rows)
    print("\n--- CORRIDOR DELAY DECOMPOSITION ---")
    print(df.to_string(index=False))
    df.to_csv("reports/corridor_delay_decomposition.csv", index=False)


if __name__ == "__main__":
    decompose_delays()
