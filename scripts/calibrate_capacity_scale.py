"""Single-parameter calibration check for capacity_scale against empirical NIST corridor travel times (Phase 4e).

Empirical Observations (NIST TN 2252 sup Table A-2):
1. 08:45 PST (T+132 min): Bille Rd to Skyway to Almond St to PPD (Source TD-078, p. 29).
   - "normally 5 min, but took 1 hour [60 min]"
2. 09:00 PST (T+147 min): Kmart (Clark Rd at Wagstaff Rd) to Almond St (Source TD-053, p. 35).
   - "two hours [120 min] to go from Kmart to Almond St"

Procedure:
- Evaluates candidate capacity_scale values: [1.0, 0.8, 0.7, 0.6, 0.5, 0.4].
- Computes dynamic congested traversal times on both empirical corridors.
- Identifies best-fit capacity_scale minimizing root mean squared error (RMSE).
- Reports the calibration fit and the resulting Town of Paradise clearance range,
  explicitly labeled as a qualitative planning assessment.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyproj

from wf052.routing.congestion import CongestionModel, TimeSlicedAssignmentEngine
from wf052.routing.contracts import RoutingMode
from wf052.routing.demand import EvacuationDemandModel
from wf052.routing.graph import RoadGraph, load_camp_fire_road_graph
from wf052.routing.router import TimeDependentRouter


def run_calibration_check(
    road_graph: RoadGraph, candidate_scales: list[float] | None = None
) -> pd.DataFrame:
    if candidate_scales is None:
        candidate_scales = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4]

    node_ids = np.array(list(road_graph.nodes.keys()), dtype=np.int64)
    node_coords = np.array([road_graph.nodes[nid] for nid in node_ids])
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)

    # 1. Bille Rd at Skyway (Lat 39.764646, Lon -121.618712)
    bille_x, bille_y = transformer.transform(-121.618712, 39.764646)
    nid_bille = int(
        node_ids[np.argmin(np.hypot(node_coords[:, 0] - bille_x, node_coords[:, 1] - bille_y))]
    )

    # 2. Kmart at Clark Rd / Wagstaff Rd (Lat 39.777696, Lon -121.590550)
    kmart_x, kmart_y = transformer.transform(-121.590550, 39.777696)
    nid_kmart = int(
        node_ids[np.argmin(np.hypot(node_coords[:, 0] - kmart_x, node_coords[:, 1] - kmart_y))]
    )

    # 3. Almond St / PPD (Lat 39.756200, Lon -121.621200)
    almond_x, almond_y = transformer.transform(-121.621200, 39.756200)
    nid_almond = int(
        node_ids[np.argmin(np.hypot(node_coords[:, 0] - almond_x, node_coords[:, 1] - almond_y))]
    )

    print(f"Calibration corridor nodes: Bille={nid_bille}, Kmart={nid_kmart}, Almond={nid_almond}")

    obs_bille_t = 60.0  # 60.0 min (1.0 hr) at 08:45 PST (T+132 min)
    obs_kmart_t = 120.0  # 120.0 min (2.0 hr) at 09:00 PST (T+147 min)

    destinations = set(road_graph.boundary_destinations.values())
    edge_arrivals = np.full(len(road_graph.edges), np.inf, dtype=np.float64)

    results = []

    for scale in candidate_scales:
        print(f"\nEvaluating capacity_scale = {scale:.2f}...")
        demand_model = EvacuationDemandModel(
            config_path="configs/evacuation_demand.yaml", road_graph=road_graph
        )
        congestion_model = CongestionModel(
            road_graph=road_graph,
            config_path="configs/evacuation_demand.yaml",
            use_point_queue=True,
            capacity_scale=scale,
        )
        router = TimeDependentRouter(road_graph=road_graph)
        engine = TimeSlicedAssignmentEngine(
            road_graph=road_graph,
            demand_model=demand_model,
            congestion_model=congestion_model,
            router=router,
        )

        # Run assignment through peak morning period (up to 180 min / 09:33 PST)
        summaries, _ = engine.run_assignment(
            edge_fire_arrivals=edge_arrivals,
            destination_node_ids=destinations,
            apply_exit_capacities=True,
        )

        paradise_zones = [z for z in summaries.keys() if z.startswith("Zone_") or "Paradise" in z]
        paradise_clearance_hrs = max(summaries[z].clearance_time_hours for z in paradise_zones)

        # 1. Traversal time Bille -> Almond at 08:45 PST (T+132 min, nearest slice 130.0 min)
        t_bille = 132.0
        t_bille_slice = 130.0
        dyn_edge_times_bille = np.zeros(len(road_graph.edges), dtype=np.float64)
        for e_idx in range(len(road_graph.edges)):
            cap_e = congestion_model.edge_capacities[e_idx].capacity_veh_per_hr
            q_e = engine.last_edge_queues_over_time[e_idx].get(t_bille_slice, 0.0)
            q_del = (q_e / cap_e) * 60.0 if cap_e > 0 else 0.0
            dyn_edge_times_bille[e_idx] = (
                congestion_model.edge_capacities[e_idx].free_traversal_time_min + q_del
            )

        route_b = router.find_route(
            origin_node=nid_bille,
            destination_nodes={nid_almond},
            t0=t_bille,
            edge_fire_arrivals=edge_arrivals,
            mode=RoutingMode.FASTEST_SAFE,
            dynamic_edge_traversal_times=dyn_edge_times_bille,
        )
        sim_bille_t = (route_b.arrival_time_min - t_bille) if route_b else float("inf")

        # 2. Traversal time Kmart -> Almond at 09:00 PST (T+147 min, nearest slice 145.0 min)
        t_kmart = 147.0
        t_kmart_slice = 145.0
        dyn_edge_times_kmart = np.zeros(len(road_graph.edges), dtype=np.float64)
        for e_idx in range(len(road_graph.edges)):
            cap_e = congestion_model.edge_capacities[e_idx].capacity_veh_per_hr
            q_e = engine.last_edge_queues_over_time[e_idx].get(t_kmart_slice, 0.0)
            q_del = (q_e / cap_e) * 60.0 if cap_e > 0 else 0.0
            dyn_edge_times_kmart[e_idx] = (
                congestion_model.edge_capacities[e_idx].free_traversal_time_min + q_del
            )

        route_k = router.find_route(
            origin_node=nid_kmart,
            destination_nodes={nid_almond},
            t0=t_kmart,
            edge_fire_arrivals=edge_arrivals,
            mode=RoutingMode.FASTEST_SAFE,
            dynamic_edge_traversal_times=dyn_edge_times_kmart,
        )
        sim_kmart_t = (route_k.arrival_time_min - t_kmart) if route_k else float("inf")

        err_bille = sim_bille_t - obs_bille_t
        err_kmart = sim_kmart_t - obs_kmart_t
        rmse = float(np.sqrt(0.5 * (err_bille**2 + err_kmart**2)))

        print(
            f"scale={scale:.2f} | Bille->Almond: sim={sim_bille_t:.1f}m (obs=60.0m) | Kmart->Almond: sim={sim_kmart_t:.1f}m (obs=120.0m) | RMSE={rmse:.1f}m | Paradise Clearance={paradise_clearance_hrs:.1f}h"
        )

        results.append(
            {
                "capacity_scale": scale,
                "sim_bille_almond_min": round(sim_bille_t, 1),
                "obs_bille_almond_min": obs_bille_t,
                "err_bille_min": round(err_bille, 1),
                "sim_kmart_almond_min": round(sim_kmart_t, 1),
                "obs_kmart_almond_min": obs_kmart_t,
                "err_kmart_min": round(err_kmart, 1),
                "rmse_min": round(rmse, 1),
                "paradise_clearance_hours": round(paradise_clearance_hrs, 2),
                "status_label": "qualitative planning estimate",
            }
        )

    df = pd.DataFrame(results)
    out_csv = Path("reports/capacity_scale_calibration.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved capacity_scale calibration table to {out_csv}")
    print("\n" + "=" * 115)
    print("CAPACITY_SCALE CALIBRATION AGAINST EMPIRICAL NIST CORRIDOR TRAVEL TIMES")
    print("=" * 115)
    print(df.to_string(index=False))

    best_row = df.loc[df["rmse_min"].idxmin()]
    print(
        f"\nBest-fit single parameter: capacity_scale = {best_row['capacity_scale']:.2f} (RMSE: {best_row['rmse_min']:.1f} min)"
    )
    print(
        f"Resulting qualitative Paradise clearance range: {df['paradise_clearance_hours'].min():.1f}h to {df['paradise_clearance_hours'].max():.1f}h (nominal calibrated: {best_row['paradise_clearance_hours']:.1f}h)"
    )

    return df


def main():
    print("Loading full road graph for capacity_scale calibration check...")
    road_graph = load_camp_fire_road_graph(full_detail=True)
    run_calibration_check(road_graph, candidate_scales=[1.0, 0.8, 0.7, 0.6, 0.5])


if __name__ == "__main__":
    main()
