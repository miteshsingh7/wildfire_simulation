"""Rerun Agent A2 over 5 Teacher v2 seeds (389, 117, 304, 100, 236).

Addresses Step 4:
- Diagnoses and documents the seed invariance bug.
- Runs A2 under both:
  1. Causal assimilation (assimilate=True, as in 6D audit)
  2. Pure unassimilated forward teacher physics (assimilate=False)
- Reports combined misses, trapped origins, and abandoned vehicles encounters.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from evaluate_heldout_camp_fire import (
    TeacherParams,
    build_camp_fire_dataset_60m,
)
from wf052.sim.ember_kernel import EmberParams, simulate_teacher_with_embers_60m
from eval_replay_6d_audit import count_abandoned_vehicles_for_route
from wf052.replay.contracts import ReplayOriginResult
from wf052.replay.harness import CausalReplayHarness
from wf052.replay.methods import (
    run_agent_simulation,
    sample_block_origins,
)
from wf052.replay.providers import DynamicTeacherV2Provider, TeacherFieldProvider
from wf052.routing.graph import load_camp_fire_road_graph


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("STEP 4: RERUN AGENT A2 OVER 5 TEACHER V2 SEEDS")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Camp Fire dataset
    ds_camp, bbox, transform = build_camp_fire_dataset_60m(device)
    h, w = ds_camp.fbfm_arr.shape

    # Load Road Graph
    teacher_field = TeacherFieldProvider().get_field(0.0)
    gx = teacher_field["x"].values
    gy = teacher_field["y"].values

    road_graph = load_camp_fire_road_graph(
        full_detail=True,
        grid_x_coords=gx,
        grid_y_coords=gy,
    )
    destinations = set(road_graph.boundary_destinations.values())
    origins = sample_block_origins(road_graph=road_graph, num_samples=200, seed=42)

    # Truth Hazards
    harness_base = CausalReplayHarness(road_graph=road_graph, provider=TeacherFieldProvider(), latency_min=0.0)
    truth_hazards = harness_base.truth_hazards

    test_seeds = [389, 117, 304, 100, 236]

    t_params = TeacherParams(
        p_base=0.22,
        a=0.06,
        c_1=0.045,
        c_2=0.15,
        p_continue=0.30,
        fuel_scale_grass_shrub=1.0,
        fuel_scale_timber=1.0,
        canopy_den_scale=1.0,
        wind_moisture_damping=0.0,
    )
    e_params = EmberParams(
        alpha_launch=0.0040,
        beta_dist_m_per_ms=110.0,
        p_urban_wui=0.15,
    )

    rows = []

    for s in test_seeds:
        print(f"\n--- Simulating Teacher v2 Seed {s} (12h, 30 steps/h) ---")
        t_sim_start = time.time()
        sim_s = simulate_teacher_with_embers_60m(
            ds_camp,
            t_params,
            ember_params=e_params,
            use_ember_kernel=True,
            duration_hours=12,
            steps_per_hour=30,
            seed=s,
        )
        sim_duration = time.time() - t_sim_start
        arr_s = sim_s["arrival_time_min"]

        # Calculate burned area
        burned_cells = int(np.sum(np.isfinite(arr_s) & (arr_s >= 0.0)))
        area_ha = burned_cells * (60.0 * 60.0) / 10000.0
        print(f"Seed {s} burned area: {area_ha:.1f} ha ({burned_cells} cells) in {sim_duration:.1f}s")

        # 1. Rerun A2 WITH CAUSAL ASSIMILATION (Standard A2)
        prov_assim = DynamicTeacherV2Provider(arr_s, gx, gy, assimilate=True)
        h_assim = CausalReplayHarness(road_graph=road_graph, provider=prov_assim, latency_min=0.0)
        res_assim, lat_assim = run_agent_simulation(f"A2_assim_seed_{s}", origins, road_graph, destinations, truth_hazards, h_assim)

        miss_assim = sum(1 for r in res_assim if r.hazard_miss)
        trap_assim = sum(1 for r in res_assim if r.route is None or not r.route.edges)
        av_routed_assim = [
            count_abandoned_vehicles_for_route(r.route, truth_hazards, road_graph)[0]
            for r in res_assim
            if r.route is not None and r.route.edges
        ]

        rows.append({
            "teacher_seed": s,
            "mode": "assimilated (standard A2)",
            "area_ha": round(area_ha, 1),
            "combined_misses": f"{miss_assim}/200 ({miss_assim/2.0:.1f}%)",
            "trapped_count": f"{trap_assim}/200 ({trap_assim/2.0:.1f}%)",
            "av_routed_mean": round(float(np.mean(av_routed_assim)), 3) if av_routed_assim else 0.0,
            "av_routed_median": round(float(np.median(av_routed_assim)), 1) if av_routed_assim else 0.0,
            "av_routed_p90": round(float(np.percentile(av_routed_assim, 90.0)), 1) if av_routed_assim else 0.0,
        })

        # 2. Rerun A2 WITHOUT ASSIMILATION (Pure forward physics)
        prov_pure = DynamicTeacherV2Provider(arr_s, gx, gy, assimilate=False)
        h_pure = CausalReplayHarness(road_graph=road_graph, provider=prov_pure, latency_min=0.0)
        res_pure, lat_pure = run_agent_simulation(f"A2_pure_seed_{s}", origins, road_graph, destinations, truth_hazards, h_pure)

        miss_pure = sum(1 for r in res_pure if r.hazard_miss)
        trap_pure = sum(1 for r in res_pure if r.route is None or not r.route.edges)
        av_routed_pure = [
            count_abandoned_vehicles_for_route(r.route, truth_hazards, road_graph)[0]
            for r in res_pure
            if r.route is not None and r.route.edges
        ]

        rows.append({
            "teacher_seed": s,
            "mode": "pure physics (unassimilated)",
            "area_ha": round(area_ha, 1),
            "combined_misses": f"{miss_pure}/200 ({miss_pure/2.0:.1f}%)",
            "trapped_count": f"{trap_pure}/200 ({trap_pure/2.0:.1f}%)",
            "av_routed_mean": round(float(np.mean(av_routed_pure)), 3) if av_routed_pure else 0.0,
            "av_routed_median": round(float(np.median(av_routed_pure)), 1) if av_routed_pure else 0.0,
            "av_routed_p90": round(float(np.percentile(av_routed_pure, 90.0)), 1) if av_routed_pure else 0.0,
        })

    df = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    out_csv = Path("reports/step4_rerun_a2_5seeds.csv")
    df.to_csv(out_csv, index=False)
    print("\n" + "=" * 80)
    print(f"STEP 4 RESULTS SAVED TO {out_csv}")
    print("=" * 80)
    print(df.to_string(index=False))
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
