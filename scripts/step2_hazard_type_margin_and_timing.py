#!/usr/bin/env python3
"""Phase 4D Step 2: Hazard-Type Margins and Timing Error Analysis.

Evaluates:
2b. Margin (ii):
    - hazards persist after onset
    - margin = onset minus time vehicle ENTERS the edge (re.entry_time_min)
    - assign each hazard to its nearest route edge only (no cross product)
    - sensitivity: matching radius 30, 60, and 120 m
    - report per method (7 methods) and per type (burnover, roadblock, abandoned_vehicle, all):
      origins affected out of 200, negatives, median and P10 over affected origins, trapped counts.
    - Burnover-only is the primary table.

2d. Timing:
    - per type and corridor over UNIQUE hazards (dedupe by hazard_id):
      n, n with finite teacher arrival, median signed difference (teacher minus truth, minutes)
      over finite ones, share never-burn, share teacher later.
    - Under Seed 42 and 5-seed ensemble median [389, 117, 304, 100, 236].
"""

from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from evaluate_heldout_camp_fire import (
    TeacherParams,
    build_camp_fire_dataset_60m,
)
from wf052.replay.contracts import ReplayOriginResult, TruthHazard
from wf052.replay.harness import CausalReplayHarness
from wf052.replay.methods import (
    run_agent_simulation,
    run_reactive_baseline,
    run_static_baseline,
    sample_block_origins,
)
from wf052.replay.providers import (
    ObservationInflatedProvider,
    TeacherFieldProvider,
    TeacherV2FieldProvider,
)
from wf052.routing.graph import RoadGraph, load_camp_fire_road_graph
from wf052.sim.ember_kernel import (
    EmberParams,
    simulate_teacher_with_embers_60m,
)
from wf052.surrogate.standin_generator import load_and_verify_frozen_teacher_v2


def compute_hazard_type_margins(
    results: list[ReplayOriginResult],
    origins: list,
    hazards: list[TruthHazard],
    road_graph: RoadGraph,
    radius_m: float,
    hazard_type: str | None = None,
) -> dict:
    """Compute Margin (ii) for a specific hazard type and radius.
    
    Assigns each hazard to its nearest route edge only.
    margin = hazard.onset_elapsed_min - nearest_edge.entry_time_min.
    """
    edge_map = {(e.u, e.v, e.key): e for e in road_graph.edges}
    
    # Filter hazards if hazard_type is specified
    if hazard_type and hazard_type != "all":
        target_hazards = [h for h in hazards if h.hazard_type == hazard_type]
    else:
        target_hazards = hazards

    affected_margins = []
    trapped_count = 0

    for i, res in enumerate(results):
        orig = origins[i]
        r = res.route
        if r is None or not r.edges:
            trapped_count += 1
            continue

        matched_margins = []
        for h in target_hazards:
            min_d = float("inf")
            best_entry = None
            for re in r.edges:
                ed = edge_map.get((re.u, re.v, re.key))
                if ed and ed.geometry and not ed.geometry.is_empty:
                    d = float(ed.geometry.distance(h.geometry_utm))
                    if d < min_d:
                        min_d = d
                        best_entry = re.entry_time_min

            if min_d <= radius_m and best_entry is not None:
                matched_margins.append(h.onset_elapsed_min - best_entry)

        if matched_margins:
            affected_margins.append(min(matched_margins))

    n_affected = len(affected_margins)
    n_negatives = sum(1 for m in affected_margins if m < 0.0)
    med_val = float(np.median(affected_margins)) if affected_margins else float("nan")
    p10_val = float(np.percentile(affected_margins, 10.0)) if affected_margins else float("nan")

    return {
        "radius_m": radius_m,
        "hazard_type": hazard_type if hazard_type else "all",
        "affected_count": n_affected,
        "negatives_count": n_negatives,
        "median_min": round(med_val, 1) if not np.isnan(med_val) else "N/A",
        "p10_min": round(p10_val, 1) if not np.isnan(p10_val) else "N/A",
        "trapped_count": trapped_count,
    }


def main() -> None:
    print("=" * 80)
    print("PHASE 4D STEP 2: HAZARD-TYPE MARGINS AND TIMING ANALYSIS")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Road Graph
    prov_seed42 = TeacherV2FieldProvider(assimilate=False)
    tf2 = prov_seed42.get_field(0.0)
    gx, gy = tf2["x"].values, tf2["y"].values
    road_graph = load_camp_fire_road_graph(full_detail=True, grid_x_coords=gx, grid_y_coords=gy)
    destinations = set(road_graph.boundary_destinations.values())
    origins = sample_block_origins(road_graph=road_graph, num_samples=200, seed=42)

    # Truth Hazards
    harness_base = CausalReplayHarness(road_graph=road_graph, provider=prov_seed42, latency_min=0.0)
    truth_hazards = harness_base.truth_hazards
    print(f"Loaded road graph: {len(road_graph.nodes)} nodes, {len(road_graph.edges)} edges")
    print(f"Loaded {len(origins)} evacuation origins, {len(truth_hazards)} empirical truth hazards")

    # =========================================================================
    # SIMULATE 7 AGENTS
    # =========================================================================
    print("\n--- Running Simulations for 7 Agents ---")
    print("1/7: Simulating A0 static...")
    res_a0, _ = run_static_baseline(origins, road_graph, destinations, truth_hazards)

    print("2/7: Simulating A1 reactive...")
    res_a1, _ = run_reactive_baseline(origins, road_graph, destinations, truth_hazards, harness_base)

    print("3/7: Simulating A (teacher_v1)...")
    prov_v1 = TeacherFieldProvider()
    harness_v1 = CausalReplayHarness(road_graph=road_graph, provider=prov_v1, latency_min=0.0)
    res_a_v1, _ = run_agent_simulation("A_teacher_v1", origins, road_graph, destinations, truth_hazards, harness_v1)

    print("4/7: Simulating A2-assim...")
    prov_v2_assim = TeacherV2FieldProvider(assimilate=True)
    harness_v2_assim = CausalReplayHarness(road_graph=road_graph, provider=prov_v2_assim, latency_min=0.0)
    res_a2_assim, _ = run_agent_simulation("A2-assim", origins, road_graph, destinations, truth_hazards, harness_v2_assim)

    print("5/7: Simulating A2-pure...")
    harness_v2_pure = CausalReplayHarness(road_graph=road_graph, provider=prov_seed42, latency_min=0.0)
    res_a2_pure, _ = run_agent_simulation("A2-pure", origins, road_graph, destinations, truth_hazards, harness_v2_pure)

    print("6/7: Simulating B on A2-assim...")
    prov_b_assim = ObservationInflatedProvider(base_teacher_provider=prov_v2_assim, r_spot_km=4.0)
    harness_b_assim = CausalReplayHarness(road_graph=road_graph, provider=prov_b_assim, latency_min=0.0)
    res_b_assim, _ = run_agent_simulation("B_on_A2-assim", origins, road_graph, destinations, truth_hazards, harness_b_assim)

    print("7/7: Simulating B on A2-pure...")
    prov_b_pure = ObservationInflatedProvider(base_teacher_provider=prov_seed42, r_spot_km=4.0)
    harness_b_pure = CausalReplayHarness(road_graph=road_graph, provider=prov_b_pure, latency_min=0.0)
    res_b_pure, _ = run_agent_simulation("B_on_A2-pure", origins, road_graph, destinations, truth_hazards, harness_b_pure)

    all_agents = [
        ("A0 static", res_a0),
        ("A1 reactive", res_a1),
        ("A (teacher_v1)", res_a_v1),
        ("A2-assim", res_a2_assim),
        ("A2-pure", res_a2_pure),
        ("B on A2-assim", res_b_assim),
        ("B on A2-pure", res_b_pure),
    ]

    # =========================================================================
    # 2b. MARGIN (ii) ACROSS HAZARD TYPES & RADII
    # =========================================================================
    print("\n--- Computing Hazard-Type Margins across Radii (30m, 60m, 120m) ---")
    radii = [30.0, 60.0, 120.0]
    hazard_types = ["burnover", "roadblock", "abandoned_vehicle", "all"]

    margin_rows = []
    for rad in radii:
        for ag_name, res_list in all_agents:
            for ht in hazard_types:
                m_info = compute_hazard_type_margins(
                    results=res_list,
                    origins=origins,
                    hazards=truth_hazards,
                    road_graph=road_graph,
                    radius_m=rad,
                    hazard_type=ht,
                )
                margin_rows.append({
                    "radius_m": int(rad),
                    "method": ag_name,
                    "hazard_type": ht,
                    "affected_out_of_200": f"{m_info['affected_count']}/200",
                    "negatives": f"{m_info['negatives_count']}/{m_info['affected_count']}",
                    "median_margin_min": m_info["median_min"],
                    "p10_margin_min": m_info["p10_min"],
                    "trapped_out_of_200": f"{m_info['trapped_count']}/200",
                })

    df_margins = pd.DataFrame(margin_rows)
    out_margins_csv = Path("reports/step2_hazard_type_margins.csv")
    df_margins.to_csv(out_margins_csv, index=False)
    print(f"Saved margin tables to {out_margins_csv}")

    # Print PRIMARY TABLE: Burnover-only (120m, 60m, 30m)
    print("\n" + "=" * 80)
    print("PRIMARY TABLE: BURNOVER-ONLY MARGINS (23 Incidents)")
    print("=" * 80)
    df_bo = df_margins[df_margins["hazard_type"] == "burnover"]
    print(df_bo.to_string(index=False))

    # Print ROADBLOCKS TABLE
    print("\n" + "=" * 80)
    print("SECONDARY TABLE: ROADBLOCKS MARGINS (178 Records)")
    print("=" * 80)
    df_rb = df_margins[df_margins["hazard_type"] == "roadblock"]
    print(df_rb.to_string(index=False))

    # Print ABANDONED VEHICLES TABLE
    print("\n" + "=" * 80)
    print("SECONDARY TABLE: ABANDONED VEHICLES MARGINS (235 Records)")
    print("=" * 80)
    df_av = df_margins[df_margins["hazard_type"] == "abandoned_vehicle"]
    print(df_av.to_string(index=False))

    # =========================================================================
    # 2d. TIMING PER TYPE AND CORRIDOR OVER UNIQUE HAZARDS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2d: TIMING ERROR PER TYPE & CORRIDOR OVER UNIQUE HAZARDS")
    print("=" * 80)

    # 1. Edge arrivals under Seed 42
    truth_edge_arrivals_s42 = road_graph.compute_edge_fire_arrivals(tf2)

    # 2. Simulate 5 Teacher v2 seeds on Camp Fire domain to get ensemble median
    print("Simulating 5 Teacher v2 seeds on Camp Fire domain for ensemble median...")
    _, _, _, t_params, e_params = load_and_verify_frozen_teacher_v2("configs/teacher_v2.yaml")
    ds_camp, _, _ = build_camp_fire_dataset_60m(device)

    ensemble_seeds = [389, 117, 304, 100, 236]
    ensemble_edge_arrivals = []

    for s in ensemble_seeds:
        sim_s = simulate_teacher_with_embers_60m(
            ds=ds_camp,
            params=t_params,
            ember_params=e_params,
            use_ember_kernel=True,
            duration_hours=12,
            steps_per_hour=30,
            seed=s,
        )
        arr_s = sim_s["arrival_time_min"]
        arr_s_clean = np.where(np.isfinite(arr_s) & (arr_s >= 0.0), arr_s, np.inf)
        ds_s = tf2.copy(deep=True)
        ds_s["arrival_q10"].values[:] = arr_s_clean
        ds_s["arrival_q50"].values[:] = arr_s_clean
        ds_s["arrival_q90"].values[:] = arr_s_clean
        ea_s = road_graph.compute_edge_fire_arrivals(ds_s)
        ensemble_edge_arrivals.append(ea_s)

    stacked_ea = np.stack(ensemble_edge_arrivals, axis=0)  # (5, num_edges)
    ensemble_median_ea = np.median(stacked_ea, axis=0)

    # 3. Associate UNIQUE hazards with corridors via A0 static routes
    edge_map = {(e.u, e.v, e.key): (i, e) for i, e in enumerate(road_graph.edges)}
    corridor_nodes = {
        86429619: "Clark Rd",
        2611113946: "Skyway",
        86469871: "Pentz/Hwy 70",
        10131737665: "Pentz/Hwy 70",
    }

    # corridor -> hazard_id -> dict of info
    corridor_unique_hazards = defaultdict(dict)

    for res in res_a0:
        r = res.route
        if r and r.edges:
            dest_v = r.edges[-1].v
            c_name = corridor_nodes.get(dest_v, "Pentz/Hwy 70")
            for re in r.edges:
                edge_info = edge_map.get((re.u, re.v, re.key))
                if edge_info is None:
                    continue
                edge_idx, ed = edge_info
                if not ed.geometry or ed.geometry.is_empty:
                    continue

                for h in truth_hazards:
                    d = float(ed.geometry.distance(h.geometry_utm))
                    if d <= 120.0:
                        if (
                            h.hazard_id not in corridor_unique_hazards[c_name]
                            or d < corridor_unique_hazards[c_name][h.hazard_id]["dist_m"]
                        ):
                            corridor_unique_hazards[c_name][h.hazard_id] = {
                                "hazard_id": h.hazard_id,
                                "hazard_type": h.hazard_type,
                                "onset_min": h.onset_elapsed_min,
                                "edge_idx": edge_idx,
                                "dist_m": d,
                                "t_s42": truth_edge_arrivals_s42[edge_idx],
                                "t_ens": ensemble_median_ea[edge_idx],
                            }

    timing_rows = []
    for c_name in ["Clark Rd", "Skyway", "Pentz/Hwy 70"]:
        h_dict = corridor_unique_hazards[c_name]
        for ht in ["burnover", "roadblock", "abandoned_vehicle", "total"]:
            if ht == "total":
                haz_list = list(h_dict.values())
            else:
                haz_list = [h for h in h_dict.values() if h["hazard_type"] == ht]

            n_haz = len(haz_list)
            if n_haz == 0:
                timing_rows.append({
                    "corridor": c_name,
                    "hazard_type": ht,
                    "n_unique": 0,
                    "s42_n_finite": 0,
                    "s42_med_signed_diff_min": "N/A",
                    "s42_share_never_burn": "N/A",
                    "s42_share_teacher_later": "N/A",
                    "ens_n_finite": 0,
                    "ens_med_signed_diff_min": "N/A",
                    "ens_share_never_burn": "N/A",
                    "ens_share_teacher_later": "N/A",
                })
                continue

            # Seed 42 analysis
            s42_finite_diffs = []
            s42_never = 0
            s42_later = 0
            for h in haz_list:
                t_p = h["t_s42"]
                t_t = h["onset_min"]
                if np.isnan(t_p) or np.isposinf(t_p) or np.isneginf(t_p):
                    s42_never += 1
                else:
                    diff = t_p - t_t
                    s42_finite_diffs.append(diff)
                    if diff > 0.0:
                        s42_later += 1

            s42_n_fin = len(s42_finite_diffs)
            s42_med_diff = (
                round(float(np.median(s42_finite_diffs)), 1) if s42_finite_diffs else "N/A"
            )
            s42_never_share = f"{(s42_never / n_haz) * 100.0:.1f}%"
            s42_later_share = f"{(s42_later / n_haz) * 100.0:.1f}%"

            # 5-Seed Ensemble Median analysis
            ens_finite_diffs = []
            ens_never = 0
            ens_later = 0
            for h in haz_list:
                t_p = h["t_ens"]
                t_t = h["onset_min"]
                if np.isnan(t_p) or np.isposinf(t_p) or np.isneginf(t_p):
                    ens_never += 1
                else:
                    diff = t_p - t_t
                    ens_finite_diffs.append(diff)
                    if diff > 0.0:
                        ens_later += 1

            ens_n_fin = len(ens_finite_diffs)
            ens_med_diff = (
                round(float(np.median(ens_finite_diffs)), 1) if ens_finite_diffs else "N/A"
            )
            ens_never_share = f"{(ens_never / n_haz) * 100.0:.1f}%"
            ens_later_share = f"{(ens_later / n_haz) * 100.0:.1f}%"

            timing_rows.append({
                "corridor": c_name,
                "hazard_type": ht,
                "n_unique": n_haz,
                "s42_n_finite": s42_n_fin,
                "s42_med_signed_diff_min": s42_med_diff,
                "s42_share_never_burn": s42_never_share,
                "s42_share_teacher_later": s42_later_share,
                "ens_n_finite": ens_n_fin,
                "ens_med_signed_diff_min": ens_med_diff,
                "ens_share_never_burn": ens_never_share,
                "ens_share_teacher_later": ens_later_share,
            })

    df_timing = pd.DataFrame(timing_rows)
    out_timing_csv = Path("reports/step2_timing_by_hazard_type.csv")
    df_timing.to_csv(out_timing_csv, index=False)

    print("\n" + "=" * 80)
    print("TIMING ERROR TABLE PER TYPE AND CORRIDOR (OVER UNIQUE HAZARDS)")
    print("=" * 80)
    print(df_timing.to_string(index=False))
    print(f"\nSaved timing breakdown to {out_timing_csv}")


if __name__ == "__main__":
    main()
