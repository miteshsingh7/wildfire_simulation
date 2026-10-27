#!/usr/bin/env python3
"""Phase 4C Step 2 & Step 3: Truth Margin vs NIST Hazards and Timing Error Analysis.

Evaluates:
STEP 2:
- Truth margin (ii): min over route edges of (NIST hazard onset on edge minus vehicle exit time on edge).
  Hazards = 436 NIST truth hazards (burnovers, roadblocks, abandoned vehicles).
  Zero teacher raster used for (ii).
- Evaluates 7 agents:
  1. A0 static
  2. A1 reactive
  3. A (teacher_v1)
  4. A2-assim
  5. A2-pure
  6. B on A2-assim
  7. B on A2-pure
- Reports:
  - count of routed origins with no hazard on route (reported separately, not infinite)
  - median over origins that have a hazard on route
  - P10 over origins that have a hazard on route (finite values only)
  - count of negative margins
  - trapped counts
  - predicted margin (i) (Not applicable for A0)
  - assert that P10 is -inf only if >= 10% of values are -inf.

STEP 3:
- For every truth hazard on a route, compares onset time with Teacher v2 arrival time
  at the same edge under seed 42 and 5-seed ensemble median [389, 117, 304, 100, 236].
- Reports per corridor (Clark Rd, Skyway, Pentz/Hwy 70):
  - n
  - median signed difference (predicted minus truth, minutes)
  - share where teacher arrival is later than truth onset
  - share never-burn
"""

from pathlib import Path
import numpy as np
import pandas as pd
import shapely.geometry
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


def compute_nist_truth_margins_for_agent(
    results: list[ReplayOriginResult],
    origins: list,
    hazards: list[TruthHazard],
    road_graph: RoadGraph,
    spatial_radius_m: float = 120.0,
) -> list[dict]:
    """Compute NIST empirical truth hazard margins for an agent's simulation results."""
    edge_map = {(e.u, e.v, e.key): e for e in road_graph.edges}
    records = []

    for i, res in enumerate(results):
        orig = origins[i]
        r = res.route
        if r is None or not r.edges:
            records.append({
                "origin_id": orig.origin_id,
                "trapped": True,
                "no_hazard": False,
                "margin_min": -float("inf"),
                "pred_margin_min": -float("inf"),
                "hazard_edges": [],
            })
            continue

        # Predicted margin (i) from agent's planned route
        pred_edge_margins = [e.fire_arrival_min - e.exit_time_min for e in r.edges]
        min_pred = min(pred_edge_margins) if pred_edge_margins else float("inf")

        # Truth margin (ii) against NIST empirical hazards
        edge_hazard_margins = []
        hazard_details = []

        for re in r.edges:
            ed = edge_map.get((re.u, re.v, re.key))
            geom = ed.geometry if ed is not None else None

            for h in hazards:
                if geom is not None and not geom.is_empty:
                    dist = float(geom.distance(h.geometry_utm))
                else:
                    dist = float("inf")

                if dist <= spatial_radius_m:
                    m = h.onset_elapsed_min - re.exit_time_min
                    edge_hazard_margins.append(m)
                    hazard_details.append({
                        "hazard_id": h.hazard_id,
                        "hazard_type": h.hazard_type,
                        "onset_elapsed_min": h.onset_elapsed_min,
                        "exit_time_min": re.exit_time_min,
                        "margin_min": m,
                        "edge_u": re.u,
                        "edge_v": re.v,
                        "edge_key": re.key,
                    })

        if not edge_hazard_margins:
            records.append({
                "origin_id": orig.origin_id,
                "trapped": False,
                "no_hazard": True,
                "margin_min": float("inf"),
                "pred_margin_min": min_pred,
                "hazard_edges": [],
            })
        else:
            records.append({
                "origin_id": orig.origin_id,
                "trapped": False,
                "no_hazard": False,
                "margin_min": min(edge_hazard_margins),
                "pred_margin_min": min_pred,
                "hazard_edges": hazard_details,
            })

    return records


def main() -> None:
    print("=" * 80)
    print("PHASE 4C STEP 2 & STEP 3: NIST MARGINS & TIMING ERROR AUDIT")
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
    # STEP 2: SIMULATE 7 AGENTS
    # =========================================================================
    print("\n--- Running Simulations for 7 Agents ---")
    # 1. A0 static
    print("1/7: Simulating A0 static...")
    res_a0, _ = run_static_baseline(origins, road_graph, destinations, truth_hazards)

    # 2. A1 reactive
    print("2/7: Simulating A1 reactive...")
    res_a1, _ = run_reactive_baseline(origins, road_graph, destinations, truth_hazards, harness_base)

    # 3. A (teacher_v1)
    print("3/7: Simulating A (teacher_v1)...")
    prov_v1 = TeacherFieldProvider()
    h_v1 = CausalReplayHarness(road_graph=road_graph, provider=prov_v1, latency_min=0.0)
    res_a_v1, _ = run_agent_simulation("A_teacher_v1", origins, road_graph, destinations, truth_hazards, h_v1)

    # 4. A2-assim
    print("4/7: Simulating A2-assim...")
    prov_a2_assim = TeacherV2FieldProvider(assimilate=True)
    h_a2_assim = CausalReplayHarness(road_graph=road_graph, provider=prov_a2_assim, latency_min=0.0)
    res_a2_assim, _ = run_agent_simulation("A2-assim", origins, road_graph, destinations, truth_hazards, h_a2_assim)

    # 5. A2-pure
    print("5/7: Simulating A2-pure...")
    prov_a2_pure = TeacherV2FieldProvider(assimilate=False)
    h_a2_pure = CausalReplayHarness(road_graph=road_graph, provider=prov_a2_pure, latency_min=0.0)
    res_a2_pure, _ = run_agent_simulation("A2-pure", origins, road_graph, destinations, truth_hazards, h_a2_pure)

    # 6. B on A2-assim
    print("6/7: Simulating B on A2-assim...")
    prov_b_assim = ObservationInflatedProvider(base_teacher_provider=prov_a2_assim, r_spot_km=4.0)
    h_b_assim = CausalReplayHarness(road_graph=road_graph, provider=prov_b_assim, latency_min=0.0)
    res_b_assim, _ = run_agent_simulation("B_on_A2-assim", origins, road_graph, destinations, truth_hazards, h_b_assim)

    # 7. B on A2-pure
    print("7/7: Simulating B on A2-pure...")
    prov_b_pure = ObservationInflatedProvider(base_teacher_provider=prov_a2_pure, r_spot_km=4.0)
    h_b_pure = CausalReplayHarness(road_graph=road_graph, provider=prov_b_pure, latency_min=0.0)
    res_b_pure, _ = run_agent_simulation("B_on_A2-pure", origins, road_graph, destinations, truth_hazards, h_b_pure)

    agent_dict = {
        "A0 static": (res_a0, False),  # False means predicted margin is N/A
        "A1 reactive": (res_a1, True),
        "A (teacher_v1)": (res_a_v1, True),
        "A2-assim": (res_a2_assim, True),
        "A2-pure": (res_a2_pure, True),
        "B on A2-assim": (res_b_assim, True),
        "B on A2-pure": (res_b_pure, True),
    }

    step2_rows = []

    for ag_name, (res_list, has_pred) in agent_dict.items():
        recs = compute_nist_truth_margins_for_agent(res_list, origins, truth_hazards, road_graph)

        trapped_cnt = sum(1 for r in recs if r["trapped"])
        no_hazard_cnt = sum(1 for r in recs if r["no_hazard"])

        # Origins that have a hazard on route
        with_hazard_recs = [r for r in recs if not r["trapped"] and not r["no_hazard"]]
        finite_margins = [r["margin_min"] for r in with_hazard_recs]
        neg_margin_cnt = sum(1 for m in finite_margins if m < 0.0)

        pos_inf_cnt = sum(1 for r in recs if np.isposinf(r["margin_min"]))
        neg_inf_cnt = sum(1 for r in recs if np.isneginf(r["margin_min"]))

        if finite_margins:
            med_val = float(np.median(finite_margins))
            # Compute P10 on finite values only
            p10_val = float(np.percentile(finite_margins, 10.0))
        else:
            med_val = float("nan")
            p10_val = float("nan")

        # Assert requirement: P10 is -inf only if at least 10 percent of values are -inf
        all_vals = [r["margin_min"] for r in recs]
        frac_neginf = sum(1 for v in all_vals if np.isneginf(v)) / len(all_vals)
        p10_all = float(np.percentile(all_vals, 10.0))
        if np.isneginf(p10_all):
            assert frac_neginf >= 0.10, f"P10 is -inf but frac -inf is only {frac_neginf:.4f} < 0.10!"

        # Predicted margin (i)
        if has_pred:
            pred_finite = [r["pred_margin_min"] for r in recs if np.isfinite(r["pred_margin_min"])]
            pred_med_str = f"{np.median(pred_finite):.1f} min" if pred_finite else ("+inf" if any(np.isposinf(r["pred_margin_min"]) for r in recs) else "-inf")
            pred_p10_str = f"{np.percentile(pred_finite, 10.0):.1f} min" if pred_finite else ("+inf" if any(np.isposinf(r["pred_margin_min"]) for r in recs) else "-inf")
        else:
            pred_med_str = "NOT APPLICABLE"
            pred_p10_str = "NOT APPLICABLE"

        step2_rows.append({
            "agent": ag_name,
            "no_hazard_on_route_count": no_hazard_cnt,
            "origins_with_hazard_count": len(with_hazard_recs),
            "truth_margin_median": f"{med_val:.1f} min" if not np.isnan(med_val) else "N/A",
            "truth_margin_p10_finite": f"{p10_val:.1f} min" if not np.isnan(p10_val) else "N/A",
            "negative_margin_count": f"{neg_margin_cnt}/{len(with_hazard_recs)}",
            "trapped_count": f"{trapped_cnt}/200",
            "pos_inf_count": pos_inf_cnt,
            "neg_inf_count": neg_inf_cnt,
            "pred_margin_median": pred_med_str,
            "pred_margin_p10": pred_p10_str,
        })

    df_step2 = pd.DataFrame(step2_rows)
    Path("reports").mkdir(exist_ok=True)
    out_s2 = Path("reports/step2_truth_margin_vs_nist_hazards.csv")
    df_step2.to_csv(out_s2, index=False)

    print("\n" + "=" * 80)
    print("STEP 2 TABLE: TRUTH MARGIN VS NIST EMPIRICAL HAZARDS (436 HAZARDS)")
    print("=" * 80)
    print(df_step2.to_string(index=False))
    print(f"\nSaved Step 2 table to {out_s2}")

    # =========================================================================
    # STEP 3: TIMING ERROR AT HAZARDS (Seed 42 & 5-Seed Ensemble Median)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: TIMING ERROR AT HAZARDS (Teacher v2 vs NIST Onset)")
    print("=" * 80)

    # 1. Edge arrivals under Seed 42
    print("Computing Teacher v2 fire arrivals for Seed 42...")
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

    # Compute per-edge median arrival across 5 seeds
    stacked_ea = np.stack(ensemble_edge_arrivals, axis=0)  # (5, num_edges)
    ensemble_median_ea = np.median(stacked_ea, axis=0)

    # 3. Associate hazards on routes with corridors
    # Map corridor by destination node:
    # 86429619: Clark Rd, 2611113946: Skyway, 86469871 & 10131737665: Pentz/Hwy 70
    edge_map = {(e.u, e.v, e.key): (i, e) for i, e in enumerate(road_graph.edges)}
    corridor_nodes = {
        86429619: "Clark Rd",
        2611113946: "Skyway",
        86469871: "Pentz/Hwy 70",
        10131737665: "Pentz/Hwy 70",
    }

    # For every route from A0, map hazards encountered to corridor
    corridor_hazard_pairs = {"Clark Rd": [], "Skyway": [], "Pentz/Hwy 70": []}
    seen_pairs = set()

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
                geom = ed.geometry

                for h in truth_hazards:
                    if geom is not None and not geom.is_empty:
                        d = float(geom.distance(h.geometry_utm))
                    else:
                        d = float("inf")

                    if d <= 120.0:
                        pair_key = (c_name, h.hazard_id, edge_idx)
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            corridor_hazard_pairs[c_name].append({
                                "corridor": c_name,
                                "hazard_id": h.hazard_id,
                                "hazard_type": h.hazard_type,
                                "onset_min": h.onset_elapsed_min,
                                "edge_idx": edge_idx,
                                "t_s42": truth_edge_arrivals_s42[edge_idx],
                                "t_ens": ensemble_median_ea[edge_idx],
                            })

    step3_rows = []
    for c_name in ["Clark Rd", "Skyway", "Pentz/Hwy 70"]:
        pairs = corridor_hazard_pairs[c_name]
        n_haz = len(pairs)
        if n_haz == 0:
            continue

        # Under Seed 42:
        diffs_s42 = []
        later_s42 = 0
        never_s42 = 0
        for p in pairs:
            t_pred = p["t_s42"]
            t_truth = p["onset_min"]
            if np.isnan(t_pred) or np.isneginf(t_pred) or np.isposinf(t_pred):
                never_s42 += 1
            else:
                diff = t_pred - t_truth
                diffs_s42.append(diff)
                if diff > 0.0:
                    later_s42 += 1

        med_diff_s42 = float(np.median(diffs_s42)) if diffs_s42 else float("nan")
        share_later_s42 = later_s42 / n_haz
        share_never_s42 = never_s42 / n_haz

        # Under 5-Seed Ensemble Median:
        diffs_ens = []
        later_ens = 0
        never_ens = 0
        for p in pairs:
            t_pred = p["t_ens"]
            t_truth = p["onset_min"]
            if np.isnan(t_pred) or np.isneginf(t_pred) or np.isposinf(t_pred):
                never_ens += 1
            else:
                diff = t_pred - t_truth
                diffs_ens.append(diff)
                if diff > 0.0:
                    later_ens += 1

        med_diff_ens = float(np.median(diffs_ens)) if diffs_ens else float("nan")
        share_later_ens = later_ens / n_haz
        share_never_ens = never_ens / n_haz

        step3_rows.append({
            "corridor": c_name,
            "n_hazard_edge_pairs": n_haz,
            "s42_median_signed_diff_min": round(med_diff_s42, 1) if not np.isnan(med_diff_s42) else "N/A",
            "s42_share_teacher_later": f"{share_later_s42 * 100.0:.1f}%",
            "s42_share_never_burn": f"{share_never_s42 * 100.0:.1f}%",
            "ens_median_signed_diff_min": round(med_diff_ens, 1) if not np.isnan(med_diff_ens) else "N/A",
            "ens_share_teacher_later": f"{share_later_ens * 100.0:.1f}%",
            "ens_share_never_burn": f"{share_never_ens * 100.0:.1f}%",
        })

    df_step3 = pd.DataFrame(step3_rows)
    out_s3 = Path("reports/step3_timing_error_at_hazards.csv")
    df_step3.to_csv(out_s3, index=False)

    print("\n" + "=" * 80)
    print("STEP 3 TABLE: TIMING ERROR AT HAZARDS (TEACHER V2 VS NIST TRUTH)")
    print("=" * 80)
    print(df_step3.to_string(index=False))
    print(f"\nSaved Step 3 table to {out_s3}")


if __name__ == "__main__":
    main()
