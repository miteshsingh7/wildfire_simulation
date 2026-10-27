#!/usr/bin/env python3
"""Phase 4B Step 2: Route Difference & Margin Analysis (A0 vs A1 vs A2-assim vs A2-pure).

Evaluates:
- 2a: Rename A2 into A2-assim (causal assimilation) and A2-pure (unassimilated physics).
- 2b: Route difference table for A0 vs A1 vs A2-assim vs A2-pure:
      Route distance (m), edge count, overlap with A0 (Jaccard on edge sets),
      min margin (predicted arrival - pass time), and truth-observed margin (truth arrival - pass time).
      Where margins are negative: reports which edges and what time the vehicle was overrun.
- 2c: Margin bug fix: Predicted margin uses agent's fire model, truth margin uses Teacher ground truth.
- 2e: A3-reignite variant evaluation.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from wf052.replay.contracts import ReplayOriginResult
from wf052.replay.harness import CausalReplayHarness
from wf052.replay.methods import (
    run_agent_simulation,
    run_reactive_baseline,
    run_static_baseline,
    sample_block_origins,
)
from wf052.replay.providers import (
    SurrogateFieldProvider,
    TeacherFieldProvider,
    TeacherV2FieldProvider,
)
from wf052.routing.graph import load_camp_fire_road_graph


def compute_route_edge_set(route) -> set[tuple[int, int]]:
    if route is None or not route.edges:
        return set()
    return {(e.u, e.v) for e in route.edges}


def jaccard_similarity(s1: set[tuple[int, int]], s2: set[tuple[int, int]]) -> float:
    if not s1 and not s2:
        return 1.0
    union = s1 | s2
    if not union:
        return 1.0
    return len(s1 & s2) / float(len(union))


def main() -> None:
    print("=" * 80)
    print("PHASE 4B STEP 2: ROUTE DIFFERENCE & HONEST MARGIN ANALYSIS")
    print("=" * 80)

    # 1. Load Road Graph and Ground Truth Teacher Arrival Field
    truth_provider = TeacherV2FieldProvider(assimilate=False)
    truth_field = truth_provider.get_field(0.0)
    gx = truth_field["x"].values
    gy = truth_field["y"].values

    road_graph = load_camp_fire_road_graph(
        full_detail=True,
        grid_x_coords=gx,
        grid_y_coords=gy,
    )
    destinations = set(road_graph.boundary_destinations.values())
    origins = sample_block_origins(road_graph=road_graph, num_samples=200, seed=42)

    # Precompute ground truth fire arrival for every edge in the road graph using Teacher v2
    print("Computing ground-truth Teacher v2 fire arrival times across all road graph edges...")
    truth_edge_arrivals = road_graph.compute_edge_fire_arrivals(truth_field)
    finite_truth_edges = int(np.sum(np.isfinite(truth_edge_arrivals)))
    print(f"Teacher v2 fire reaches {finite_truth_edges} / {len(road_graph.edges)} road graph edges.")

    # Edge map from (u, v, key) to index in road_graph.edges
    edge_to_idx = {(e.u, e.v, e.key): i for i, e in enumerate(road_graph.edges)}

    # Base harness for hazards
    harness_base = CausalReplayHarness(road_graph=road_graph, provider=truth_provider, latency_min=0.0)
    truth_hazards = harness_base.truth_hazards

    print(f"Loaded road graph: {len(road_graph.nodes)} nodes, {len(road_graph.edges)} edges")
    print(f"Loaded {len(origins)} evacuation origins, {len(truth_hazards)} empirical truth hazards")

    # 2. Run the 4 agents
    print("\n--- Running Evacuation Simulations ---")
    print("1. Simulating A0 (Static fastest path)...")
    res_a0, lat_a0 = run_static_baseline(origins, road_graph, destinations, truth_hazards)

    print("2. Simulating A1 (Reactive routing around active roadblocks & observations)...")
    res_a1, lat_a1 = run_reactive_baseline(origins, road_graph, destinations, truth_hazards, harness_base)

    print("3. Simulating A2-assim (Teacher v2 with causal assimilation)...")
    prov_a2_assim = TeacherV2FieldProvider(assimilate=True)
    h_a2_assim = CausalReplayHarness(road_graph=road_graph, provider=prov_a2_assim, latency_min=0.0)
    res_a2_assim, lat_a2_assim = run_agent_simulation("A2-assim", origins, road_graph, destinations, truth_hazards, h_a2_assim)

    print("4. Simulating A2-pure (Teacher v2 pure forward physics, unassimilated)...")
    prov_a2_pure = TeacherV2FieldProvider(assimilate=False)
    h_a2_pure = CausalReplayHarness(road_graph=road_graph, provider=prov_a2_pure, latency_min=0.0)
    res_a2_pure, lat_a2_pure = run_agent_simulation("A2-pure", origins, road_graph, destinations, truth_hazards, h_a2_pure)

    agents = {
        "A0 (Static)": res_a0,
        "A1 (Reactive)": res_a1,
        "A2-assim": res_a2_assim,
        "A2-pure": res_a2_pure,
    }

    # 3. Analyze Routes, Overlaps, Predicted Margins, and Ground-Truth Margins
    a0_edge_sets = [compute_route_edge_set(r.route) for r in res_a0]

    overrun_events = []
    agent_summary_rows = []

    for agent_name, results in agents.items():
        distances_m = []
        edge_counts = []
        jaccard_scores = []
        pred_margins = []
        truth_margins = []

        for i, res in enumerate(results):
            orig = origins[i]
            r = res.route
            if r is not None and r.edges:
                dist_m = sum(e.length_m for e in r.edges)
                cnt = len(r.edges)
                e_set = compute_route_edge_set(r)
                jacc = jaccard_similarity(e_set, a0_edge_sets[i])

                # Predicted Margin under Agent's fire model:
                # min_{e} (t_fire_pred(e) - t_vehicle_exit(e))
                # Note: For A0 and A1, agent model fire arrival is inf unless hazard closed
                p_margins = [e.fire_arrival_min - e.exit_time_min for e in r.edges]
                min_pred = min(p_margins) if p_margins else float("inf")

                # Ground Truth Margin under Teacher v2 ground truth arrival:
                # min_{e} (t_fire_truth(e) - t_vehicle_exit(e))
                t_margins = []
                for e in r.edges:
                    idx = edge_to_idx.get((e.u, e.v, e.key))
                    if idx is not None:
                        t_truth = truth_edge_arrivals[idx]
                    else:
                        t_truth = float("inf")
                    e_margin = t_truth - e.exit_time_min
                    t_margins.append(e_margin)

                    # Check for overrun (fire arrives before vehicle clears the edge)
                    if e_margin < 0.0:
                        overrun_events.append({
                            "agent": agent_name,
                            "origin_id": orig.origin_id,
                            "assigned_zone": orig.assigned_zone,
                            "departure_time_min": orig.departure_time_min,
                            "edge_u": e.u,
                            "edge_v": e.v,
                            "edge_key": e.key,
                            "length_m": round(e.length_m, 1),
                            "vehicle_entry_min": round(e.entry_time_min, 2),
                            "vehicle_exit_min": round(e.exit_time_min, 2),
                            "truth_fire_arrival_min": round(t_truth, 2),
                            "overrun_time_min": round(t_truth, 2),
                            "negative_margin_min": round(e_margin, 2),
                        })

                min_truth = min(t_margins) if t_margins else float("inf")

                distances_m.append(dist_m)
                edge_counts.append(cnt)
                jaccard_scores.append(jacc)
                pred_margins.append(min_pred)
                truth_margins.append(min_truth)
            else:
                # Trapped origin
                distances_m.append(float("nan"))
                edge_counts.append(0)
                jaccard_scores.append(0.0)
                pred_margins.append(-float("inf"))
                truth_margins.append(-float("inf"))

        # Compute aggregate statistics
        valid_dist = [d for d in distances_m if not np.isnan(d)]
        mean_dist = float(np.mean(valid_dist)) if valid_dist else float("nan")
        median_dist = float(np.median(valid_dist)) if valid_dist else float("nan")

        mean_edges = float(np.mean(edge_counts))
        median_edges = float(np.median(edge_counts))

        mean_jaccard = float(np.mean(jaccard_scores))

        pred_med = float(np.median(pred_margins))
        pred_p10 = float(np.percentile(pred_margins, 10.0))
        pred_neg_cnt = sum(1 for m in pred_margins if m < 0.0)

        truth_med = float(np.median(truth_margins))
        truth_p10 = float(np.percentile(truth_margins, 10.0))
        truth_neg_cnt = sum(1 for m in truth_margins if m < 0.0)

        agent_summary_rows.append({
            "agent": agent_name,
            "mean_distance_m": round(mean_dist, 1),
            "median_distance_m": round(median_dist, 1),
            "mean_edge_count": round(mean_edges, 1),
            "median_edge_count": round(median_edges, 1),
            "jaccard_overlap_with_a0": round(mean_jaccard, 4),
            "pred_margin_median": f"{pred_med:.1f} min" if np.isfinite(pred_med) else ("+inf" if pred_med > 0 else "-inf"),
            "pred_margin_p10": f"{pred_p10:.1f} min" if np.isfinite(pred_p10) else ("+inf" if pred_p10 > 0 else "-inf"),
            "pred_negative_count": f"{pred_neg_cnt}/200 ({pred_neg_cnt/2.0:.1f}%)",
            "truth_margin_median": f"{truth_med:.1f} min" if np.isfinite(truth_med) else ("+inf" if truth_med > 0 else "-inf"),
            "truth_margin_p10": f"{truth_p10:.1f} min" if np.isfinite(truth_p10) else ("+inf" if truth_p10 > 0 else "-inf"),
            "truth_negative_count": f"{truth_neg_cnt}/200 ({truth_neg_cnt/2.0:.1f}%)",
        })

    df_summary = pd.DataFrame(agent_summary_rows)
    df_overrun = pd.DataFrame(overrun_events)

    Path("reports").mkdir(exist_ok=True)
    out_summary_csv = Path("reports/step2b_route_difference_a0_a1_a2.csv")
    out_overrun_csv = Path("reports/step2b_overrun_details.csv")

    df_summary.to_csv(out_summary_csv, index=False)
    df_overrun.to_csv(out_overrun_csv, index=False)

    print("\n" + "=" * 80)
    print("STEP 2b: ROUTE DIFFERENCE SUMMARY TABLE (A0 vs A1 vs A2-assim vs A2-pure)")
    print("=" * 80)
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"STEP 2b: VEHICLE OVERRUN DETAILS ({len(df_overrun)} edge overrun events)")
    print("=" * 80)
    if len(df_overrun) > 0:
        # Group by agent
        for ag, grp in df_overrun.groupby("agent"):
            print(f"\n--- Overruns for {ag} (Total {len(grp)} edge overruns across {grp['origin_id'].nunique()} origins) ---")
            cols_show = ["origin_id", "assigned_zone", "departure_time_min", "edge_u", "edge_v", "vehicle_entry_min", "vehicle_exit_min", "truth_fire_arrival_min", "negative_margin_min"]
            print(grp[cols_show].to_string(index=False))
    else:
        print("No vehicle overruns observed.")

    print(f"\nSaved summary table to {out_summary_csv}")
    print(f"Saved overrun details to {out_overrun_csv}")


if __name__ == "__main__":
    main()
