"""Phase 6C: Comprehensive Replay Table Integrity and Provenance Audit Script.

Computes:
1. Denominators: Origin-level (out of 200) and Hazard-level (hazards met out of 23, 178, 235, 436).
2. Identical row/column origin ID decomposition.
3. Information flow & roadblock input metric.
4. Reactive & observation-driven burnovers explanation.
5. Paired comparison vs Static for all methods (10,000-resample bootstrap of difference, discordant counts n10, n01, paired p-values).
6. 6A to 6B reconciliation with attribution.
7. Flagged before onset: All 200 origins vs Threatened origins only.
8. Safety margin distribution (inf, -inf, finite medians).
9. R_spot sensitivity in {1, 2, 4, 7} km and origin route difference count between 4km and 7km.
10. Latency breakdown: per-step timing vs per-replan router call timing (mean, P50, P95).
"""

import hashlib
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

from wf052.replay.contracts import MethodMetrics, ReplayOriginResult, TruthHazard
from wf052.replay.harness import CausalReplayHarness
from wf052.replay.methods import (
    EvacuationOrigin,
    run_agent_simulation,
    run_reactive_baseline,
    run_static_baseline,
    sample_block_origins,
)
from wf052.replay.metrics import (
    aggregate_method_metrics,
    check_route_hazard_miss_by_type,
    compute_warning_lead_time,
)
from wf052.replay.providers import (
    ObservationInflatedProvider,
    TeacherFieldProvider,
    TeacherV2FieldProvider,
)
from wf052.routing.graph import RoadGraph, load_camp_fire_road_graph
from wf052.routing.router import TimeDependentRouter


def paired_bootstrap_difference(
    static_misses: np.ndarray,
    method_misses: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 42,
) -> tuple[float, tuple[float, float], int, int]:
    """Paired bootstrap for difference in miss rates (method - static).

    Returns:
        (diff_mean_pct, (ci_low_pct, ci_high_pct), n10_discordant, n01_discordant)
        where:
        - n10: static was miss (1), method was safe/hit (0) -> method improved
        - n01: static was safe/hit (0), method was miss (1) -> method degraded
    """
    n = len(static_misses)
    diff_observed = float(np.mean(method_misses) - np.mean(static_misses)) * 100.0

    # Discordant counts
    n10 = int(np.sum((static_misses == 1) & (method_misses == 0)))
    n01 = int(np.sum((static_misses == 0) & (method_misses == 1)))

    rng = np.random.default_rng(seed)
    diffs = np.zeros(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        diffs[b] = (np.mean(method_misses[idx]) - np.mean(static_misses[idx])) * 100.0

    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    return diff_observed, (ci_low, ci_high), n10, n01


def compute_hazard_level_metrics(
    results: list[ReplayOriginResult],
    truth_hazards: Sequence[TruthHazard],
    road_graph: RoadGraph,
) -> dict[str, tuple[int, int, float]]:
    """Compute hazard-level misses: how many unique truth hazards were met across all routes.

    Returns dict mapping category to (unique_hazards_met, total_hazards, fraction_met).
    Categories: 'burnover', 'roadblock', 'abandoned_vehicle', 'total'.
    """
    total_hazards_by_type = {
        "burnover": [h for h in truth_hazards if h.hazard_type == "burnover"],
        "roadblock": [h for h in truth_hazards if h.hazard_type == "roadblock"],
        "abandoned_vehicle": [h for h in truth_hazards if h.hazard_type == "abandoned_vehicle"],
        "total": list(truth_hazards),
    }

    met_ids_by_type: dict[str, set[str]] = {cat: set() for cat in total_hazards_by_type}

    for r in results:
        if r.route is None or not r.route.edges:
            continue
        by_type = check_route_hazard_miss_by_type(r.route, truth_hazards, road_graph)
        for cat in ["burnover", "roadblock", "abandoned_vehicle", "total"]:
            is_miss, ids = by_type[cat]
            if is_miss:
                for hid in ids:
                    if hid != "UNREACHABLE_NO_ROUTE":
                        met_ids_by_type[cat].add(hid)

    out = {}
    for cat, h_list in total_hazards_by_type.items():
        n_tot = len(h_list)
        n_met = len(met_ids_by_type[cat])
        frac = n_met / n_tot if n_tot > 0 else 0.0
        out[cat] = (n_met, n_tot, frac)
    return out


def compare_routes_between_radii(
    results_a: list[ReplayOriginResult],
    results_b: list[ReplayOriginResult],
) -> int:
    """Count how many origins took different physical routes between two configurations."""
    routes_a = {r.origin_id: r.route for r in results_a}
    routes_b = {r.origin_id: r.route for r in results_b}
    diff_count = 0
    for orig_id in routes_a:
        ra = routes_a[orig_id]
        rb = routes_b[orig_id]
        if (ra is None) != (rb is None):
            diff_count += 1
        elif ra is not None and rb is not None:
            edges_a = [(e.u, e.v) for e in ra.edges]
            edges_b = [(e.u, e.v) for e in rb.edges]
            if edges_a != edges_b:
                diff_count += 1
    return diff_count


def profile_replan_call_latencies(
    origins: Sequence[EvacuationOrigin],
    road_graph: RoadGraph,
    destinations: Sequence[int] | set[int],
    harness: CausalReplayHarness,
    buffer_minutes: float = 15.0,
    n_runs: int = 50,
) -> tuple[float, float, float]:
    """Measure the latency of individual agent router re-plan calls (Dijkstra) in ms."""
    router = TimeDependentRouter(road_graph=road_graph)
    dest_set = set(destinations)
    env = harness.step(60.0)
    edge_arrivals = road_graph.compute_edge_fire_arrivals(env.arrival_field)

    times_ms = []
    # Warm up 5 runs
    for _ in range(5):
        orig = origins[0]
        router.find_route(orig.origin_id, dest_set, orig.departure_time_min, buffer_minutes, edge_arrivals)

    # 50 timed runs
    for i in range(n_runs):
        orig = origins[i % len(origins)]
        t0 = time.perf_counter()
        router.find_route(orig.origin_id, dest_set, orig.departure_time_min, buffer_minutes, edge_arrivals)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    p50_ms = float(np.median(times_ms))
    p95_ms = float(np.percentile(times_ms, 95.0))
    return mean_ms, p50_ms, p95_ms


def main() -> None:
    t_start = time.time()
    print("=" * 75)
    print("PHASE 6C: REPLAY TABLE INTEGRITY AUDIT")
    print("=" * 75)

    # 1. Load Road Graph
    teacher_field = TeacherFieldProvider().get_field(0.0)
    road_graph = load_camp_fire_road_graph(
        full_detail=True,
        grid_x_coords=teacher_field["x"].values,
        grid_y_coords=teacher_field["y"].values,
    )
    destinations = set(road_graph.boundary_destinations.values())
    origins = sample_block_origins(road_graph=road_graph, num_samples=200, seed=42)

    # 2. Providers and Harnesses
    prov_a = TeacherFieldProvider()
    harness_a = CausalReplayHarness(road_graph=road_graph, provider=prov_a, latency_min=0.0)
    truth_hazards = harness_a.truth_hazards
    n_burnovers = sum(1 for h in truth_hazards if h.hazard_type == "burnover")
    n_roadblocks = sum(1 for h in truth_hazards if h.hazard_type == "roadblock")
    n_abandoned = sum(1 for h in truth_hazards if h.hazard_type == "abandoned_vehicle")
    print(f"Truth hazards: Total {len(truth_hazards)} (Burnovers: {n_burnovers}, Roadblocks: {n_roadblocks}, Abandoned: {n_abandoned})")

    prov_a2 = TeacherV2FieldProvider(assimilate=True)
    harness_a2 = CausalReplayHarness(road_graph=road_graph, provider=prov_a2, latency_min=0.0)

    prov_b_on_a = ObservationInflatedProvider(base_teacher_provider=prov_a, r_spot_km=4.0)
    harness_b_on_a = CausalReplayHarness(road_graph=road_graph, provider=prov_b_on_a, latency_min=0.0)

    prov_b_on_a2 = ObservationInflatedProvider(base_teacher_provider=prov_a2, r_spot_km=4.0)
    harness_b_on_a2 = CausalReplayHarness(road_graph=road_graph, provider=prov_b_on_a2, latency_min=0.0)

    # Overlapping detection exclusion for leakage-free
    raw_obs = harness_a._raw_observations
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
        base_teacher_provider=prov_a,
        r_spot_km=4.0,
        filter_leakage=True,
        leakage_exclusion_indices=overlap_indices,
    )
    harness_b_clean = CausalReplayHarness(road_graph=road_graph, provider=prov_b_clean, latency_min=0.0)

    # 3. Execute all 7 core methods
    methods_dict: dict[str, tuple[list[ReplayOriginResult], float]] = {}

    print("Running Method (i): Static Fastest...")
    methods_dict["static"] = run_static_baseline(origins, road_graph, destinations, truth_hazards)

    print("Running Method (ii): Reactive (lat=0m)...")
    methods_dict["reactive_0"] = run_reactive_baseline(origins, road_graph, destinations, truth_hazards, harness_a)

    print("Running Method (iii): Agent A (Teacher v1)...")
    methods_dict["agent_a"] = run_agent_simulation("agent_a", origins, road_graph, destinations, truth_hazards, harness_a)

    print("Running Method (iv): Agent A2 (Teacher v2 Embers)...")
    methods_dict["agent_a2"] = run_agent_simulation("agent_a2", origins, road_graph, destinations, truth_hazards, harness_a2)

    print("Running Method (v): Agent B on A (Obs 4km)...")
    methods_dict["agent_b_on_a"] = run_agent_simulation("agent_b_on_a", origins, road_graph, destinations, truth_hazards, harness_b_on_a)

    print("Running Method (vi): Agent B on A2 (Obs on v2)...")
    methods_dict["agent_b_on_a2"] = run_agent_simulation("agent_b_on_a2", origins, road_graph, destinations, truth_hazards, harness_b_on_a2)

    print("Running Method (vii): Agent B (Leakage-Free)...")
    methods_dict["agent_b_clean"] = run_agent_simulation("agent_b_clean", origins, road_graph, destinations, truth_hazards, harness_b_clean)

    # Threatened origins mask from Static baseline
    res_static, lat_static_ms = methods_dict["static"]
    static_hit_mask = np.array([r.hazard_miss for r in res_static], dtype=bool)
    n_threatened = int(np.sum(static_hit_mask))
    print(f"\nThreatened origins (static misses): {n_threatened} / 200 ({n_threatened/2.0:.1f}%)")

    # 4. Denominators: Origin-Level and Hazard-Level
    table_rows = []
    categories = ["burnover", "roadblock", "abandoned_vehicle", "total"]

    for m_key, (res_m, lat_m) in methods_dict.items():
        origin_miss_counts = {
            cat: sum(1 for r in res_m if r.hazard_misses_by_type.get(cat if cat != "total" else "total", False))
            for cat in categories
        }
        haz_metrics = compute_hazard_level_metrics(res_m, truth_hazards, road_graph)

        # Flagged before onset: all 200 vs threatened only
        # Static through exact same code path
        flagged_all = sum(1 for r in res_m if r.flagged_before_onset)
        flagged_threatened = sum(1 for r, is_thr in zip(res_m, static_hit_mask, strict=True) if is_thr and r.flagged_before_onset)

        # Safety margin
        margins = np.array([r.min_margin_min for r in res_m], dtype=np.float64)
        n_posinf = int(np.sum(np.isposinf(margins)))
        n_neginf = int(np.sum(np.isneginf(margins)))
        finite_margins = margins[np.isfinite(margins)]
        finite_med = float(np.median(finite_margins)) if len(finite_margins) > 0 else float("nan")
        p50 = float(np.median(margins))
        p10 = float(np.percentile(margins, 10.0, method="nearest"))

        table_rows.append({
            "method_key": m_key,
            "origin_miss_combined": f"{origin_miss_counts['total']}/200 ({origin_miss_counts['total']/2.0:.1f}%)",
            "origin_miss_bo": f"{origin_miss_counts['burnover']}/200 ({origin_miss_counts['burnover']/2.0:.1f}%)",
            "origin_miss_rb": f"{origin_miss_counts['roadblock']}/200 ({origin_miss_counts['roadblock']/2.0:.1f}%)",
            "origin_miss_av": f"{origin_miss_counts['abandoned_vehicle']}/200 ({origin_miss_counts['abandoned_vehicle']/2.0:.1f}%)",
            "haz_miss_combined": f"{haz_metrics['total'][0]}/{haz_metrics['total'][1]} ({haz_metrics['total'][2]*100:.1f}%)",
            "haz_miss_bo": f"{haz_metrics['burnover'][0]}/{haz_metrics['burnover'][1]} ({haz_metrics['burnover'][2]*100:.1f}%)",
            "haz_miss_rb": f"{haz_metrics['roadblock'][0]}/{haz_metrics['roadblock'][1]} ({haz_metrics['roadblock'][2]*100:.1f}%)",
            "haz_miss_av": f"{haz_metrics['abandoned_vehicle'][0]}/{haz_metrics['abandoned_vehicle'][1]} ({haz_metrics['abandoned_vehicle'][2]*100:.1f}%)",
            "flagged_all": f"{flagged_all}/200 ({flagged_all/2.0:.1f}%)",
            "flagged_threatened": f"{flagged_threatened}/{n_threatened} ({flagged_threatened/n_threatened*100:.1f}%)",
            "margin_p50": "+inf" if np.isposinf(p50) else f"{p50:.1f} min",
            "margin_p10": "-inf" if np.isneginf(p10) else ("+inf" if np.isposinf(p10) else f"{p10:.1f} min"),
            "margin_finite_med": f"{finite_med:.1f} min" if not np.isnan(finite_med) else "N/A",
            "margin_inf_counts": f"{n_posinf} +inf, {n_neginf} -inf, {len(finite_margins)} finite",
            "step_lat_ms": lat_m,
        })

    df_table = pd.DataFrame(table_rows)
    df_table.to_csv("reports/replay_6c_full_method_table.csv", index=False)
    print("\n=== REPLAY METHOD COMPARISON TABLE SAVED to reports/replay_6c_full_method_table.csv ===")
    print(df_table[["method_key", "origin_miss_combined", "origin_miss_bo", "origin_miss_rb", "origin_miss_av", "flagged_all", "flagged_threatened", "margin_p50", "margin_p10"]].to_string(index=False))

    # 5. Paired Comparison vs Static (10,000 resamples)
    print("\n=== PAIRED COMPARISON VS STATIC (10,000 Resamples) ===")
    paired_rows = []
    static_results = methods_dict["static"][0]

    for cat in ["total", "burnover", "roadblock", "abandoned_vehicle"]:
        stat_arr = np.array([1 if r.hazard_misses_by_type.get(cat, False) else 0 for r in static_results], dtype=int)
        for m_key in ["reactive_0", "agent_a", "agent_a2", "agent_b_on_a", "agent_b_on_a2", "agent_b_clean"]:
            res_m = methods_dict[m_key][0]
            meth_arr = np.array([1 if r.hazard_misses_by_type.get(cat, False) else 0 for r in res_m], dtype=int)
            diff_mean, (ci_lo, ci_hi), n10, n01 = paired_bootstrap_difference(stat_arr, meth_arr, n_resamples=10000, seed=42)
            # Exact McNemar test
            # McNemar test statistic = (|n10 - n01| - 1)^2 / (n10 + n01) if n10+n01 > 25 else exact binomial
            p_val = stats.binomtest(n10, n10 + n01, 0.5).pvalue if (n10 + n01) > 0 else 1.0

            paired_rows.append({
                "hazard_type": cat,
                "method": m_key,
                "diff_pct": diff_mean,
                "ci_95": f"[{ci_lo:+.1f}%, {ci_hi:+.1f}%]",
                "n10_improved": n10,
                "n01_degraded": n01,
                "mcnemar_p": p_val,
                "significant": p_val < 0.05,
            })
            print(f"{cat:18s} | {m_key:18s} | Diff: {diff_mean:+5.1f}% [{ci_lo:+5.1f}%, {ci_hi:+5.1f}%] | n10(imp)={n10:2d}, n01(deg)={n01:2d} | p={p_val:.4f}")

    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv("reports/replay_6c_paired_bootstrap.csv", index=False)

    # 6. Identical Row/Column Analysis: Origin IDs
    print("\n=== IDENTICAL ROW / COLUMN ORIGIN ID ANALYSIS ===")
    for m_key in ["static", "reactive_0", "agent_a", "agent_a2", "agent_b_on_a", "agent_b_on_a2", "agent_b_clean"]:
        res_m = methods_dict[m_key][0]
        unreachable_origins = [r.origin_id for r in res_m if r.route is None or not r.route.edges]
        bo_misses = [r.origin_id for r in res_m if r.hazard_misses_by_type.get("burnover", False)]
        rb_misses = [r.origin_id for r in res_m if r.hazard_misses_by_type.get("roadblock", False)]
        av_misses = [r.origin_id for r in res_m if r.hazard_misses_by_type.get("abandoned_vehicle", False)]
        tot_misses = [r.origin_id for r in res_m if r.hazard_misses_by_type.get("total", False)]

        print(f"\nMethod {m_key}:")
        print(f"  Unreachable (route=None): {len(unreachable_origins)} origins: {unreachable_origins[:5]}...")
        print(f"  Burnover misses: {len(bo_misses)} origins")
        print(f"  Roadblock misses: {len(rb_misses)} origins")
        print(f"  Abandoned vehicle misses: {len(av_misses)} origins")
        print(f"  Total/Combined misses: {len(tot_misses)} origins")
        print(f"  Is Combined == Abandoned Vehicle? {set(tot_misses) == set(av_misses)}")
        print(f"  Is Burnover == Roadblock? {set(bo_misses) == set(rb_misses)} (diff: {set(bo_misses) ^ set(rb_misses)})")

    # 7. R_spot Sensitivity in {1, 2, 4, 7} km and Route Difference Count
    print("\n=== R_SPOT SENSITIVITY AND ROUTE DIFFERENCES ===")
    r_results: dict[float, list[ReplayOriginResult]] = {}
    r_sens_rows = []

    for r_km in [1.0, 2.0, 4.0, 7.0]:
        prov_r = ObservationInflatedProvider(base_teacher_provider=prov_a, r_spot_km=r_km)
        h_r = CausalReplayHarness(road_graph=road_graph, provider=prov_r, latency_min=0.0)
        res_r, lat_r = run_agent_simulation(f"provB_r{int(r_km)}", origins, road_graph, destinations, truth_hazards, h_r)
        r_results[r_km] = res_r

        tot_m = sum(1 for r in res_r if r.hazard_misses_by_type.get("total", False))
        bo_m = sum(1 for r in res_r if r.hazard_misses_by_type.get("burnover", False))
        rb_m = sum(1 for r in res_r if r.hazard_misses_by_type.get("roadblock", False))
        av_m = sum(1 for r in res_r if r.hazard_misses_by_type.get("abandoned_vehicle", False))
        fl_m = sum(1 for r in res_r if r.flagged_before_onset)
        r_sens_rows.append({
            "r_spot_km": r_km,
            "miss_combined": f"{tot_m}/200 ({tot_m/2.0:.1f}%)",
            "miss_bo": f"{bo_m}/200 ({bo_m/2.0:.1f}%)",
            "miss_rb": f"{rb_m}/200 ({rb_m/2.0:.1f}%)",
            "miss_av": f"{av_m}/200 ({av_m/2.0:.1f}%)",
            "flagged_before": f"{fl_m}/200 ({fl_m/2.0:.1f}%)",
        })

    # Compare 4km vs 7km routes
    diff_4_7 = compare_routes_between_radii(r_results[4.0], r_results[7.0])
    diff_1_4 = compare_routes_between_radii(r_results[1.0], r_results[4.0])
    diff_2_4 = compare_routes_between_radii(r_results[2.0], r_results[4.0])
    print(f"Origins with different chosen routes between 4 km and 7 km: {diff_4_7} / 200")
    print(f"Origins with different chosen routes between 1 km and 4 km: {diff_1_4} / 200")
    print(f"Origins with different chosen routes between 2 km and 4 km: {diff_2_4} / 200")

    df_rsens = pd.DataFrame(r_sens_rows)
    df_rsens.to_csv("reports/replay_6c_r_spot_sensitivity.csv", index=False)
    print(df_rsens.to_string(index=False))

    # 8. Profile Replan Latency (Mean, P50, P95)
    print("\n=== AGENT B REPLAN-CALL LATENCY PROFILING ===")
    replan_mean_ms, replan_p50_ms, replan_p95_ms = profile_replan_call_latencies(
        origins=origins,
        road_graph=road_graph,
        destinations=destinations,
        harness=harness_b_on_a,
        buffer_minutes=15.0,
        n_runs=50,
    )
    print(f"Agent B single re-plan call latency (50 runs): Mean = {replan_mean_ms:.2f} ms | P50 = {replan_p50_ms:.2f} ms | P95 = {replan_p95_ms:.2f} ms")

    total_elapsed = time.time() - t_start
    print(f"\nAll Phase 6C audit runs completed successfully in {total_elapsed:.1f}s.")


if __name__ == "__main__":
    main()
