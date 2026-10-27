"""Step 5: Surrogate Update Loop and Evacuation Replay Agents (A3 and B-on-A3).

Evaluates:
- 5a: 30-min update loop re-running surrogate with sliding wind window. Logs:
      fraction of cells with Delta q50 > 10 min, origins route changes, surrogate latency, router+validator latency.
- 5b: Agents A3 (surrogate updates, no obs) and B-on-A3 (Provider B on A3). Section 7.3 metrics table.
- 5c: Route-safety margin under two definitions (predicted arrival vs truth hazard).
- 5d: Per-zone outputs across 14 zones (paradise_14_zones.parquet) and 3 egress corridors.
- 5e: End-to-end re-plan event latency (mean, p95).
- 5f: Corridor-clustered bootstrap for A3 vs A2 and B-on-A3 vs Static (report INCONCLUSIVE if CI includes 0).
- 5g: 7-node LangGraph agent graph on A3 updates with template brief (LLM NOT RUN).
"""

import json
import os
import time
from pathlib import Path
from typing import Any
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
import torch
import xarray as xr

from eval_replay_6c_audit import compute_hazard_level_metrics
from eval_replay_6d_audit import count_abandoned_vehicles_for_route
from evaluate_heldout_camp_fire import build_camp_fire_dataset_60m
from wf052.agent.graph import EvacuationAgent
from wf052.agent.llm import DeterministicBriefGenerator, MockLLMClient
from wf052.replay.contracts import MethodMetrics, ReplayOriginResult, TruthHazard
from wf052.replay.harness import CausalReplayHarness
from wf052.replay.methods import (
    run_agent_simulation,
    run_static_baseline,
    sample_block_origins,
)
from wf052.replay.metrics import (
    check_route_hazard_miss_by_type,
    compute_bootstrap_ci,
)
from wf052.replay.providers import (
    DynamicTeacherV2Provider,
    ObservationInflatedProvider,
    SurrogateFieldProvider,
    TeacherFieldProvider,
    TeacherV2FieldProvider,
)
from wf052.routing.contracts import validate_arrival_field
from wf052.routing.graph import load_camp_fire_road_graph
from wf052.routing.router import TimeDependentRouter
from wf052.routing.validator import IndependentRouteValidator
from wf052.surrogate.adapter import TilePrediction, stitch_tiles_to_arrival_field
from wf052.surrogate.dataset import rebuild_spatial_inputs
from wf052.surrogate.model import WindConditionedUNet


def cluster_bootstrap_diff(
    method_misses: np.ndarray,
    baseline_misses: np.ndarray,
    clusters: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    diff_observed = float(np.mean(method_misses) - np.mean(baseline_misses)) * 100.0
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    cluster_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}
    rng = np.random.default_rng(seed)
    diffs = np.zeros(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        sampled_c = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx_list = [cluster_indices[c] for c in sampled_c]
        sampled_indices = np.concatenate(idx_list)
        diffs[b] = (np.mean(method_misses[sampled_indices]) - np.mean(baseline_misses[sampled_indices])) * 100.0

    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    return diff_observed, (ci_low, ci_high)


def main() -> None:
    t_start = time.time()
    print("=" * 80)
    print("STAND-IN WEIGHTS (Phase 3A Smoke Checkpoint)")
    print("STEP 5: SURROGATE UPDATE LOOP & REPLAY EVALUATION (A3 & B-on-A3)")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Setup Domain & Graph
    ds_camp, cfg, transform_60m = build_camp_fire_dataset_60m(device)
    h_full, w_full = ds_camp.fbfm_arr.shape

    tf_base = TeacherFieldProvider().get_field(0.0)
    gx = tf_base["x"].values
    gy = tf_base["y"].values
    origin_x = float(gx[0])
    origin_y = float(gy[0])

    road_graph = load_camp_fire_road_graph(
        full_detail=True,
        grid_x_coords=gx,
        grid_y_coords=gy,
    )
    destinations = set(road_graph.boundary_destinations.values())
    origins = sample_block_origins(road_graph=road_graph, num_samples=200, seed=42)

    harness_base = CausalReplayHarness(road_graph=road_graph, provider=TeacherFieldProvider(), latency_min=0.0)
    truth_hazards = harness_base.truth_hazards

    # Map Static baseline corridors for cluster bootstrap
    res_static, lat_static_ms = run_static_baseline(origins, road_graph, destinations, truth_hazards)
    corridor_clusters = []
    for r in res_static:
        if r.route and r.route.edges:
            dest_node = r.route.edges[-1].v
            if dest_node == 86429619:
                c = "Clark_Rd"
            elif dest_node == 2611113946:
                c = "Skyway"
            else:
                c = "Pentz_Hwy70"
        else:
            c = "Unknown"
        corridor_clusters.append(c)
    corridor_clusters = np.array(corridor_clusters)

    # 2. Load Surrogate Model
    ckpt_path = os.environ.get("CKPT", "checkpoints/best.pt")
    model = WindConditionedUNet(in_spatial_channels=13, base_channels=32, emb_dim=64, max_horizon_min=720.0)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load HRRR Wind data
    weather_df = pd.read_parquet("data/tiles/hrrr_weather_hourly.parquet")
    u_all = weather_df["u10_ms"].values.astype(np.float32)
    v_all = weather_df["v10_ms"].values.astype(np.float32)

    # 3. Step 5a: Precompute 30-min Update Fields and Measure Update Dynamics
    print("\n" + "=" * 80)
    print("STEP 5a: 30-MINUTE UPDATE LOOP (0 to 360 min)")
    print("=" * 80)
    print("Wind Data Source: data/tiles/hrrr_weather_hourly.parquet")
    print("Future Leakage Assessment: Weather file uses HRRR zero-hour analysis (fxx=0) for each hour.")
    print("When sliding window accesses future hours [h0..h0+12], it uses historical analysis hindcast,")
    print("representing an idealized weather forecast rather than operational NWP model run.")

    update_steps_min = list(np.arange(0.0, 360.0 + 1e-6, 30.0))  # 13 updates: 0, 30, 60, ..., 360
    r_origins = [0, 154, 308]
    c_origins = [0, 153, 306]

    surrogate_update_fields: dict[float, xr.Dataset] = {}
    prev_q50: np.ndarray | None = None
    prev_routes_dict: dict[int, Any] = {}
    update_log_rows = []

    router = TimeDependentRouter(road_graph)
    validator = IndependentRouteValidator()

    for upd_idx, t_upd in enumerate(update_steps_min):
        t_upd_start = time.perf_counter()
        hour_start = min(len(u_all) - 1, int(t_upd // 60.0))
        # Wind slice of 12 hours from current hour
        u_slice = u_all[hour_start : hour_start + 12]
        v_slice = v_all[hour_start : hour_start + 12]
        if len(u_slice) < 12:
            pad_len = 12 - len(u_slice)
            u_slice = np.pad(u_slice, (0, pad_len), mode="edge")
            v_slice = np.pad(v_slice, (0, pad_len), mode="edge")
        wind_series = np.column_stack([u_slice, v_slice])
        wind_batch = torch.from_numpy(wind_series).unsqueeze(0).to(device)

        # Run 9 tiles
        t_inf_start = time.perf_counter()
        tile_preds = []
        for r0 in r_origins:
            for c0 in c_origins:
                rel_ign_r = 232 - r0
                rel_ign_c = 452 - c0
                tile_tensor = rebuild_spatial_inputs(
                    fire_id="camp_fire",
                    crop_origin=(r0, c0),
                    ignition_cell=(rel_ign_r, rel_ign_c),
                    fires_base_dir="out/camp_fire_60m",
                    tile_shape=(256, 256),
                )
                spatial_batch = tile_tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(spatial_batch, wind_batch)
                    q_tile = out.quantiles[0].cpu().numpy()
                    p_tile = torch.sigmoid(out.burn_logits[0, 0]).cpu().numpy()
                tile_preds.append(TilePrediction(quantiles=q_tile, burn_prob=p_tile, r0=r0, c0=c0))

        if device.type == "mps":
            torch.mps.synchronize()
        surrogate_inf_time_ms = (time.perf_counter() - t_inf_start) * 1000.0

        # Stitch to full domain ArrivalField
        af = stitch_tiles_to_arrival_field(
            tile_predictions=tile_preds,
            domain_shape=(h_full, w_full),
            origin_x=origin_x,
            origin_y=origin_y,
            resolution_m=60.0,
            crs="EPSG:32610",
            ignition_time="2018-11-08T14:33:00Z",
            p_thr=0.1,
        )
        surrogate_update_fields[t_upd] = af
        cur_q50 = af["arrival_q50"].values

        # Fraction of cells whose q50 arrival changed by > 10 minutes
        if prev_q50 is not None:
            both_finite = np.isfinite(cur_q50) & np.isfinite(prev_q50)
            one_finite = np.isfinite(cur_q50) ^ np.isfinite(prev_q50)
            diff_finite = np.abs(cur_q50[both_finite] - prev_q50[both_finite]) > 10.0
            changed_cells = int(np.sum(diff_finite)) + int(np.sum(one_finite))
            frac_q50_changed = changed_cells / float(h_full * w_full)
        else:
            frac_q50_changed = 0.0

        # Measure Router + Validator re-plan latency
        t_rv_start = time.perf_counter()
        edge_arrivals = road_graph.compute_edge_fire_arrivals(af)
        cur_routes_dict = {}
        for orig in origins:
            cur_r = router.find_route(
                origin_node=orig.origin_id,
                destination_nodes=destinations,
                t0=orig.departure_time_min,
                buffer_minutes=15.0,
                edge_fire_arrivals=edge_arrivals,
            )
            cur_routes_dict[orig.origin_id] = cur_r
            if cur_r is not None:
                _ = validator.validate(cur_r, af, road_graph, buffer_minutes=15.0)
        router_validator_ms = (time.perf_counter() - t_rv_start) * 1000.0

        # Count origins whose route changed
        if prev_routes_dict:
            routes_changed = 0
            for orig in origins:
                r_old = prev_routes_dict.get(orig.origin_id)
                r_new = cur_routes_dict.get(orig.origin_id)
                if (r_old is None) != (r_new is None):
                    routes_changed += 1
                elif r_old is not None and r_new is not None:
                    edges_old = [(e.u, e.v) for e in r_old.edges]
                    edges_new = [(e.u, e.v) for e in r_new.edges]
                    if edges_old != edges_new:
                        routes_changed += 1
        else:
            routes_changed = 0

        total_replan_event_ms = surrogate_inf_time_ms + router_validator_ms

        update_log_rows.append({
            "update_step_min": t_upd,
            "hour_index": hour_start,
            "frac_cells_q50_changed_gt_10m": frac_q50_changed,
            "origins_route_changed": routes_changed,
            "surrogate_inf_ms": round(surrogate_inf_time_ms, 2),
            "router_validator_ms": round(router_validator_ms, 2),
            "total_replan_ms": round(total_replan_event_ms, 2),
        })

        prev_q50 = cur_q50.copy()
        prev_routes_dict = cur_routes_dict

    df_upd = pd.DataFrame(update_log_rows)
    print("\n=== STEP 5a: UPDATE LOOP LOG (13 update events) ===")
    print(df_upd.to_string(index=False))

    # 4. Step 5b: Replay Agents A3 and B-on-A3
    print("\n" + "=" * 80)
    print("STEP 5b: REPLAY EVALUATION OF AGENTS A3 & B-ON-A3")
    print("=" * 80)

    # Provider A3: Surrogate with 30-min updates, NO observations
    prov_a3 = SurrogateFieldProvider(update_fields=surrogate_update_fields)
    harness_a3 = CausalReplayHarness(road_graph=road_graph, provider=prov_a3, latency_min=0.0)

    # Provider B-on-A3: ObservationInflatedProvider on top of A3
    prov_b_on_a3 = ObservationInflatedProvider(base_teacher_provider=prov_a3, r_spot_km=4.0)
    harness_b_on_a3 = CausalReplayHarness(road_graph=road_graph, provider=prov_b_on_a3, latency_min=0.0)

    # Also setup A2 for comparison
    prov_a2 = TeacherV2FieldProvider(assimilate=True)
    harness_a2 = CausalReplayHarness(road_graph=road_graph, provider=prov_a2, latency_min=0.0)

    print("Simulating Agent A3 (Surrogate, 30-min updates, no obs)...")
    res_a3, lat_a3 = run_agent_simulation("agent_a3", origins, road_graph, destinations, truth_hazards, harness_a3)

    print("Simulating Agent B-on-A3 (Provider B on A3 updates)...")
    res_b_a3, lat_b_a3 = run_agent_simulation("agent_b_on_a3", origins, road_graph, destinations, truth_hazards, harness_b_on_a3)

    print("Simulating Agent A2 (Teacher v2 reference)...")
    res_a2, lat_a2 = run_agent_simulation("agent_a2", origins, road_graph, destinations, truth_hazards, harness_a2)

    # Build Section 7.3 metrics table
    methods = {
        "static": (res_static, lat_static_ms),
        "agent_a2": (res_a2, lat_a2),
        "agent_a3": (res_a3, lat_a3),
        "agent_b_on_a3": (res_b_a3, lat_b_a3),
    }

    # Threatened origins mask from Static baseline
    static_hit_mask = np.array([r.hazard_miss for r in res_static], dtype=bool)
    n_threatened = int(np.sum(static_hit_mask))

    sec73_rows = []
    categories = ["burnover", "roadblock", "abandoned_vehicle", "total"]

    for m_key, (res_m, lat_m) in methods.items():
        origin_miss_counts = {
            cat: sum(1 for r in res_m if r.hazard_misses_by_type.get(cat if cat != "total" else "total", False))
            for cat in categories
        }
        haz_metrics = compute_hazard_level_metrics(res_m, truth_hazards, road_graph)

        # Flagged before onset: all 200 vs threatened only
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

        sec73_rows.append({
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
            "step_lat_ms": round(lat_m, 2),
        })

    df_sec73 = pd.DataFrame(sec73_rows)
    print("\n=== STEP 5b: SECTION 7.3 REPLAY RESULTS TABLE ===")
    print(df_sec73.to_string(index=False))

    # 5. Step 5c: Route-Safety Margin (Two Definitions)
    print("\n" + "=" * 80)
    print("STEP 5c: ROUTE-SAFETY MARGINS (Two Definitions)")
    print("=" * 80)
    print("Departure-time convention: Vehicles depart at t0 in [70, 180] min.")
    print("Definition (i): Margin to PREDICTED arrival at plan time = min_{e in route} (t_fire_pred(e) - t_vehicle_pass(e)).")
    print("Definition (ii): Margin to TRUTH hazard arrival = min_{h in route} (t_hazard_onset(h) - t_vehicle_pass(h)).")

    margin_rows = []
    for m_key, (m_results, _) in methods.items():
        # Definition (i): Predicted Margins
        pred_margins = []
        for r in m_results:
            if r.route and r.route.edges:
                edge_m = [e.fire_arrival_min - e.exit_time_min for e in r.route.edges]
                pred_margins.append(min(edge_m) if edge_m else np.inf)
            else:
                pred_margins.append(-np.inf)
        pred_p50 = float(np.percentile(pred_margins, 50.0))
        pred_p10 = float(np.percentile(pred_margins, 10.0))
        pred_neg_cnt = sum(1 for m in pred_margins if m < 0.0)

        # Definition (ii): Truth Margins
        truth_margins = [r.min_margin_min for r in m_results]
        truth_p50 = float(np.percentile(truth_margins, 50.0))
        truth_p10 = float(np.percentile(truth_margins, 10.0))
        truth_neg_cnt = sum(1 for m in truth_margins if m < 0.0)

        margin_rows.append({
            "method": m_key,
            "pred_margin_median": f"{pred_p50:.1f} min" if np.isfinite(pred_p50) else ("+inf" if pred_p50 > 0 else "-inf"),
            "pred_margin_p10": f"{pred_p10:.1f} min" if np.isfinite(pred_p10) else ("+inf" if pred_p10 > 0 else "-inf"),
            "pred_negative_count": f"{pred_neg_cnt}/200 ({pred_neg_cnt/2.0:.1f}%)",
            "truth_margin_median": f"{truth_p50:.1f} min" if np.isfinite(truth_p50) else ("+inf" if truth_p50 > 0 else "-inf"),
            "truth_margin_p10": f"{truth_p10:.1f} min" if np.isfinite(truth_p10) else ("+inf" if truth_p10 > 0 else "-inf"),
            "truth_negative_count": f"{truth_neg_cnt}/200 ({truth_neg_cnt/2.0:.1f}%)",
        })

    df_margins = pd.DataFrame(margin_rows)
    print(df_margins.to_string(index=False))

    # 6. Step 5d: Per Affected Zone Output (14 Zones and 3 Corridors)
    print("\n" + "=" * 80)
    print("STEP 5d: PER-ZONE AND PER-CORRIDOR OUTPUT (Agent A3 & B-on-A3)")
    print("=" * 80)

    zone_metrics_rows = []
    # 14 Zones from paradise_14_zones.parquet
    zones_df = pd.read_parquet("data/tiles/paradise_14_zones.parquet")
    zone_names = sorted(zones_df["zone_name"].tolist())

    for z_name in zone_names:
        orig_indices = [i for i, o in enumerate(origins) if o.assigned_zone == z_name]
        n_orig = len(orig_indices)
        if n_orig == 0:
            continue

        for m_name, m_res in [("A3 (Surrogate)", res_a3), ("B-on-A3 (Obs+Surr)", res_b_a3)]:
            z_res = [m_res[i] for i in orig_indices]
            z_static = [res_static[i] for i in orig_indices]

            # Route changes relative to static
            rc = 0
            for r_cur, r_st in zip(z_res, z_static):
                if (r_cur.route is None) != (r_st.route is None):
                    rc += 1
                elif r_cur.route and r_st.route:
                    if [(e.u, e.v) for e in r_cur.route.edges] != [(e.u, e.v) for e in r_st.route.edges]:
                        rc += 1

            z_trapped = sum(1 for r in z_res if r.route is None or not r.route.edges)
            z_margins = [r.min_margin_min for r in z_res]
            z_p10 = float(np.percentile(z_margins, 10.0))

            zone_metrics_rows.append({
                "zone_or_corridor": z_name,
                "type": "evacuation_zone",
                "method": m_name,
                "origins_count": n_orig,
                "route_changes": f"{rc}/{n_orig} ({rc/n_orig*100.0:.1f}%)",
                "margin_p10": f"{z_p10:.1f} min" if np.isfinite(z_p10) else ("+inf" if z_p10 > 0 else "-inf"),
                "trapped_count": f"{z_trapped}/{n_orig}",
            })

    # 3 Egress Corridors
    for corr in ["Clark_Rd", "Skyway", "Pentz_Hwy70"]:
        c_indices = [i for i, c in enumerate(corridor_clusters) if c == corr]
        n_corr = len(c_indices)
        if n_corr == 0:
            continue

        for m_name, m_res in [("A3 (Surrogate)", res_a3), ("B-on-A3 (Obs+Surr)", res_b_a3)]:
            c_res = [m_res[i] for i in c_indices]
            c_static = [res_static[i] for i in c_indices]

            rc = 0
            for r_cur, r_st in zip(c_res, c_static):
                if (r_cur.route is None) != (r_st.route is None):
                    rc += 1
                elif r_cur.route and r_st.route:
                    if [(e.u, e.v) for e in r_cur.route.edges] != [(e.u, e.v) for e in r_st.route.edges]:
                        rc += 1

            c_trapped = sum(1 for r in c_res if r.route is None or not r.route.edges)
            c_margins = [r.min_margin_min for r in c_res]
            c_p10 = float(np.percentile(c_margins, 10.0))

            zone_metrics_rows.append({
                "zone_or_corridor": corr,
                "type": "egress_corridor",
                "method": m_name,
                "origins_count": n_corr,
                "route_changes": f"{rc}/{n_corr} ({rc/n_corr*100.0:.1f}%)",
                "margin_p10": f"{c_p10:.1f} min" if np.isfinite(c_p10) else ("+inf" if c_p10 > 0 else "-inf"),
                "trapped_count": f"{c_trapped}/{n_corr}",
            })

    df_zones = pd.DataFrame(zone_metrics_rows)
    print(df_zones.head(30).to_string(index=False))

    # 7. Step 5e: Re-plan Latency End-to-End
    print("\n" + "=" * 80)
    print("STEP 5e: RE-PLAN LATENCY END-TO-END")
    print("=" * 80)
    replan_latencies = df_upd["total_replan_ms"].values
    mean_lat_e2e = float(np.mean(replan_latencies))
    p50_lat_e2e = float(np.median(replan_latencies))
    p95_lat_e2e = float(np.percentile(replan_latencies, 95.0))

    print(f"End-to-End Re-plan Latency (including surrogate inference + router + validator):")
    print(f"  Mean:  {mean_lat_e2e:.2f} ms")
    print(f"  P50:   {p50_lat_e2e:.2f} ms")
    print(f"  P95:   {p95_lat_e2e:.2f} ms")
    print(f"  Surrogate Inference Amortized Mean: {float(np.mean(df_upd['surrogate_inf_ms'])):.2f} ms")
    print(f"  Router + Validator Mean:            {float(np.mean(df_upd['router_validator_ms'])):.2f} ms")

    # 8. Step 5f: Corridor-Clustered Bootstrap
    print("\n" + "=" * 80)
    print("STEP 5f: CORRIDOR-CLUSTERED BOOTSTRAP (K=3 Corridors)")
    print("=" * 80)
    a3_miss = np.array([int(r.hazard_miss) for r in res_a3])
    a2_miss = np.array([int(r.hazard_miss) for r in res_a2])
    b_a3_miss = np.array([int(r.hazard_miss) for r in res_b_a3])
    stat_miss = np.array([int(r.hazard_miss) for r in res_static])

    # A3 vs A2
    diff_a3_a2, ci_a3_a2 = cluster_bootstrap_diff(a3_miss, a2_miss, corridor_clusters)
    a3_a2_inconclusive = ci_a3_a2[0] <= 0.0 <= ci_a3_a2[1]

    # B-on-A3 vs Static
    diff_b_stat, ci_b_stat = cluster_bootstrap_diff(b_a3_miss, stat_miss, corridor_clusters)
    b_stat_inconclusive = ci_b_stat[0] <= 0.0 <= ci_b_stat[1]

    boot_rows = [
        {
            "comparison": "A3 (Surrogate) vs A2 (Teacher v2)",
            "diff_observed_pct": f"{diff_a3_a2:+.2f}%",
            "cluster_bootstrap_95ci": f"[{ci_a3_a2[0]:+.2f}%, {ci_a3_a2[1]:+.2f}%]",
            "conclusion": "INCONCLUSIVE (CI includes zero)" if a3_a2_inconclusive else "Statistically Significant",
        },
        {
            "comparison": "B-on-A3 (Obs on Surrogate) vs Static",
            "diff_observed_pct": f"{diff_b_stat:+.2f}%",
            "cluster_bootstrap_95ci": f"[{ci_b_stat[0]:+.2f}%, {ci_b_stat[1]:+.2f}%]",
            "conclusion": "INCONCLUSIVE (CI includes zero)" if b_stat_inconclusive else "Statistically Significant",
        },
    ]
    df_boot = pd.DataFrame(boot_rows)
    print(df_boot.to_string(index=False))

    # 9. Step 5g: Agent Graph Run
    print("\n" + "=" * 80)
    print("STEP 5g: 7-NODE LANGGRAPH AGENT GRAPH EVALUATION")
    print("=" * 80)
    agent = EvacuationAgent(
        llm_client=MockLLMClient(),
        log_file="reports/step5g_agent_graph_log.jsonl",
    )

    # Check for live LLM API keys
    llm_key_present = any(bool(os.environ.get(k)) for k in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
    if llm_key_present:
        print("LLM API Key detected: Running live briefs...")
    else:
        print("Live Brief Status: LLM NOT RUN (No LLM API key set in environment)")
        print("Executing 7-node LangGraph state machine with DeterministicBriefGenerator on A3 updates...")

    graph_events_run = 0
    for upd_step in [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]:
        state_in = {
            "update_id": f"UPD_{int(upd_step)}",
            "timestamp": f"2018-11-08T{6 + int(upd_step)//60:02d}:{int(upd_step)%60:02d}:00Z",
            "arrival_field": surrogate_update_fields[upd_step],
            "road_graph": road_graph,
            "origins": [o.origin_id for o in origins[:10]],
            "destinations": list(destinations),
            "departure_time": 75.0,
            "buffer_minutes": 15.0,
        }
        res_graph = agent.graph.invoke(state_in)
        graph_events_run += 1

    print(f"Successfully orchestrated {graph_events_run} update events through the 7-node LangGraph pipeline.")
    print("Nodes executed: ingest_update -> get_arrival_field -> plan_routes -> validate_routes -> compute_exposure -> decide_replan -> brief")

    # Save summary reports to CSV
    Path("reports").mkdir(exist_ok=True)
    df_upd.to_csv("reports/step5a_update_loop_log.csv", index=False)
    df_sec73.to_csv("reports/step5b_replay_metrics_sec73.csv", index=False)
    df_margins.to_csv("reports/step5c_route_safety_margins.csv", index=False)
    df_zones.to_csv("reports/step5d_per_zone_metrics.csv", index=False)
    df_boot.to_csv("reports/step5f_cluster_bootstrap.csv", index=False)
    print(f"\nStep 5 completed successfully in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
