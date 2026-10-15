"""Phase 1B-2 Part D: Time-Boxed Prototype of Rothermel ROS-Driven Contact Rule.

Evaluates:
1. Verification of `wf052.sim.rothermel_ros` against `pyretechnics-2026.8.10` (EPL-2.0) reference
   values across 5 standard FBFM40 fuel models (102 GR2, 122 GS2, 145 SH5, 165 TU5, 188 TL8).
2. Go/No-Go Criterion (1): Smoothness and monotonicity of burned area vs gamma_ros sweep
   (gamma_ros in [0.02..0.16] step 0.01, 10 seeds) on Kincade (2019) and Camp Fire (2018).
3. Go/No-Go Criterion (2): LOFO rolling 1-6 h evaluation on Kincade, Glass, Zogg comparing
   Rothermel ROS-driven contact rule vs P-based contact rule on 1-6 h IoU and front error.
4. Go/No-Go Criterion (3): Wall-clock runtime benchmark comparing Rothermel ROS CA vs P-based CA
   (target: within 2.0x of current CA).
"""

from pathlib import Path
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, "scripts")
from evaluate_heldout_camp_fire import build_camp_fire_dataset_60m

from wf052.sim.ember_kernel import EmberParams, simulate_teacher_with_embers_60m
from wf052.sim.rolling_protocol import evaluate_rolling_horizon_fire
from wf052.sim.rothermel_ros import (
    ROSTeacherParams,
    compute_rothermel_max_behavior,
    compute_rothermel_no_wind_no_slope,
    simulate_ros_teacher_60m,
)
from wf052.sim.teacher_calibrator import (
    FireDataset60m,
    TeacherParams,
    load_fire_dataset_60m,
)

WIND_DRIVEN_FIRES = ("Kincade", "Glass", "Zogg")

# Golden reference values extracted directly from pyretechnics-2026.8.10 (EPL-2.0)
PYRETECHNICS_REF: dict[int, dict[str, Any]] = {
    102: {
        "code": "GR2",
        "base_spread_rate": 0.6787914542031215,
        "base_fireline_intensity": 59.56666111604902,
        "phiS_G": 49.080376640232345,
        "phiW_scalr": 0.025557941580860583,
        "phiW_expnt": 1.4835859429132645,
        "max_spread_rate": 38.74587320742831,
        "length_to_width_ratio": 2.4161401969451366,
        "eccentricity": 0.9103861294592174,
    },
    122: {
        "code": "GS2",
        "base_spread_rate": 0.6251153512921849,
        "base_fireline_intensity": 93.41391944914797,
        "phiS_G": 42.33728219220262,
        "phiW_scalr": 0.017792074217371588,
        "phiW_expnt": 1.4744180200668443,
        "max_spread_rate": 24.479367814974615,
        "length_to_width_ratio": 2.4139873397848056,
        "eccentricity": 0.9102109412767779,
    },
    145: {
        "code": "SH5",
        "base_spread_rate": 1.1583635069967925,
        "base_fireline_intensity": 574.1605900181529,
        "phiS_G": 45.77829400505166,
        "phiW_scalr": 0.0566037067847082,
        "phiW_expnt": 1.2446840034978827,
        "max_spread_rate": 43.363598225951636,
        "length_to_width_ratio": 2.416549111912714,
        "eccentricity": 0.9104193524328024,
    },
    165: {
        "code": "TU5",
        "base_spread_rate": 0.36988907906064844,
        "base_fireline_intensity": 173.49269439188212,
        "phiS_G": 23.621963038148252,
        "phiW_scalr": 0.015641471099247514,
        "phiW_expnt": 1.1422979540903335,
        "max_spread_rate": 2.5784592942701256,
        "length_to_width_ratio": 2.4173698486365384,
        "eccentricity": 0.9104859571667022,
    },
    188: {
        "code": "TL8",
        "base_spread_rate": 0.2512848528741573,
        "base_fireline_intensity": 66.20892427920744,
        "phiS_G": 16.58559568981349,
        "phiW_scalr": 0.008074489050606194,
        "phiW_expnt": 1.4310654325285507,
        "max_spread_rate": 3.8419655712854024,
        "length_to_width_ratio": 2.4170147949642073,
        "eccentricity": 0.9104571535648834,
    },
}


def run_pyretechnics_verification() -> pd.DataFrame:
    """Verify wf052.sim.rothermel_ros against pyretechnics-2026.8.10 across 5 FBFM40 fuels."""
    moist = (0.06, 0.07, 0.08, 0.0, 0.60, 0.90)
    rows: list[dict[str, Any]] = []
    for fm_num, ref in PYRETECHNICS_REF.items():
        bmin = compute_rothermel_no_wind_no_slope(fm_num, moist)
        bmax = compute_rothermel_max_behavior(
            bmin,
            midflame_wind_speed_m_min=180.0,
            upwind_direction_deg=45.0,
            slope_rise_run=0.20,
            aspect_deg=225.0,
        )
        rel_errs = [
            abs(bmin["base_spread_rate"] - ref["base_spread_rate"]) / ref["base_spread_rate"],
            abs(bmin["base_fireline_intensity"] - ref["base_fireline_intensity"]) / ref["base_fireline_intensity"],
            abs(bmin["phiS_G"] - ref["phiS_G"]) / ref["phiS_G"],
            abs(bmin["phiW_scalr"] - ref["phiW_scalr"]) / ref["phiW_scalr"],
            abs(bmax["max_spread_rate"] - ref["max_spread_rate"]) / ref["max_spread_rate"],
            abs(bmax["length_to_width_ratio"] - ref["length_to_width_ratio"]) / ref["length_to_width_ratio"],
        ]
        rows.append(
            {
                "fbfm40_num": fm_num,
                "fbfm40_code": ref["code"],
                "pyretechnics_R0_m_min": ref["base_spread_rate"],
                "wf052_R0_m_min": bmin["base_spread_rate"],
                "pyretechnics_Rmax_m_min": ref["max_spread_rate"],
                "wf052_Rmax_m_min": bmax["max_spread_rate"],
                "pyretechnics_I0_kW_m": ref["base_fireline_intensity"],
                "wf052_I0_kW_m": bmin["base_fireline_intensity"],
                "wf052_residence_time_min": bmin["residence_time_min"],
                "max_rel_err": max(rel_errs),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv("reports/phase1b2_partD_pyretechnics_verification.csv", index=False)
    return df


def run_gamma_smoothness_sweep(
    ds_kincade: FireDataset60m,
    ds_camp: FireDataset60m,
) -> pd.DataFrame:
    """Sweep gamma_ros in [0.02..0.16] step 0.01 across 10 seeds on Kincade and Camp Fire."""
    gammas = np.round(np.arange(0.02, 0.165, 0.01), 3)
    seeds = list(range(100, 110))
    rows: list[dict[str, Any]] = []
    cell_ha = 0.36

    for fire_label, split_lbl, ds, obs_12h_ha in [
        ("Kincade (2019)", "in-sample / wind-driven fold", ds_kincade, float(ds_kincade.gofer_areas_ha[12])),
        ("Camp Fire (2018)", "development", ds_camp, 32333.7),
    ]:
        prev_med: float | None = None
        for g_val in gammas:
            areas_12h: list[float] = []
            for sd in seeds:
                out = simulate_ros_teacher_60m(
                    ds=ds,
                    ros_params=ROSTeacherParams(gamma_ros=float(g_val), res_time_scale=1.5),
                    use_ember_kernel=False,
                    duration_hours=12,
                    steps_per_hour=30,
                    seed=sd,
                )
                areas_12h.append(float(np.count_nonzero(out["hourly_sim_masks"][12]) * cell_ha))
            arr_np = np.array(areas_12h)
            med_ha = float(np.median(arr_np))
            p10_ha = float(np.percentile(arr_np, 10))
            p90_ha = float(np.percentile(arr_np, 90))
            step_delta = (med_ha - prev_med) if prev_med is not None else 0.0
            prev_med = med_ha
            rows.append(
                {
                    "split": split_lbl,
                    "fire_name": fire_label,
                    "gamma_ros": float(g_val),
                    "num_seeds": len(seeds),
                    "obs_12h_ha": obs_12h_ha,
                    "median_area_12h_ha": med_ha,
                    "p10_area_12h_ha": p10_ha,
                    "p90_area_12h_ha": p90_ha,
                    "step_delta_ha": step_delta,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv("reports/phase1b2_partD_gamma_sweep.csv", index=False)
    return df


def run_ros_lofo_rolling_comparison(
    wind_datasets: dict[str, FireDataset60m],
    p_base_off: float,
    p_base_on: float,
    ember_on: EmberParams,
) -> pd.DataFrame:
    """Fit ROSTeacherParams via LOFO on Kincade, Glass, Zogg and compare against P-based teacher."""
    candidate_ros = [
        ROSTeacherParams(gamma_ros=g, res_time_scale=rt)
        for g in (0.045, 0.060, 0.075, 0.090, 0.105, 0.120)
        for rt in (1.0, 1.5)
    ]

    # Cache rolling losses & window rows for each candidate on each wind fire
    ros_off_cache: dict[tuple[float, float], dict[str, dict[str, Any]]] = {}
    ros_on_cache: dict[tuple[float, float], dict[str, dict[str, Any]]] = {}
    for rp in candidate_ros:
        key = (rp.gamma_ros, rp.res_time_scale)
        ros_off_cache[key] = {}
        ros_on_cache[key] = {}
        for fn, ds in wind_datasets.items():
            ros_off_cache[key][fn] = evaluate_rolling_horizon_fire(
                ds=ds,
                teacher_params=TeacherParams(p_base=0.22),
                ember_params=EmberParams(),
                use_ember_kernel=False,
                ros_params=rp,
                horizons_h=(1, 3, 6),
                seed=42,
                include_12h_secondary=True,
            )
            ros_on_cache[key][fn] = evaluate_rolling_horizon_fire(
                ds=ds,
                teacher_params=TeacherParams(p_base=0.22),
                ember_params=ember_on,
                use_ember_kernel=True,
                ros_params=rp,
                horizons_h=(1, 3, 6),
                seed=42,
                include_12h_secondary=True,
            )

    # LOFO selection on the 3 wind-driven fires
    lofo_ros_off_rows: list[dict[str, Any]] = []
    lofo_ros_on_rows: list[dict[str, Any]] = []
    pbased_off_rows: list[dict[str, Any]] = []
    pbased_on_rows: list[dict[str, Any]] = []

    for held_out in WIND_DRIVEN_FIRES:
        train_names = [f for f in WIND_DRIVEN_FIRES if f != held_out]
        best_key_off = min(
            ros_off_cache.keys(),
            key=lambda k: float(np.mean([ros_off_cache[k][tn]["rolling_loss"] for tn in train_names])),
        )
        best_key_on = min(
            ros_on_cache.keys(),
            key=lambda k: float(np.mean([ros_on_cache[k][tn]["rolling_loss"] for tn in train_names])),
        )
        lofo_ros_off_rows.extend(ros_off_cache[best_key_off][held_out]["window_rows"])
        lofo_ros_on_rows.extend(ros_on_cache[best_key_on][held_out]["window_rows"])

        ds_ho = wind_datasets[held_out]
        res_p_off = evaluate_rolling_horizon_fire(
            ds=ds_ho,
            teacher_params=TeacherParams(p_base=p_base_off),
            ember_params=EmberParams(),
            use_ember_kernel=False,
            horizons_h=(1, 3, 6),
            seed=42,
            include_12h_secondary=False,
        )
        res_p_on = evaluate_rolling_horizon_fire(
            ds=ds_ho,
            teacher_params=TeacherParams(p_base=p_base_on),
            ember_params=ember_on,
            use_ember_kernel=True,
            horizons_h=(1, 3, 6),
            seed=42,
            include_12h_secondary=False,
        )
        pbased_off_rows.extend(res_p_off["window_rows"])
        pbased_on_rows.extend(res_p_on["window_rows"])

    summary_rows: list[dict[str, Any]] = []
    for model_lbl, w_rows in [
        (f"P-Based Teacher (Ember OFF, P_base={p_base_off:.2f})", pbased_off_rows),
        (
            f"P-Based Teacher v2 (Ember ON, P_base={p_base_on:.2f}, alpha={ember_on.alpha_launch:.4f})",
            pbased_on_rows,
        ),
        ("Rothermel ROS Prototype (Ember OFF, LOFO Fit)", lofo_ros_off_rows),
        ("Rothermel ROS Prototype (Ember ON, LOFO Fit)", lofo_ros_on_rows),
    ]:
        df_w = pd.DataFrame(w_rows)
        for h_val in [1, 3, 6, "1-6h_mean"]:
            sub = df_w if h_val == "1-6h_mean" else df_w[df_w["horizon_h"] == h_val]
            tp_2km = int(sub["tp_2km"].sum())
            un_2km = int(sub["denom_obs_2km_union"].sum())
            tp_prog = int(sub["tp_prog_60m"].sum())
            un_prog = int(sub["union_prog_60m"].sum())
            summary_rows.append(
                {
                    "split": "held-out (3 wind-driven LOFO folds: Kincade, Glass, Zogg)",
                    "model": model_lbl,
                    "horizon": f"+{h_val} h" if isinstance(h_val, int) else "1-6 h mean",
                    "num_windows": len(sub),
                    "pooled_2km_tp_over_union": f"{tp_2km}/{un_2km}",
                    "mean_iou_2km": float(sub["iou_2km"].mean()),
                    "mean_f1_2km": float(sub["f1_2km"].mean()),
                    "pooled_prog_tp_over_union": f"{tp_prog}/{un_prog}",
                    "mean_prog_iou_60m": float(sub["prog_iou_60m"].mean()),
                    "mean_front_err_km": float(sub["front_dist_err_km"].mean()),
                    "mean_arrival_mae_min": float(sub["arrival_mae_min"].mean()),
                }
            )
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv("reports/phase1b2_partD_ros_vs_pbased_lofo.csv", index=False)
    return df_sum


def run_runtime_benchmark(ds_kincade: FireDataset60m, ds_camp: FireDataset60m) -> pd.DataFrame:
    """Measure wall-clock runtime of P-based CA vs Rothermel ROS CA."""
    rows: list[dict[str, Any]] = []
    for fire_lbl, split_lbl, ds in [
        ("Kincade (2019, 550x550 grid)", "in-sample / wind-driven fold", ds_kincade),
        ("Camp Fire (2018, 562x602 grid)", "development", ds_camp),
    ]:
        # Warmup
        _ = simulate_teacher_with_embers_60m(ds, TeacherParams(p_base=0.22), duration_hours=6, seed=1)
        _ = simulate_ros_teacher_60m(ds, ROSTeacherParams(gamma_ros=0.075), duration_hours=6, seed=1)

        t0 = time.perf_counter()
        for idx in range(5):
            _ = simulate_teacher_with_embers_60m(ds, TeacherParams(p_base=0.22), duration_hours=6, seed=10 + idx)
        pbased_ms = (time.perf_counter() - t0) * 1000.0 / 5.0

        t1 = time.perf_counter()
        for idx in range(5):
            _ = simulate_ros_teacher_60m(ds, ROSTeacherParams(gamma_ros=0.075), duration_hours=6, seed=10 + idx)
        ros_ms = (time.perf_counter() - t1) * 1000.0 / 5.0

        rows.append(
            {
                "split": split_lbl,
                "fire_domain": fire_lbl,
                "horizon_hours": 6,
                "num_trials": 5,
                "pbased_ca_runtime_ms": pbased_ms,
                "rothermel_ros_ca_runtime_ms": ros_ms,
                "runtime_ratio_ros_over_pbased": ros_ms / max(pbased_ms, 1e-3),
                "passes_2x_criterion": bool((ros_ms / max(pbased_ms, 1e-3)) <= 2.0),
            }
        )
    df_rt = pd.DataFrame(rows)
    df_rt.to_csv("reports/phase1b2_partD_runtime_benchmark.csv", index=False)
    return df_rt


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Running Phase 1B-2 Part D (Rothermel ROS-Driven Contact Rule Prototype) on {device}...")

    # 1. Verify against pyretechnics-2026.8.10
    df_ver = run_pyretechnics_verification()
    print("\n=== PART D.1: Verification vs pyretechnics-2026.8.10 (EPL-2.0) ===")
    print(df_ver.to_string(index=False))

    # 2. Load datasets
    setup_df = pd.read_csv("reports/phase_1a_fires_setup.csv")
    wind_datasets: dict[str, FireDataset60m] = {}
    for _, row in setup_df.iterrows():
        fname = str(row["fire_name"])
        regime = str(row["regime"])
        if fname in WIND_DRIVEN_FIRES:
            t_dir = Path("data/tiles/fires") / fname.lower()
            wind_datasets[fname] = load_fire_dataset_60m(fname, regime, t_dir, device=device)
    ds_camp, _, _ = build_camp_fire_dataset_60m(device)

    # 3. Criterion (1): Gamma smoothness & monotonicity sweep
    df_sweep = run_gamma_smoothness_sweep(wind_datasets["Kincade"], ds_camp)
    print("\n=== PART D.2: Gamma_ROS Smoothness & Monotonicity Sweep (10 seeds) ===")
    print(df_sweep.to_string(index=False))

    # 4. Criterion (2): LOFO Rolling 1-6 h comparison vs P-based teacher
    cfg_v2_path = Path("configs/teacher_v2.yaml")
    if cfg_v2_path.exists():
        with open(cfg_v2_path, encoding="utf-8") as f:
            cfg_v2 = yaml.safe_load(f)
        c_key = "contact_parameters" if "contact_parameters" in cfg_v2 else "teacher_parameters"
        p_base_on = float(cfg_v2[c_key]["p_base"])
        e_cfg = cfg_v2["ember_parameters"]
        ember_on = EmberParams(
            alpha_launch=float(e_cfg["alpha_launch"]),
            beta_dist_m_per_ms=float(e_cfg["beta_dist_m_per_ms"]),
            p_urban_wui=float(e_cfg["p_urban_wui"]),
        )
        p_base_off = float(cfg_v2.get("rolling_ember_off_baseline", {}).get("p_base", p_base_on))
    else:
        p_base_on = 0.20
        p_base_off = 0.21
        ember_on = EmberParams(alpha_launch=0.0020, beta_dist_m_per_ms=110.0, p_urban_wui=0.16)

    df_lofo = run_ros_lofo_rolling_comparison(wind_datasets, p_base_off, p_base_on, ember_on)
    print("\n=== PART D.3: LOFO Rolling 1-6 h Comparison (Rothermel ROS vs P-Based Teacher) ===")
    print(df_lofo.to_string(index=False))

    # 5. Criterion (3): Runtime benchmark
    df_rt = run_runtime_benchmark(wind_datasets["Kincade"], ds_camp)
    print("\n=== PART D.4: Runtime Benchmark (6 h Forecast Window, 5 Trials) ===")
    print(df_rt.to_string(index=False))

    # 6. Plot Sweep & LOFO comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    for fire_lbl, color, marker in [
        ("Kincade (2019)", "#1f77b4", "o"),
        ("Camp Fire (2018)", "#d62728", "s"),
    ]:
        sub_s = df_sweep[df_sweep["fire_name"] == fire_lbl]
        axes[0].plot(
            sub_s["gamma_ros"],
            sub_s["median_area_12h_ha"],
            marker=marker,
            color=color,
            linewidth=2,
            label=f"{fire_lbl} Median (10 seeds)",
        )
        axes[0].fill_between(
            sub_s["gamma_ros"],
            sub_s["p10_area_12h_ha"],
            sub_s["p90_area_12h_ha"],
            color=color,
            alpha=0.2,
            label=f"{fire_lbl} P10-P90",
        )
    axes[0].set_xlabel("Hazard Scale gamma_ros in p = 1 - exp(-gamma * ROS * dt / d)")
    axes[0].set_ylabel("Simulated Burned Area at +12 h (ha)")
    axes[0].set_title("A. Criterion (1): Burned Area vs gamma_ros (10 Seeds)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    sub_mean = df_lofo[df_lofo["horizon"] == "1-6 h mean"].copy()
    x_pos = np.arange(len(sub_mean))
    ax2 = axes[1]
    ax2_r = ax2.twinx()
    w_bar = 0.35
    ax2.bar(
        x_pos - w_bar / 2,
        sub_mean["mean_iou_2km"],
        w_bar,
        color="#2ca02c",
        edgecolor="black",
        label="Rolling 1-6h Mean 2km IoU (Higher=Better)",
    )
    ax2_r.bar(
        x_pos + w_bar / 2,
        sub_mean["mean_front_err_km"],
        w_bar,
        color="#ff7f0e",
        edgecolor="black",
        label="Rolling 1-6h Mean Front Error km (Lower=Better)",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        ["P-Based\n(Ember OFF)", "P-Based v2\n(Ember ON)", "Rothermel ROS\n(Ember OFF)", "Rothermel ROS\n(Ember ON)"],
        fontsize=8.5,
    )
    ax2.set_ylabel("Rolling 1-6h Mean 2km IoU", color="#2ca02c")
    ax2_r.set_ylabel("Rolling 1-6h Mean Front Error (km)", color="#ff7f0e")
    ax2.set_title("B. Criterion (2): Held-Out LOFO Rolling 1-6h Performance")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/phase_1b2_partD_ros_sweep_and_eval.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
