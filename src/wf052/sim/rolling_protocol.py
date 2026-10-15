"""Phase 1B-2: Rolling-horizon joint calibration protocol, perturbed uncertainty ensemble,
and constant-speed assimilation baseline.

Implements:
1. Rolling-horizon evaluation & joint calibration objective (re-init from every 3rd GOFER
   perimeter, forecast 1-6 h; 12 h point-start reported as secondary only).
2. Multi-source perturbed uncertainty ensemble (seeds + P_base band + wind speed +-15% +
   wind direction +-15 deg + ember params + ignition-extent jitter) with survival-quantile
   P10-P90 arrival coverage and explicit denominators.
3. Nearest assimilated point + constant front speed baseline (fit on training fires,
   evaluated on strictly-future points only; no cumulative metrics).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from wf052.sim.ember_kernel import EmberParams, simulate_teacher_with_embers_60m
from wf052.sim.rothermel_ros import ROSTeacherParams, simulate_ros_teacher_60m
from wf052.sim.teacher_calibrator import (
    FireDataset60m,
    TeacherParams,
    aggregate_to_coarse_fraction,
)


def _signed_and_abs_front_dist_km(
    sim_mask: np.ndarray,
    obs_mask: np.ndarray,
    wind_axis_proj_km: np.ndarray,
    quantile: float = 0.98,
) -> tuple[float, float, float, float]:
    """Return (front_sim_km, front_obs_km, abs_err_km, signed_err_km)."""
    f_sim = float(np.quantile(wind_axis_proj_km[sim_mask], quantile)) if np.any(sim_mask) else 0.0
    f_obs = float(np.quantile(wind_axis_proj_km[obs_mask], quantile)) if np.any(obs_mask) else 0.0
    signed_err = f_sim - f_obs
    return f_sim, f_obs, abs(signed_err), signed_err


def _jitter_initial_mask(mask: np.ndarray, jitter_steps: int, is_burnable: np.ndarray) -> np.ndarray:
    """Apply morphological dilation (+jitter_steps) or erosion (-jitter_steps) to initial extent."""
    if jitter_steps == 0 or not np.any(mask):
        return mask.copy()
    if jitter_steps > 0:
        dil = binary_dilation(mask, iterations=int(jitter_steps))
        return (dil & is_burnable).astype(bool)
    ero = binary_erosion(mask, iterations=int(abs(jitter_steps)))
    return ero.astype(bool) if np.any(ero) else mask.copy()


def get_active_rolling_origins(
    ds: FireDataset60m,
    candidate_origins: tuple[int, ...] = (0, 3, 6, 9),
) -> list[int]:
    """Return every 3rd GOFER perimeter index during active spread (6h growth >= 250 ha and >= 15%)."""
    active: list[int] = []
    max_hr = len(ds.gofer_hourly_masks) - 1
    for t0 in candidate_origins:
        if t0 + 6 > max_hr:
            continue
        a0 = float(ds.gofer_areas_ha[t0])
        a6 = float(ds.gofer_areas_ha[t0 + 6])
        growth_ha = a6 - a0
        rel_growth = growth_ha / max(a0, 100.0)
        if growth_ha >= 250.0 and rel_growth >= 0.15:
            active.append(t0)
    return active if active else [0]


def evaluate_rolling_horizon_fire(
    ds: FireDataset60m,
    teacher_params: TeacherParams,
    ember_params: EmberParams | None = None,
    use_ember_kernel: bool = False,
    ros_params: ROSTeacherParams | None = None,
    origins_h: tuple[int, ...] | None = None,
    horizons_h: tuple[int, ...] = (1, 3, 6),
    seed: int = 42,
    include_12h_secondary: bool = True,
) -> dict[str, Any]:
    """Evaluate a fire under the rolling-horizon protocol (re-init every 3rd GOFER perimeter, 1-6 h forecast)."""
    if ember_params is None:
        ember_params = EmberParams()

    active_origins = list(origins_h) if origins_h is not None else get_active_rolling_origins(ds)
    max_h = max(horizons_h)

    window_rows: list[dict[str, Any]] = []
    horizon_losses: list[float] = []

    # Weights emphasize 1-6h rolling accuracy (IoU, front distance, arrival error, progression area)
    h_weights = {1: 0.25, 3: 0.35, 6: 0.40}

    for t0_h in active_origins:
        t0_min = float(t0_h * 60.0)
        init_mask = ds.gofer_hourly_masks[t0_h]

        if ros_params is not None:
            sim_out = simulate_ros_teacher_60m(
                ds=ds,
                ros_params=ros_params,
                ember_params=ember_params,
                use_ember_kernel=use_ember_kernel,
                duration_hours=max_h,
                steps_per_hour=30,
                seed=seed + t0_h,
                init_burned_mask=init_mask,
                start_hour_index=t0_h,
                start_elapsed_min=t0_min,
            )
        else:
            sim_out = simulate_teacher_with_embers_60m(
                ds=ds,
                params=teacher_params,
                ember_params=ember_params,
                use_ember_kernel=use_ember_kernel,
                duration_hours=max_h,
                steps_per_hour=30,
                seed=seed + t0_h,
                init_burned_mask=init_mask,
                start_hour_index=t0_h,
                start_elapsed_min=t0_min,
            )

        sim_masks = sim_out["hourly_sim_masks"]
        sim_arr_min = sim_out["arrival_time_min"]

        for dt_h in horizons_h:
            target_h = t0_h + dt_h
            s_mask = sim_masks[dt_h]
            o_mask = ds.gofer_hourly_masks[target_h]

            # 1. ~2km GOFER footprint overlap (33x33 block = 1,980m) with denominators
            s_2km = aggregate_to_coarse_fraction(s_mask, 33) >= 0.5
            o_2km = aggregate_to_coarse_fraction(o_mask, 33) >= 0.5
            tp_2km = int(np.count_nonzero(s_2km & o_2km))
            union_2km = int(np.count_nonzero(s_2km | o_2km))
            denom_2km_sum = int(np.count_nonzero(s_2km) + np.count_nonzero(o_2km))
            iou_2km = float(tp_2km / max(union_2km, 1))
            f1_2km = float(2 * tp_2km / max(denom_2km_sum, 1))

            # 2. Strictly-future 60m progression overlap in (T0, T0 + dt_h] with denominators
            s_prog = s_mask & (~init_mask)
            o_prog = o_mask & (~init_mask)
            tp_prog = int(np.count_nonzero(s_prog & o_prog))
            union_prog = int(np.count_nonzero(s_prog | o_prog))
            obs_prog_cells = int(np.count_nonzero(o_prog))
            sim_prog_cells = int(np.count_nonzero(s_prog))
            prog_iou_60m = float(tp_prog / max(union_prog, 1))
            prog_f1_60m = float(2 * tp_prog / max(sim_prog_cells + obs_prog_cells, 1))
            prog_recall_60m = float(tp_prog / max(obs_prog_cells, 1))

            # 3. Total & progression area ratios
            obs_cells = int(np.count_nonzero(o_mask))
            sim_cells = int(np.count_nonzero(s_mask))
            area_ratio = float(sim_cells / max(obs_cells, 1))
            prog_area_ratio = float((sim_prog_cells + 50.0) / (obs_prog_cells + 50.0))

            # 4. Wind-axis front distance error
            f_sim, f_obs, f_abs_km, f_signed_km = _signed_and_abs_front_dist_km(
                s_mask, o_mask, ds.wind_axis_proj_km
            )

            # 5. Strictly-future arrival-time MAE in (T0, T0 + dt_h]
            target_min = float(target_h * 60.0)
            valid_future_obs = (
                (~np.isnan(ds.gofer_arrival_min))
                & (ds.gofer_arrival_min > t0_min)
                & (ds.gofer_arrival_min <= target_min)
            )
            denom_arr_cells = int(np.count_nonzero(valid_future_obs))
            if denom_arr_cells > 0:
                # Censor unreached cells at target_min + 60 min penalty
                s_arr_c = np.where(np.isnan(sim_arr_min), target_min + 60.0, np.minimum(sim_arr_min, target_min + 60.0))
                arr_mae = float(np.mean(np.abs(s_arr_c[valid_future_obs] - ds.gofer_arrival_min[valid_future_obs])))
            else:
                arr_mae = 0.0

            w_h = h_weights.get(dt_h, 0.33)
            dice_loss = 1.0 - f1_2km
            prog_dice_loss = 1.0 - prog_f1_60m
            prog_area_loss = abs(float(np.log(np.clip(prog_area_ratio, 0.1, 10.0))))
            front_loss = f_abs_km / 6.0
            arr_loss = arr_mae / (dt_h * 60.0)
            step_loss = w_h * (
                0.6 * dice_loss
                + 0.8 * prog_dice_loss
                + 0.5 * prog_area_loss
                + 0.6 * front_loss
                + 0.5 * arr_loss
            )
            horizon_losses.append(step_loss)

            window_rows.append(
                {
                    "fire_name": ds.fire_name,
                    "regime": ds.regime,
                    "t0_h": t0_h,
                    "horizon_h": dt_h,
                    "target_h": target_h,
                    "tp_2km": tp_2km,
                    "denom_obs_2km_union": union_2km,
                    "iou_2km": iou_2km,
                    "f1_2km": f1_2km,
                    "tp_prog_60m": tp_prog,
                    "obs_prog_cells_60m": obs_prog_cells,
                    "sim_prog_cells_60m": sim_prog_cells,
                    "union_prog_60m": union_prog,
                    "prog_iou_60m": prog_iou_60m,
                    "prog_recall_60m": prog_recall_60m,
                    "obs_cells_60m": obs_cells,
                    "sim_cells_60m": sim_cells,
                    "area_ratio": area_ratio,
                    "prog_area_ratio": prog_area_ratio,
                    "front_sim_km": f_sim,
                    "front_obs_km": f_obs,
                    "front_dist_err_km": f_abs_km,
                    "front_signed_km": f_signed_km,
                    "denom_arrival_cells": denom_arr_cells,
                    "arrival_mae_min": arr_mae,
                }
            )

    # Secondary 12h point-start evaluation (T0 = 0 -> +12h)
    sec_12h: dict[str, float] = {}
    if include_12h_secondary and len(ds.gofer_hourly_masks) > 12:
        if ros_params is not None:
            out_12 = simulate_ros_teacher_60m(
                ds=ds,
                ros_params=ros_params,
                ember_params=ember_params,
                use_ember_kernel=use_ember_kernel,
                duration_hours=12,
                steps_per_hour=30,
                seed=seed,
            )
        else:
            out_12 = simulate_teacher_with_embers_60m(
                ds=ds,
                params=teacher_params,
                ember_params=ember_params,
                use_ember_kernel=use_ember_kernel,
                duration_hours=12,
                steps_per_hour=30,
                seed=seed,
            )
        s12 = out_12["hourly_sim_masks"][12]
        o12 = ds.gofer_hourly_masks[12]
        s12_2km = aggregate_to_coarse_fraction(s12, 33) >= 0.5
        o12_2km = aggregate_to_coarse_fraction(o12, 33) >= 0.5
        tp12 = int(np.count_nonzero(s12_2km & o12_2km))
        un12 = int(np.count_nonzero(s12_2km | o12_2km))
        _, _, f12_abs, f12_signed = _signed_and_abs_front_dist_km(s12, o12, ds.wind_axis_proj_km)
        sec_12h = {
            "tp_2km_12h": float(tp12),
            "union_2km_12h": float(un12),
            "iou_2km_12h": float(tp12 / max(un12, 1)),
            "area_ratio_12h": float(np.count_nonzero(s12) / max(np.count_nonzero(o12), 1)),
            "front_err_12h_km": f12_abs,
            "front_signed_12h_km": f12_signed,
        }

    return {
        "rolling_loss": float(np.mean(horizon_losses)) if horizon_losses else 999.0,
        "window_rows": window_rows,
        "secondary_12h": sec_12h,
    }


@dataclass(frozen=True)
class PerturbationConfig:
    """Configuration for Part B multi-source parameter/forcing uncertainty ensemble."""

    num_members: int = 20
    p_base_half_width: float = 0.055  # Perturb P_base across [P_base - dP, P_base + dP]
    wind_speed_frac: float = 0.15  # Wind speed perturbation +-15%
    wind_dir_deg: float = 15.0  # Wind direction perturbation +-15 deg
    ember_frac: float = 0.35  # Relative perturbation on alpha_launch and beta_dist
    init_jitter_cells: int = 2  # Morphological dilation/erosion jitter at T0 (+-2 cells = +-120m)


def run_perturbed_rolling_window(
    ds: FireDataset60m,
    base_teacher: TeacherParams,
    base_ember: EmberParams,
    use_ember_kernel: bool,
    t0_h: int,
    duration_h: int,
    perturb_cfg: PerturbationConfig,
    base_seed: int = 1000,
) -> dict[str, Any]:
    """Run an M-member perturbed ensemble for a single rolling forecast window [T0, T0 + duration_h]."""
    m_count = perturb_cfg.num_members
    h, w = ds.fbfm_arr.shape
    t0_min = float(t0_h * 60.0)
    target_min = float((t0_h + duration_h) * 60.0)

    init_base = ds.gofer_hourly_masks[t0_h]
    is_burnable_np = ds.is_burnable.cpu().numpy()

    # Stratified quantile levels u_m in [-1, +1] so ensemble spans full slow-to-fast spread envelope
    u_levels = np.linspace(-1.0, 1.0, m_count)
    rng = np.random.default_rng(base_seed + t0_h * 37)
    ws_offsets = rng.permutation(np.linspace(-perturb_cfg.wind_speed_frac, perturb_cfg.wind_speed_frac, m_count))
    wd_offsets = rng.permutation(np.linspace(-perturb_cfg.wind_dir_deg, perturb_cfg.wind_dir_deg, m_count))

    censor_min = float(target_min + 10000.0)
    arr_stack = np.full((m_count, h, w), censor_min, dtype=np.float32)

    for idx in range(m_count):
        u_val = float(u_levels[idx])
        p_b = float(np.clip(base_teacher.p_base + u_val * perturb_cfg.p_base_half_width, 0.11, 0.38))
        ws_mult = 1.0 + float(ws_offsets[idx])
        wd_off = float(wd_offsets[idx])

        e_mult = max(0.2, 1.0 + u_val * perturb_cfg.ember_frac)
        e_member = EmberParams(
            alpha_launch=base_ember.alpha_launch * e_mult,
            beta_dist_m_per_ms=base_ember.beta_dist_m_per_ms * (1.0 + 0.5 * u_val * perturb_cfg.ember_frac),
            p_urban_wui=float(np.clip(base_ember.p_urban_wui * (1.0 + 0.3 * u_val), 0.02, 0.45)),
        )

        # Jitter initial perimeter: negative u -> erosion, positive u -> dilation
        jitter_k = int(np.round(u_val * perturb_cfg.init_jitter_cells))
        init_m = _jitter_initial_mask(init_base, jitter_k, is_burnable_np)

        out = simulate_teacher_with_embers_60m(
            ds=ds,
            params=TeacherParams(p_base=p_b),
            ember_params=e_member,
            use_ember_kernel=use_ember_kernel,
            duration_hours=duration_h,
            steps_per_hour=30,
            seed=base_seed + idx * 19 + t0_h,
            init_burned_mask=init_m,
            start_hour_index=t0_h,
            start_elapsed_min=t0_min,
            wind_speed_mult=ws_mult,
            wind_dir_offset_deg=wd_off,
        )
        arr = out["arrival_time_min"]
        # Cells not reached by target_min have arrival time > target_min (represented as censor_min for survival quantiles)
        arr_stack[idx] = np.where(np.isnan(arr), censor_min, arr)

    # Strictly-future observed progression cells in (T0, T0 + duration_h]
    obs_arr = ds.gofer_arrival_min
    future_obs_mask = (~np.isnan(obs_arr)) & (obs_arr > t0_min) & (obs_arr <= target_min)
    denom_cells = int(np.count_nonzero(future_obs_mask))

    if denom_cells == 0:
        return {
            "t0_h": t0_h,
            "duration_h": duration_h,
            "denom_obs_progression_cells": 0,
            "covered_cells": 0,
            "coverage_pct": 0.0,
            "coverage_with_denom": "0/0 (0.0%)",
            "raw_covered_cells": 0,
            "raw_coverage_pct": 0.0,
            "raw_coverage_with_denom": "0/0 (0.0%)",
        }

    # Survival quantiles across all M members at observed progression cells
    sub_stack = arr_stack[:, future_obs_mask]  # shape [M, N_future]
    obs_vals = obs_arr[future_obs_mask]  # shape [N_future]

    p10 = np.percentile(sub_stack, 10, axis=0)
    p90 = np.percentile(sub_stack, 90, axis=0)

    # Raw coverage and hourly-quantized (+-30 min for GOFER 60-min discrete step) coverage
    raw_cov = (obs_vals >= p10) & (obs_vals <= p90)
    quant_cov = (obs_vals >= (p10 - 30.0)) & (obs_vals <= (p90 + 30.0))

    raw_n = int(np.count_nonzero(raw_cov))
    quant_n = int(np.count_nonzero(quant_cov))
    raw_pct = 100.0 * raw_n / denom_cells
    quant_pct = 100.0 * quant_n / denom_cells

    return {
        "t0_h": t0_h,
        "duration_h": duration_h,
        "denom_obs_progression_cells": denom_cells,
        "covered_cells": quant_n,
        "coverage_pct": quant_pct,
        "coverage_with_denom": f"{quant_n}/{denom_cells} ({quant_pct:.1f}%)",
        "raw_covered_cells": raw_n,
        "raw_coverage_pct": raw_pct,
        "raw_coverage_with_denom": f"{raw_n}/{denom_cells} ({raw_pct:.1f}%)",
    }


def evaluate_constant_speed_baseline_grid(
    init_mask: np.ndarray,
    obs_arrival_min: np.ndarray,
    t0_min: float,
    horizon_min: float,
    speed_m_per_min: float,
    res_m: float = 60.0,
) -> dict[str, Any]:
    """Evaluate nearest-assimilated-point + constant front speed baseline on a 60m grid."""
    target_min = t0_min + horizon_min
    future_mask = (~np.isnan(obs_arrival_min)) & (obs_arrival_min > t0_min) & (obs_arrival_min <= target_min)
    total_future = int(np.count_nonzero(future_mask))
    if total_future == 0 or not np.any(init_mask):
        return {
            "total_future_cells": total_future,
            "reached_cells": 0,
            "arrival_mae_min": float("nan"),
        }

    dist_m = distance_transform_edt(~init_mask) * float(res_m)
    pred_arrival_min = float(t0_min) + dist_m / max(float(speed_m_per_min), 1e-4)

    reached_future = future_mask & (pred_arrival_min <= target_min)
    n_reached = int(np.count_nonzero(reached_future))
    if n_reached > 0:
        err = np.abs(pred_arrival_min[reached_future] - obs_arrival_min[reached_future])
        mae = float(np.mean(err))
    else:
        mae = float("nan")

    return {
        "total_future_cells": total_future,
        "reached_cells": n_reached,
        "arrival_mae_min": mae,
        "pred_arrival_min": pred_arrival_min,
    }


def fit_constant_front_speed_on_training_fires(
    train_datasets: list[FireDataset60m],
    horizon_h: int = 3,
) -> float:
    """Fit the optimal constant front speed (m/min) on rolling windows of training fires."""
    empirical_speeds: list[float] = []
    for ds in train_datasets:
        origins = get_active_rolling_origins(ds)
        for t0_h in origins:
            t0_min = float(t0_h * 60.0)
            target_min = float((t0_h + horizon_h) * 60.0)
            init_m = ds.gofer_hourly_masks[t0_h]
            if not np.any(init_m):
                continue
            dist_m = distance_transform_edt(~init_m) * float(ds.res_m)
            fut = (~np.isnan(ds.gofer_arrival_min)) & (ds.gofer_arrival_min > t0_min) & (ds.gofer_arrival_min <= target_min)
            if np.any(fut):
                dt = ds.gofer_arrival_min[fut] - t0_min
                d_vals = dist_m[fut]
                valid = (dt > 0.0) & (d_vals > 0.0)
                if np.any(valid):
                    # Use 75th percentile of radial expansion speed so front reaches majority of 3h progression
                    sp = float(np.percentile(d_vals[valid] / dt[valid], 75))
                    empirical_speeds.append(sp)
    return float(np.median(empirical_speeds)) if empirical_speeds else 8.0
