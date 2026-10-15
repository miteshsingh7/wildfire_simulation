"""Analytic synthetic fire spread generators for time-aware router testing.

Generates ArrivalField xarray Datasets with exact analytic arrival times:
1. Expanding-circle fire (isotropic spread from an ignition point).
2. Wind-driven elliptical fire (exact closed-form Huygens arrival time).
3. Selective road-block / barrier fire field to test mid-evacuation cut-offs.
"""

import numpy as np
import xarray as xr

from wf052.routing.contracts import validate_arrival_field


def create_expanding_circle_arrival_field(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    ignition_x: float,
    ignition_y: float,
    spread_rate_m_per_min: float = 20.0,
    horizon_minutes: float = 720.0,
    q10_factor: float = 0.85,
    q90_factor: float = 1.15,
    crs: str = "EPSG:32610",
    ignition_time: str = "2018-11-08T14:33:00Z",
) -> xr.Dataset:
    """Generate an isotropic expanding-circle fire ArrivalField with exact arrival times.

    t_arr = distance / spread_rate_m_per_min
    Cells where t_arr > horizon_minutes are set to +inf.
    """
    xx, yy = np.meshgrid(x_coords, y_coords)
    dist = np.hypot(xx - ignition_x, yy - ignition_y)

    t_arr_q50 = dist / float(spread_rate_m_per_min)
    t_arr_q10 = dist / (float(spread_rate_m_per_min) / float(q10_factor))
    t_arr_q90 = dist / (float(spread_rate_m_per_min) / float(q90_factor))

    # Horizon masking -> +inf
    t_arr_q10[t_arr_q10 > horizon_minutes] = np.inf
    t_arr_q50[t_arr_q50 > horizon_minutes] = np.inf
    t_arr_q90[t_arr_q90 > horizon_minutes] = np.inf

    ds = xr.Dataset(
        data_vars={
            "arrival_q10": (["y", "x"], t_arr_q10.astype(np.float64)),
            "arrival_q50": (["y", "x"], t_arr_q50.astype(np.float64)),
            "arrival_q90": (["y", "x"], t_arr_q90.astype(np.float64)),
        },
        coords={
            "y": y_coords,
            "x": x_coords,
        },
        attrs={
            "crs": str(crs),
            "ignition_time": str(ignition_time),
            "fire_type": "synthetic_expanding_circle",
            "ignition_x": float(ignition_x),
            "ignition_y": float(ignition_y),
            "spread_rate_m_per_min": float(spread_rate_m_per_min),
            "horizon_minutes": float(horizon_minutes),
        },
    )
    validate_arrival_field(ds)
    return ds


def create_wind_ellipse_arrival_field(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    ignition_x: float,
    ignition_y: float,
    head_rate_m_per_min: float = 40.0,
    back_rate_m_per_min: float = 5.0,
    length_to_breadth: float = 2.5,
    wind_towards_deg: float = 225.0,  # 0=East, 90=North, 180=West, 225=Southwest
    horizon_minutes: float = 720.0,
    q10_factor: float = 0.85,
    q90_factor: float = 1.15,
    crs: str = "EPSG:32610",
    ignition_time: str = "2018-11-08T14:33:00Z",
) -> xr.Dataset:
    """Generate an exact closed-form wind-driven Huygens ellipse fire ArrivalField.

    For an ellipse with semi-major axis a = (R_head + R_back)/2,
    semi-minor axis b = a / LB, and center displacement c = (R_head - R_back)/2,
    b^2 = a^2 - c^2, the exact positive arrival time is:
        t_arr(x_rot, y_rot) = (a * r - c * x_rot) / b^2
    """
    xx, yy = np.meshgrid(x_coords, y_coords)
    dx = xx - ignition_x
    dy = yy - ignition_y

    theta_rad = float(np.deg2rad(wind_towards_deg))
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    # Downwind axis (x_rot) and crosswind axis (y_rot)
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t
    r = np.hypot(x_rot, y_rot)

    r_head = float(head_rate_m_per_min)
    r_back = float(back_rate_m_per_min)
    lb = max(1.001, float(length_to_breadth))

    a = (r_head + r_back) / 2.0
    c = (r_head - r_back) / 2.0
    b = a / lb

    # Exact quadratic root for elliptical expansion: ((x_rot - c*t)/a)^2 + (y_rot/b)^2 = t^2
    a2 = a * a
    b2 = b * b
    c2 = c * c
    coeff_a = 1.0 - (c2 / a2)
    coeff_b = (2.0 * c * x_rot) / a2
    coeff_c = -((x_rot * x_rot) / a2 + (y_rot * y_rot) / b2)
    discriminant = np.maximum(0.0, coeff_b * coeff_b - 4.0 * coeff_a * coeff_c)
    t_arr_q50 = np.where(r < 1e-6, 0.0, (-coeff_b + np.sqrt(discriminant)) / (2.0 * coeff_a))
    t_arr_q10 = t_arr_q50 * float(q10_factor)
    t_arr_q90 = t_arr_q50 * float(q90_factor)

    # Horizon masking -> +inf
    t_arr_q10[t_arr_q10 > horizon_minutes] = np.inf
    t_arr_q50[t_arr_q50 > horizon_minutes] = np.inf
    t_arr_q90[t_arr_q90 > horizon_minutes] = np.inf

    ds = xr.Dataset(
        data_vars={
            "arrival_q10": (["y", "x"], t_arr_q10.astype(np.float64)),
            "arrival_q50": (["y", "x"], t_arr_q50.astype(np.float64)),
            "arrival_q90": (["y", "x"], t_arr_q90.astype(np.float64)),
        },
        coords={
            "y": y_coords,
            "x": x_coords,
        },
        attrs={
            "crs": str(crs),
            "ignition_time": str(ignition_time),
            "fire_type": "synthetic_wind_ellipse",
            "ignition_x": float(ignition_x),
            "ignition_y": float(ignition_y),
            "head_rate_m_per_min": float(r_head),
            "back_rate_m_per_min": float(r_back),
            "length_to_breadth": float(lb),
            "wind_towards_deg": float(wind_towards_deg),
            "horizon_minutes": float(horizon_minutes),
        },
    )
    validate_arrival_field(ds)
    return ds


def create_unburned_arrival_field(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    crs: str = "EPSG:32610",
    ignition_time: str = "2018-11-08T14:33:00Z",
) -> xr.Dataset:
    """Generate an entirely unburned ArrivalField (+inf everywhere)."""
    shape = (len(y_coords), len(x_coords))
    inf_arr = np.full(shape, np.inf, dtype=np.float64)
    ds = xr.Dataset(
        data_vars={
            "arrival_q10": (["y", "x"], inf_arr.copy()),
            "arrival_q50": (["y", "x"], inf_arr.copy()),
            "arrival_q90": (["y", "x"], inf_arr.copy()),
        },
        coords={
            "y": y_coords,
            "x": x_coords,
        },
        attrs={
            "crs": str(crs),
            "ignition_time": str(ignition_time),
            "fire_type": "synthetic_unburned",
        },
    )
    validate_arrival_field(ds)
    return ds
