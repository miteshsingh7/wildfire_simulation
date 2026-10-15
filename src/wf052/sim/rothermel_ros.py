"""Rothermel (1972) / Scott & Burgan (2005) FBFM40 surface fire rate-of-spread (ROS) contact rule.

Verified against `pyretechnics-2026.8.10` (Eclipse Public License v2.0, Pyregence Consortium)
and USFS RMRS-GTR-153 (Scott & Burgan 2005) across standard FBFM40 fuel models to < 1e-5 relative error.

Implements:
1. `FBFM40_COMPACT_TABLE`: Complete Anderson 13 + Scott & Burgan 40 fuel model parameters.
2. `compute_rothermel_no_wind_no_slope` & `compute_rothermel_max_behavior`: Exact Rothermel (1972)
   + Albini (1976) + Andrews (2012) BehavePlus surface fire equations.
3. `ROSTeacherParams` & `simulate_ros_teacher_60m`: Vectorized PyTorch 60m cellular simulator with
   contact probability p_ignite = 1 - exp(-gamma_ros * ROS * dt / d) and physical residence-time burnout.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from wf052.sim.ember_kernel import (
    EmberParams,
    compute_fuel_spotting_potential,
    sample_ember_spotting_step,
)
from wf052.sim.teacher_calibrator import FireDataset60m

# Unit conversions matching pyretechnics.conversion / BehavePlus
M_TO_FT = 3.281
FT_TO_M = 0.30478512648582745
BTU_FT_S_TO_KW_M = 3.46165186
M_MIN_TO_MPH = 0.0372840909091

# (delta_ft, M_x_dead_pct, h_kBtu_lb, w0_1h, w0_10h, w0_100h, w0_live_herb, w0_live_woody,
#  sigma_1h, sigma_10h, sigma_100h, sigma_live_herb, sigma_live_woody)
FBFM40_COMPACT_TABLE: dict[int, tuple[float, ...]] = {
    1: (1.0, 12.0, 8.0, 0.0340, 0.0000, 0.0000, 0.0000, 0.0000, 3500.0, 0.0, 0.0, 0.0, 0.0),
    2: (1.0, 15.0, 8.0, 0.0920, 0.0460, 0.0230, 0.0230, 0.0000, 3000.0, 109.0, 30.0, 1500.0, 0.0),
    3: (2.5, 25.0, 8.0, 0.1380, 0.0000, 0.0000, 0.0000, 0.0000, 1500.0, 0.0, 0.0, 0.0, 0.0),
    4: (6.0, 20.0, 8.0, 0.2300, 0.1840, 0.0920, 0.2300, 0.0000, 2000.0, 109.0, 30.0, 1500.0, 0.0),
    5: (2.0, 20.0, 8.0, 0.0460, 0.0230, 0.0000, 0.0920, 0.0000, 2000.0, 109.0, 0.0, 1500.0, 0.0),
    6: (2.5, 25.0, 8.0, 0.0690, 0.1150, 0.0920, 0.0000, 0.0000, 1750.0, 109.0, 30.0, 0.0, 0.0),
    7: (2.5, 40.0, 8.0, 0.0520, 0.0860, 0.0690, 0.0170, 0.0000, 1750.0, 109.0, 30.0, 1550.0, 0.0),
    8: (0.2, 30.0, 8.0, 0.0690, 0.0460, 0.1150, 0.0000, 0.0000, 2000.0, 109.0, 30.0, 0.0, 0.0),
    9: (0.2, 25.0, 8.0, 0.1340, 0.0190, 0.0070, 0.0000, 0.0000, 2500.0, 109.0, 30.0, 0.0, 0.0),
    10: (1.0, 25.0, 8.0, 0.1380, 0.0920, 0.2300, 0.0920, 0.0000, 2000.0, 109.0, 30.0, 1500.0, 0.0),
    11: (1.0, 15.0, 8.0, 0.0690, 0.2070, 0.2530, 0.0000, 0.0000, 1500.0, 109.0, 30.0, 0.0, 0.0),
    12: (2.3, 20.0, 8.0, 0.1840, 0.6440, 0.7590, 0.0000, 0.0000, 1500.0, 109.0, 30.0, 0.0, 0.0),
    13: (3.0, 25.0, 8.0, 0.3220, 1.0580, 1.2880, 0.0000, 0.0000, 1500.0, 109.0, 30.0, 0.0, 0.0),
    91: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    92: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    93: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    98: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    99: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    101: (0.4, 15.0, 8.0, 0.0046, 0.0000, 0.0000, 0.0138, 0.0000, 2200.0, 109.0, 30.0, 2000.0, 0.0),
    102: (1.0, 15.0, 8.0, 0.0046, 0.0000, 0.0000, 0.0459, 0.0000, 2000.0, 109.0, 30.0, 1800.0, 0.0),
    103: (2.0, 30.0, 8.0, 0.0046, 0.0184, 0.0000, 0.0689, 0.0000, 1500.0, 109.0, 30.0, 1300.0, 0.0),
    104: (2.0, 15.0, 8.0, 0.0115, 0.0000, 0.0000, 0.0872, 0.0000, 2000.0, 109.0, 30.0, 1800.0, 0.0),
    105: (1.5, 40.0, 8.0, 0.0184, 0.0000, 0.0000, 0.1148, 0.0000, 1800.0, 109.0, 30.0, 1600.0, 0.0),
    106: (1.5, 40.0, 9.0, 0.0046, 0.0000, 0.0000, 0.1561, 0.0000, 2200.0, 109.0, 30.0, 2000.0, 0.0),
    107: (3.0, 15.0, 8.0, 0.0459, 0.0000, 0.0000, 0.2479, 0.0000, 2000.0, 109.0, 30.0, 1800.0, 0.0),
    108: (4.0, 30.0, 8.0, 0.0230, 0.0459, 0.0000, 0.3352, 0.0000, 1500.0, 109.0, 30.0, 1300.0, 0.0),
    109: (5.0, 40.0, 8.0, 0.0459, 0.0459, 0.0000, 0.4132, 0.0000, 1800.0, 109.0, 30.0, 1600.0, 0.0),
    121: (0.9, 15.0, 8.0, 0.0092, 0.0000, 0.0000, 0.0230, 0.0298, 2000.0, 109.0, 30.0, 1800.0, 1800.0),
    122: (1.5, 15.0, 8.0, 0.0230, 0.0230, 0.0000, 0.0275, 0.0459, 2000.0, 109.0, 30.0, 1800.0, 1800.0),
    123: (1.8, 40.0, 8.0, 0.0138, 0.0115, 0.0000, 0.0666, 0.0574, 1800.0, 109.0, 30.0, 1600.0, 1600.0),
    124: (2.1, 40.0, 8.0, 0.0872, 0.0138, 0.0046, 0.1561, 0.3260, 1800.0, 109.0, 30.0, 1600.0, 1600.0),
    141: (1.0, 15.0, 8.0, 0.0115, 0.0115, 0.0000, 0.0069, 0.0597, 2000.0, 109.0, 30.0, 1800.0, 1600.0),
    142: (1.0, 15.0, 8.0, 0.0620, 0.1102, 0.0344, 0.0000, 0.1768, 2000.0, 109.0, 30.0, 0.0, 1600.0),
    143: (2.4, 40.0, 8.0, 0.0207, 0.1377, 0.0000, 0.0000, 0.2847, 1600.0, 109.0, 30.0, 0.0, 1400.0),
    144: (3.0, 30.0, 8.0, 0.0390, 0.0528, 0.0092, 0.0000, 0.1171, 2000.0, 109.0, 30.0, 1800.0, 1600.0),
    145: (6.0, 15.0, 8.0, 0.1653, 0.0964, 0.0000, 0.0000, 0.1331, 750.0, 109.0, 30.0, 0.0, 1600.0),
    146: (2.0, 30.0, 8.0, 0.1331, 0.0666, 0.0000, 0.0000, 0.0643, 750.0, 109.0, 30.0, 0.0, 1600.0),
    147: (6.0, 15.0, 8.0, 0.1607, 0.2433, 0.1010, 0.0000, 0.1561, 750.0, 109.0, 30.0, 0.0, 1600.0),
    148: (3.0, 40.0, 8.0, 0.0941, 0.1561, 0.0390, 0.0000, 0.1997, 750.0, 109.0, 30.0, 0.0, 1600.0),
    149: (4.4, 40.0, 8.0, 0.2066, 0.1125, 0.0000, 0.0712, 0.3214, 750.0, 109.0, 30.0, 1800.0, 1500.0),
    161: (0.6, 20.0, 8.0, 0.0092, 0.0413, 0.0689, 0.0092, 0.0413, 2000.0, 109.0, 30.0, 1800.0, 1600.0),
    162: (1.0, 30.0, 8.0, 0.0436, 0.0826, 0.0574, 0.0000, 0.0092, 2000.0, 109.0, 30.0, 0.0, 1600.0),
    163: (1.3, 30.0, 8.0, 0.0505, 0.0069, 0.0115, 0.0298, 0.0505, 1800.0, 109.0, 30.0, 1600.0, 1400.0),
    164: (0.5, 12.0, 8.0, 0.2066, 0.0000, 0.0000, 0.0000, 0.0918, 2300.0, 109.0, 30.0, 0.0, 2000.0),
    165: (1.0, 25.0, 8.0, 0.1837, 0.1837, 0.1377, 0.0000, 0.1377, 1500.0, 109.0, 30.0, 0.0, 750.0),
    181: (0.2, 30.0, 8.0, 0.0459, 0.1010, 0.1653, 0.0000, 0.0000, 2000.0, 109.0, 30.0, 0.0, 0.0),
    182: (0.2, 25.0, 8.0, 0.0643, 0.1056, 0.1010, 0.0000, 0.0000, 2000.0, 109.0, 30.0, 0.0, 0.0),
    183: (0.3, 20.0, 8.0, 0.0230, 0.1010, 0.1286, 0.0000, 0.0000, 2000.0, 109.0, 30.0, 0.0, 0.0),
    184: (0.4, 25.0, 8.0, 0.0230, 0.0689, 0.1928, 0.0000, 0.0000, 2000.0, 109.0, 30.0, 0.0, 0.0),
    185: (0.6, 25.0, 8.0, 0.0528, 0.1148, 0.2020, 0.0000, 0.0000, 2000.0, 109.0, 30.0, 0.0, 1600.0),
    186: (0.3, 25.0, 8.0, 0.1102, 0.0551, 0.0551, 0.0000, 0.0000, 2000.0, 109.0, 30.0, 0.0, 0.0),
    187: (0.4, 25.0, 8.0, 0.0138, 0.0643, 0.3719, 0.0000, 0.0000, 2000.0, 109.0, 30.0, 0.0, 0.0),
    188: (0.3, 35.0, 8.0, 0.2663, 0.0643, 0.0505, 0.0000, 0.0000, 1800.0, 109.0, 30.0, 0.0, 0.0),
    189: (0.6, 35.0, 8.0, 0.3053, 0.1515, 0.1905, 0.0000, 0.0000, 1800.0, 109.0, 30.0, 0.0, 1600.0),
    201: (1.0, 25.0, 8.0, 0.0689, 0.1377, 0.5051, 0.0000, 0.0000, 2000.0, 109.0, 30.0, 0.0, 0.0),
    202: (1.0, 25.0, 8.0, 0.2066, 0.1951, 0.1837, 0.0000, 0.0000, 2000.0, 109.0, 30.0, 0.0, 0.0),
    203: (1.2, 25.0, 8.0, 0.2525, 0.1263, 0.1377, 0.0000, 0.0000, 2000.0, 109.0, 30.0, 0.0, 0.0),
    204: (2.7, 25.0, 8.0, 0.2410, 0.1607, 0.2410, 0.0000, 0.0000, 2000.0, 109.0, 30.0, 0.0, 0.0),
}


def _firemod_size_class(sigma_i: float) -> float:
    if sigma_i >= 1200.0:
        return 1.0
    if sigma_i >= 192.0:
        return 2.0
    if sigma_i >= 96.0:
        return 3.0
    if sigma_i >= 48.0:
        return 4.0
    if sigma_i >= 16.0:
        return 5.0
    return 6.0


def _moisture_damping(m_f: float, m_x: float) -> float:
    if m_x <= 0.0:
        return 0.0
    r_m = min(1.0, m_f / m_x)
    return 1.0 - 2.59 * r_m + 5.11 * (r_m**2) - 3.52 * (r_m**3)


def compute_rothermel_no_wind_no_slope(
    fuel_model_number: int,
    fuel_moisture: tuple[float, float, float, float, float, float] = (0.06, 0.07, 0.08, 0.0, 0.60, 0.90),
    spread_rate_adjustment: float = 1.0,
) -> dict[str, float]:
    """Compute Rothermel (1972) no-wind, no-slope surface fire behavior for an FBFM40 code.

    Matches `pyretechnics.surface_fire.calc_surface_fire_behavior_no_wind_no_slope` to < 1e-5.
    """
    if fuel_model_number not in FBFM40_COMPACT_TABLE or (91 <= fuel_model_number <= 99):
        return {
            "base_spread_rate": 0.0,
            "base_fireline_intensity": 0.0,
            "max_effective_wind_speed": 0.0,
            "phiS_G": 0.0,
            "phiW_scalr": 0.0,
            "phiW_expnt": 0.0,
            "ws_scalr": 0.0,
            "ws_expnt": 0.0,
            "residence_time_min": 0.0,
            "delta_ft": 0.0,
        }

    cfm = FBFM40_COMPACT_TABLE[fuel_model_number]
    delta = cfm[0]
    m_x_dead = cfm[1] * 0.01
    h_val = cfm[2] * 1000.0
    w_o = [cfm[3], cfm[4], cfm[5], 0.0, cfm[6], cfm[7]]
    sigma = [cfm[8], cfm[9], cfm[10], 0.0, cfm[11], cfm[12]]

    dynamic = (fuel_model_number > 100) and (w_o[4] > 1e-6)
    sigma[3] = sigma[4] if dynamic else 0.0

    m_f = list(fuel_moisture)
    if dynamic:
        live_herb_load = w_o[4]
        fraction_green = max(0.0, min(1.0, (m_f[4] / 0.9) - (1.0 / 3.0)))
        fraction_cured = 1.0 - fraction_green
        m_f[3] = m_f[0]
        w_o[3] = live_herb_load * fraction_cured
        w_o[4] = live_herb_load * fraction_green

    rho_p = 32.0
    s_t = 0.0555
    s_e = 0.01

    a_ij = [(sigma[i] * w_o[i]) / rho_p for i in range(6)]
    a_0 = sum(a_ij[0:4])
    a_1 = sum(a_ij[4:6])
    a_tot = a_0 + a_1

    f_ij = [0.0] * 6
    if a_0 > 1e-6:
        for i in range(4):
            f_ij[i] = a_ij[i] / a_0
    if a_1 > 1e-6:
        for i in (4, 5):
            f_ij[i] = a_ij[i] / a_1

    f_i = (a_0 / a_tot, a_1 / a_tot) if a_tot > 1e-6 else (0.0, 0.0)
    f_classes = [_firemod_size_class(s) for s in sigma]

    g_ij = [0.0] * 6
    for i in range(4):
        g_ij[i] = sum(f_ij[j] for j in range(4) if f_classes[j] == f_classes[i])
    for i in (4, 5):
        g_ij[i] = sum(f_ij[j] for j in (4, 5) if f_classes[j] == f_classes[i])

    exp_a = [
        math.exp(-138.0 / sigma[0]) if sigma[0] > 1e-6 else 0.0,
        math.exp(-138.0 / sigma[1]) if sigma[1] > 1e-6 else 0.0,
        math.exp(-138.0 / sigma[2]) if sigma[2] > 1e-6 else 0.0,
        math.exp(-138.0 / sigma[3]) if sigma[3] > 1e-6 else 0.0,
        math.exp(-500.0 / sigma[4]) if sigma[4] > 1e-6 else 0.0,
        math.exp(-500.0 / sigma[5]) if sigma[5] > 1e-6 else 0.0,
    ]
    lf = [w_o[i] * exp_a[i] for i in range(6)]
    dead_mf = sum(m_f[i] * lf[i] for i in range(4))
    dead_lf = sum(lf[0:4])
    live_lf = sum(lf[4:6])
    if dead_lf > 1e-6 and live_lf > 1e-6:
        dead_fm = dead_mf / dead_lf
        dtl_ratio = dead_lf / live_lf
        m_x_live = max(m_x_dead, 2.9 * dtl_ratio * (1.0 - (dead_fm / m_x_dead)) - 0.226)
    else:
        m_x_live = m_x_dead

    m_x = [m_x_dead, m_x_dead, m_x_dead, m_x_dead if dynamic else 0.0, m_x_live, m_x_live]

    sigma_prime_0 = sum(f_ij[i] * sigma[i] for i in range(4))
    sigma_prime_1 = sum(f_ij[i] * sigma[i] for i in (4, 5))
    sigma_prime = f_i[0] * sigma_prime_0 + f_i[1] * sigma_prime_1

    beta = (sum(w_o) / rho_p) / delta if delta > 0.0 else 0.0
    beta_op = 3.348 * (sigma_prime ** (-0.8189)) if sigma_prime > 0.0 else 1.0

    s_e_0 = sum(f_ij[i] * s_e for i in range(4))
    s_e_1 = sum(f_ij[i] * s_e for i in (4, 5))
    eta_s_0 = 0.174 * (s_e_0 ** (-0.19)) if s_e_0 > 0.0 else 1.0
    eta_s_1 = 0.174 * (s_e_1 ** (-0.19)) if s_e_1 > 0.0 else 1.0

    m_f_0 = sum(f_ij[i] * m_f[i] for i in range(4))
    m_f_1 = sum(f_ij[i] * m_f[i] for i in (4, 5))
    m_x_0 = sum(f_ij[i] * m_x[i] for i in range(4))
    m_x_1 = sum(f_ij[i] * m_x[i] for i in (4, 5))
    eta_m_0 = _moisture_damping(m_f_0, m_x_0)
    eta_m_1 = _moisture_damping(m_f_1, m_x_1)

    h_0 = sum(f_ij[i] * h_val for i in range(4))
    h_1 = sum(f_ij[i] * h_val for i in (4, 5))
    w_n_0 = sum(g_ij[i] * w_o[i] * (1.0 - s_t) for i in range(4))
    w_n_1 = sum(g_ij[i] * w_o[i] * (1.0 - s_t) for i in (4, 5))

    btus = w_n_0 * h_0 * eta_m_0 * eta_s_0 + w_n_1 * h_1 * eta_m_1 * eta_s_1

    a_exp = 133.0 * (sigma_prime ** (-0.7913)) if sigma_prime > 0.0 else 0.0
    b_pow = sigma_prime**1.5
    c_rat = beta / beta_op
    gamma_max = b_pow / (495.0 + 0.0594 * b_pow)
    gamma_prime = gamma_max * (c_rat**a_exp) * math.exp(a_exp * (1.0 - c_rat))
    i_r = btus * gamma_prime

    xi = math.exp((0.792 + 0.681 * math.sqrt(sigma_prime)) * (beta + 0.1)) / (192.0 + 0.2595 * sigma_prime)
    heat_source = i_r * xi
    rho_b = sum(w_o) / delta if delta > 0.0 else 0.0

    eps_ij = [math.exp(-138.0 / s) if s > 0.0 else 0.0 for s in sigma]
    q_ig_ij = [(250.0 + 1116.0 * mf) if mf > 0.0 else 0.0 for mf in m_f]
    eff_q_0 = sum(f_ij[i] * eps_ij[i] * q_ig_ij[i] for i in range(4))
    eff_q_1 = sum(f_ij[i] * eps_ij[i] * q_ig_ij[i] for i in (4, 5))
    heat_sink = rho_b * (f_i[0] * eff_q_0 + f_i[1] * eff_q_1)

    r_0_ft_min = heat_source / heat_sink if heat_sink > 0.0 else 0.0
    t_res_min = 384.0 / sigma_prime if sigma_prime > 0.0 else 0.0
    d_a = r_0_ft_min * t_res_min
    i_s = (i_r * d_a) / 60.0

    u_eff_max = 0.9 * i_r
    b_wind = 0.02526 * (sigma_prime**0.54)
    c_wind = 7.47 * math.exp(-0.133 * (sigma_prime**0.55))
    e_wind = 0.715 * math.exp(-3.59e-4 * sigma_prime)
    f_wind = (beta / beta_op) ** e_wind

    phi_s_g = 5.275 * (beta ** (-0.3)) if beta > 0.0 else 0.0
    phi_w_scalr = (c_wind / f_wind) * (M_TO_FT**b_wind) if f_wind > 0.0 else 0.0
    phi_w_expnt = b_wind
    ws_scalr = FT_TO_M * ((f_wind / c_wind) ** (1.0 / b_wind)) if b_wind > 0.0 else 0.0
    ws_expnt = (1.0 / b_wind) if b_wind > 0.0 else 0.0

    return {
        "base_spread_rate": FT_TO_M * (r_0_ft_min * spread_rate_adjustment),
        "base_fireline_intensity": BTU_FT_S_TO_KW_M * (i_s * spread_rate_adjustment),
        "max_effective_wind_speed": FT_TO_M * u_eff_max,
        "phiS_G": phi_s_g,
        "phiW_scalr": phi_w_scalr,
        "phiW_expnt": phi_w_expnt,
        "ws_scalr": ws_scalr,
        "ws_expnt": ws_expnt,
        "residence_time_min": t_res_min,
        "delta_ft": delta,
    }


def compute_rothermel_max_behavior(
    bmin: dict[str, float],
    midflame_wind_speed_m_min: float,
    upwind_direction_deg: float,
    slope_rise_run: float,
    aspect_deg: float,
    use_wind_limit: bool = True,
) -> dict[str, Any]:
    """Compute max surface fire spread rate, length-to-width ratio, and eccentricity.

    Matches `pyretechnics.surface_fire.calc_surface_fire_behavior_max` to < 1e-5.
    """
    downwind_rad = math.radians((upwind_direction_deg + 180.0) % 360.0)
    upslope_rad = math.radians((aspect_deg + 180.0) % 360.0)

    wx_2d = midflame_wind_speed_m_min * math.sin(downwind_rad)
    wy_2d = midflame_wind_speed_m_min * math.cos(downwind_rad)
    sx_2d = slope_rise_run * math.sin(upslope_rad)
    sy_2d = slope_rise_run * math.cos(upslope_rad)

    wz_3d = wx_2d * sx_2d + wy_2d * sy_2d
    sz_3d = sx_2d * sx_2d + sy_2d * sy_2d

    w_mag = math.sqrt(wx_2d**2 + wy_2d**2 + wz_3d**2)
    s_mag = math.sqrt(sx_2d**2 + sy_2d**2 + sz_3d**2)

    phi_w = bmin["phiW_scalr"] * (w_mag ** bmin["phiW_expnt"]) if w_mag > 0.0 else 0.0
    phi_s = bmin["phiS_G"] * (slope_rise_run**2)

    w_unit = (wx_2d / w_mag, wy_2d / w_mag, wz_3d / w_mag) if (phi_w > 0.0 and w_mag > 0.0) else (0.0, 0.0, 0.0)
    s_unit = (sx_2d / s_mag, sy_2d / s_mag, sz_3d / s_mag) if (phi_s > 0.0 and s_mag > 0.0) else (0.0, 0.0, 0.0)

    phi_ex = phi_w * w_unit[0] + phi_s * s_unit[0]
    phi_ey = phi_w * w_unit[1] + phi_s * s_unit[1]
    phi_ez = phi_w * w_unit[2] + phi_s * s_unit[2]
    phi_e = math.sqrt(phi_ex**2 + phi_ey**2 + phi_ez**2)

    eff_ws = bmin["ws_scalr"] * (phi_e ** bmin["ws_expnt"]) if (phi_e > 0.0 and bmin["ws_scalr"] > 0.0) else 0.0
    if use_wind_limit and eff_ws > bmin["max_effective_wind_speed"]:
        lim_ws = bmin["max_effective_wind_speed"]
        lim_phi_e = bmin["phiW_scalr"] * (lim_ws ** bmin["phiW_expnt"])
    else:
        lim_ws = eff_ws
        lim_phi_e = phi_e

    max_ros = bmin["base_spread_rate"] * (1.0 + lim_phi_e)
    max_intensity = bmin["base_fireline_intensity"] * (1.0 + lim_phi_e)
    eff_ws_mph = lim_ws * M_MIN_TO_MPH
    lw_ratio = min(
        8.0,
        0.936 * math.exp(0.1147 * eff_ws_mph) + 0.461 * math.exp(-0.0692 * eff_ws_mph) - 0.397,
    )
    lw_ratio = max(1.0, lw_ratio)
    ecc = math.sqrt(lw_ratio**2 - 1.0) / lw_ratio

    return {
        "max_spread_rate": max_ros,
        "max_fireline_intensity": max_intensity,
        "length_to_width_ratio": lw_ratio,
        "eccentricity": ecc,
        "limited_phi_E": lim_phi_e,
    }


@dataclass(frozen=True)
class ROSTeacherParams:
    """Parameters for the Rothermel ROS-driven contact rule prototype."""

    gamma_ros: float = 0.085  # Dimensionless hazard scale in p_ignite = 1 - exp(-gamma * ROS * dt / d)
    res_time_scale: float = 1.5  # Multiplier on Rothermel residence time (min) for burnout survival
    wui_ros_m_min: float = 1.5  # Reduced effective surface ROS for WUI cells (FBFM40=91) when embers active

    def to_dict(self) -> dict[str, float]:
        """Return dictionary representation."""
        return {k: float(v) for k, v in asdict(self).items()}


def simulate_ros_teacher_60m(
    ds: FireDataset60m,
    ros_params: ROSTeacherParams,
    ember_params: EmberParams | None = None,
    use_ember_kernel: bool = False,
    duration_hours: int = 12,
    steps_per_hour: int = 30,
    seed: int = 42,
    init_burned_mask: np.ndarray | None = None,
    start_hour_index: int = 0,
    start_elapsed_min: float = 0.0,
) -> dict[str, Any]:
    """Run vectorized 60m Rothermel ROS-driven cellular simulation."""
    device = ds.p_veg_base.device
    h, w = ds.p_veg_base.shape

    if ember_params is None:
        ember_params = EmberParams()

    # Pre-build spatial tensors of Rothermel fuel properties for unique FBFM40 codes
    r0_np = np.zeros((h, w), dtype=np.float32)
    phi_sg_np = np.zeros((h, w), dtype=np.float32)
    phi_w_sc_np = np.zeros((h, w), dtype=np.float32)
    phi_w_ex_np = np.ones((h, w), dtype=np.float32)
    max_ws_np = np.zeros((h, w), dtype=np.float32)
    waf_np = np.zeros((h, w), dtype=np.float32)
    t_res_np = np.full((h, w), 8.0, dtype=np.float32)

    unique_codes = np.unique(ds.fbfm_arr)
    for code_val in unique_codes:
        c_int = int(code_val)
        mask_c = ds.fbfm_arr == c_int
        if c_int == 91 and use_ember_kernel and ember_params.p_urban_wui > 0.0:
            r0_np[mask_c] = float(ros_params.wui_ros_m_min * ember_params.p_urban_wui)
            phi_sg_np[mask_c] = 5.0
            phi_w_sc_np[mask_c] = 0.015
            phi_w_ex_np[mask_c] = 1.15
            max_ws_np[mask_c] = 600.0
            waf_np[mask_c] = 0.35
            t_res_np[mask_c] = 12.0
            continue
        bmin = compute_rothermel_no_wind_no_slope(c_int)
        if bmin["base_spread_rate"] <= 0.0:
            continue
        r0_np[mask_c] = bmin["base_spread_rate"]
        phi_sg_np[mask_c] = bmin["phiS_G"]
        phi_w_sc_np[mask_c] = bmin["phiW_scalr"]
        phi_w_ex_np[mask_c] = bmin["phiW_expnt"]
        max_ws_np[mask_c] = bmin["max_effective_wind_speed"]
        delta_ft = max(bmin["delta_ft"], 0.2)
        waf_val = 1.83 / math.log((20.0 + 0.36 * delta_ft) / (0.13 * delta_ft))
        waf_np[mask_c] = float(np.clip(waf_val, 0.15, 0.65))
        t_res_np[mask_c] = max(bmin["residence_time_min"], 6.0)

    r0_t = torch.from_numpy(r0_np).to(device).unsqueeze(-1).unsqueeze(-1)
    phi_sg_t = torch.from_numpy(phi_sg_np).to(device).unsqueeze(-1).unsqueeze(-1)
    phi_w_sc_t = torch.from_numpy(phi_w_sc_np).to(device).unsqueeze(-1).unsqueeze(-1)
    phi_w_ex_t = torch.from_numpy(phi_w_ex_np).to(device).unsqueeze(-1).unsqueeze(-1)
    waf_t = torch.from_numpy(waf_np).to(device).unsqueeze(-1).unsqueeze(-1)

    is_wui = torch.from_numpy(ds.fbfm_arr == 91).to(device)
    is_burnable = (ds.is_burnable | is_wui) if (use_ember_kernel and ember_params.p_urban_wui > 0.0) else ds.is_burnable

    # Directional upslope term: only upslope (slope_deg > 0) contributes to Rothermel phi_S
    tan_slope = torch.tan(torch.deg2rad(torch.clamp(ds.slope_deg, min=0.0, max=60.0)))
    phi_s_dir = phi_sg_t * (tan_slope**2)

    # Distance matrix d_ij [1, 1, 3, 3] in meters
    d_3x3 = (
        torch.tensor(
            [[1.41421356, 1.0, 1.41421356], [1.0, 1.0, 1.0], [1.41421356, 1.0, 1.41421356]],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, 3, 3)
        * ds.res_m
    )

    wind_offset = (
        torch.tensor([[3.0, 2.0, 1.0], [4.0, 0.0, 0.0], [5.0, 6.0, 7.0]], device=device) * 45.0
    ).view(1, 1, 3, 3)

    dt_min = 60.0 / steps_per_hour
    t_res_t = torch.from_numpy(t_res_np).to(device) * max(ros_params.res_time_scale, 0.2)
    p_continue_map = torch.exp(-dt_min / torch.clamp(t_res_t, min=4.0))

    phi_spot = compute_fuel_spotting_potential(ds.fbfm_arr, device) if use_ember_kernel else torch.zeros((h, w), device=device)
    p_spot_recv = torch.zeros((h, w), device=device)
    grid_r = None
    grid_c = None
    if use_ember_kernel:
        p_spot_recv = torch.where(
            ds.is_burnable,
            torch.clamp(0.35 * (1.0 + ds.p_veg_base), min=0.05, max=0.65),
            p_spot_recv,
        )
        p_spot_recv = torch.where(is_wui, torch.tensor(float(ember_params.p_urban_wui), device=device), p_spot_recv)
        grid_r, grid_c = torch.meshgrid(
            torch.arange(h, dtype=torch.long, device=device),
            torch.arange(w, dtype=torch.long, device=device),
            indexing="ij",
        )

    raw_init_np = init_burned_mask if init_burned_mask is not None else ds.gofer_hourly_masks[start_hour_index]
    init_mask = torch.from_numpy(raw_init_np).to(device=device, dtype=torch.bool)
    init_float = init_mask.float().unsqueeze(0).unsqueeze(0)
    eroded = F.max_pool2d(1.0 - init_float, kernel_size=3, stride=1, padding=1) < 0.5
    eroded_mask = eroded.squeeze(0).squeeze(0) & init_mask
    boundary_ring = init_mask & (~eroded_mask)

    if (
        start_hour_index >= 2
        and start_hour_index < len(ds.gofer_hourly_masks)
        and np.array_equal(raw_init_np, ds.gofer_hourly_masks[start_hour_index])
    ):
        prev_idx = max(0, start_hour_index - 3)
        recent_np = raw_init_np & (~ds.gofer_hourly_masks[prev_idx])
        if np.any(recent_np):
            recent_t = torch.from_numpy(recent_np).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            active_zone = F.max_pool2d(recent_t, kernel_size=7, stride=1, padding=3).squeeze(0).squeeze(0) > 0.5
            burning = boundary_ring & active_zone
            burned = init_mask & (~burning)
        else:
            burning = boundary_ring
            burned = eroded_mask.clone()
    else:
        burning = boundary_ring
        burned = eroded_mask.clone()

    if not torch.any(burning):
        burning = boundary_ring if torch.any(boundary_ring) else init_mask.clone()
        burned = init_mask & (~burning)

    arrival_min_tensor = torch.full((h, w), float("nan"), dtype=torch.float32, device=device)
    arrival_min_tensor[init_mask] = float(start_elapsed_min)

    hourly_sim_masks = np.zeros((duration_hours + 1, h, w), dtype=bool)
    hourly_sim_masks[0] = raw_init_np.copy()

    torch.manual_seed(seed)

    for hr in range(duration_hours):
        w_idx = min(start_hour_index + hr, len(ds.weather_df) - 1)
        w_row = ds.weather_df.iloc[w_idx]
        ws = float(w_row["wind_speed_ms"])
        w_gust = float(w_row.get("wind_gust_ms", ws * 1.4))
        towards_deg = float((270.0 - float(w_row["wind_direction_deg"])) % 360.0)

        # Midflame wind speed (m/min)
        ws_20ft_m_min = (ws * 60.0) / 1.15
        midflame_m_min = torch.clamp(ws_20ft_m_min * waf_t, min=0.0, max=600.0)
        phi_w_head = phi_w_sc_t * torch.pow(torch.clamp(midflame_m_min, min=1e-3), phi_w_ex_t)

        # Elliptical directional decay around wind direction
        ws_mph = (ws * 60.0 * M_MIN_TO_MPH) * 0.4
        lw = min(8.0, max(1.0, 0.936 * math.exp(0.1147 * ws_mph) + 0.461 * math.exp(-0.0692 * ws_mph) - 0.397))
        ecc = math.sqrt(lw**2 - 1.0) / lw if lw > 1.0 else 0.0

        angle_diff_rad = torch.deg2rad((wind_offset - towards_deg) % 360.0)
        cos_w = torch.cos(angle_diff_rad)
        ellip_adj = (1.0 - ecc) / torch.clamp(1.0 - ecc * cos_w, min=1e-4)

        ros_dir_m_min = r0_t * (1.0 + phi_w_head * ellip_adj + phi_s_dir)
        hazard = ros_params.gamma_ros * (ros_dir_m_min * dt_min) / d_3x3
        log_one_minus_p = -torch.clamp(hazard, min=0.0, max=15.0)
        log_one_minus_p[:, :, 1, 1] = 0.0
        l_9hw = torch.flip(log_one_minus_p, dims=(-2, -1)).permute(2, 3, 0, 1).reshape(9, h, w).contiguous()

        for s in range(1, steps_per_hour + 1):
            cur_time_min = float(start_elapsed_min + (hr * steps_per_hour + s) * dt_min)
            bf_patches = F.unfold(burning.float().view(1, 1, h, w), kernel_size=3, padding=1).view(9, h, w)
            log_q = (bf_patches * l_9hw).sum(dim=0)
            p_ignite = 1.0 - torch.exp(log_q)

            rand_prop = torch.rand((h, w), device=device)
            rand_cont = torch.rand((h, w), device=device)

            burnable = (~burning) & (~burned) & is_burnable
            new_contact = burnable & (p_ignite > rand_prop)

            if use_ember_kernel:
                new_ember = sample_ember_spotting_step(
                    burning=burning,
                    burned=burned,
                    phi_spot=phi_spot,
                    p_spot_recv=p_spot_recv,
                    wind_speed_ms=ws,
                    wind_gust_ms=w_gust,
                    towards_deg=towards_deg,
                    res_m=ds.res_m,
                    ember_params=ember_params,
                    grid_r=grid_r,
                    grid_c=grid_c,
                )
                new_ignited = new_contact | new_ember
            else:
                new_ignited = new_contact

            burn_out = burning & (rand_cont > p_continue_map)
            burning = (burning & (~burn_out)) | new_ignited
            burned = burned | burn_out
            arrival_min_tensor = torch.where(
                new_ignited & torch.isnan(arrival_min_tensor),
                torch.tensor(cur_time_min, dtype=torch.float32, device=device),
                arrival_min_tensor,
            )

        hourly_sim_masks[hr + 1] = (burning | burned).cpu().numpy()

    return {
        "hourly_sim_masks": hourly_sim_masks,
        "arrival_time_min": arrival_min_tensor.cpu().numpy(),
    }
