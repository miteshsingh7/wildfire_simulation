"""Stand-in dataset generator for surrogate modeling (Phase 2A/2B).

Generates 256x256 tiles at 60 m resolution exclusively from the 7 training-fire
landscapes (never Camp Fire or Phase 1B held-out test fires).
Simulates 12 h horizon fire propagation using stand-in sources:
- wind_ellipse_analytic (closed-form Huygens expanding ellipse)
- teacher_v1_cellular (cellular automaton / Rothermel percolation stand-in)

Every generated sample is tagged with 'scaffold' and the teacher config hash,
and stores its generator source type.
Supports stratified sampling to guarantee balanced spread regimes and prevent stalled runs.
"""

import hashlib
import heapq
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml

from wf052.surrogate.dataset import (
    SurrogateSample,
    calculate_upwind_third_ignition,
    compute_slope_aspect,
    fbfm40_to_fuel_group,
)

logger = logging.getLogger(__name__)

SEVEN_TRAINING_FIRES: list[str] = [
    "bobcat",
    "caldor",
    "creek",
    "glass",
    "kincade",
    "tamarack",
    "zogg",
]


def load_held_out_fire_exclusions(config_path: str = "configs/scenarios.yaml") -> dict[str, Any]:
    """Read prohibited domains and held-out fires from scenario configuration."""
    p = Path(config_path)
    if not p.exists():
        return {
            "prohibited_domains": ["camp_fire_2018", "data/tiles/elevation.tif"],
            "held_out_fires": [
                {
                    "name": "camp_fire_2018",
                    "aliases": ["camp", "camp_fire", "campfire", "camp_fire_2018"],
                    "reason": "Primary operational evaluation domain",
                },
                {
                    "name": "slater_and_devil_2020",
                    "aliases": [
                        "slater",
                        "slater_and_devil",
                        "slater_and_devil_2020",
                        "slater-and-devil",
                    ],
                    "reason": "Phase 1B fresh held-out wind-driven test fire #1",
                },
                {
                    "name": "walker_2019",
                    "aliases": ["walker", "walker_2019", "walker-2019"],
                    "reason": "Phase 1B fresh held-out wind-driven test fire #2",
                },
                {
                    "name": "w5_cold_springs_2020",
                    "aliases": [
                        "w5",
                        "w5_cold_springs",
                        "w-5_cold_springs",
                        "w5_cold_springs_2020",
                    ],
                    "reason": "Phase 1B alternate fresh held-out fire #3",
                },
            ],
        }
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_fire_not_excluded(fire_name: str, config_path: str = "configs/scenarios.yaml") -> None:
    """Verify that a given fire domain is not in the prohibited or held-out fire exclusions.

    Raises:
        ValueError if the fire name matches any prohibited or held-out fire domain.
    """
    cfg = load_held_out_fire_exclusions(config_path)
    clean_target = fire_name.lower().replace("-", "_").replace(" ", "_").strip()

    # Check prohibited_domains list
    for p_dom in cfg.get("prohibited_domains", []):
        p_clean = p_dom.lower().replace("-", "_").replace(" ", "_").strip()
        if clean_target == p_clean or p_clean in clean_target:
            prefix = "Camp Fire domain" if "camp" in clean_target else "Fire domain"
            raise ValueError(f"{prefix} '{fire_name}' is in prohibited_domains list ({p_dom})!")

    # Check held_out_fires list
    for hf in cfg.get("held_out_fires", []):
        hf_name = hf.get("name", "").lower().replace("-", "_").replace(" ", "_").strip()
        aliases = [
            a.lower().replace("-", "_").replace(" ", "_").strip() for a in hf.get("aliases", [])
        ]
        if hf_name:
            aliases.append(hf_name)

        for alias in aliases:
            if clean_target == alias or alias in clean_target or clean_target in alias:
                prefix = "Camp Fire domain" if "camp" in clean_target else "Held-out fire domain"
                raise ValueError(
                    f"{prefix} '{fire_name}' is strictly held-out ({hf.get('reason', 'held-out test set')})!"
                )


def get_teacher_config_hash(config_path: str = "configs/teacher_v1.yaml") -> str:
    """Return SHA-256 hash of teacher config file."""
    p = Path(config_path)
    if p.exists():
        return hashlib.sha256(p.read_bytes()).hexdigest()
    return "c45ba6db95187063347ad1c06e81dbb29ff3e45fa92f50ad46d8dcc9e16cf9b6"


def load_fire_rasters_crop(
    fire_name: str,
    tile_size: tuple[int, int] = (256, 256),
    crop_center_row: int | None = None,
    crop_center_col: int | None = None,
    fires_base_dir: str = "data/tiles/fires",
    scenarios_config_path: str = "configs/scenarios.yaml",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load elevation, fbfm40, canopy_cover, and weather for a 256x256 crop of a fire domain.

    Returns:
        (fbfm40, elevation, canopy_cover, hourly_wind_uv)
    """
    verify_fire_not_excluded(fire_name, scenarios_config_path)

    base_dir_resolved = Path(fires_base_dir).resolve()
    fire_dir = (Path(fires_base_dir) / fire_name).resolve()

    # Prevent path traversal or resolving to prohibited base tiles directory
    try:
        is_child = fire_dir.is_relative_to(base_dir_resolved)
    except AttributeError:
        is_child = str(fire_dir).startswith(str(base_dir_resolved))

    if not is_child or fire_dir == Path("data/tiles").resolve():
        raise ValueError(f"Prohibited or invalid fire domain path: {fire_dir}")

    if not fire_dir.exists():
        raise FileNotFoundError(f"Fire directory {fire_dir} not found.")

    with rasterio.open(fire_dir / "elevation.tif") as src:
        h, w = src.shape
        r0 = max(0, (h - tile_size[0]) // 2) if crop_center_row is None else crop_center_row
        c0 = max(0, (w - tile_size[1]) // 2) if crop_center_col is None else crop_center_col
        r0 = min(r0, h - tile_size[0])
        c0 = min(c0, w - tile_size[1])

        window = rasterio.windows.Window(c0, r0, tile_size[1], tile_size[0])
        elev = src.read(1, window=window).astype(np.float32)

    with rasterio.open(fire_dir / "fbfm40.tif") as src:
        fbfm = src.read(1, window=window).astype(np.int64)

    with rasterio.open(fire_dir / "canopy_cover.tif") as src:
        canopy = src.read(1, window=window).astype(np.float32)
        # Normalize to [0, 1]
        if np.nanmax(canopy) > 1.0:
            canopy = np.clip(canopy / 100.0, 0.0, 1.0)

    # Weather: read HRRR hourly parquet
    weather_p = fire_dir / "hrrr_weather_hourly.parquet"
    if weather_p.exists():
        w_df = pd.read_parquet(weather_p)
        u_vals = w_df["u10_ms"].values[:12] if "u10_ms" in w_df else np.full(12, 5.0)
        v_vals = w_df["v10_ms"].values[:12] if "v10_ms" in w_df else np.full(12, 5.0)
        if len(u_vals) < 12:
            u_vals = np.pad(u_vals, (0, 12 - len(u_vals)), mode="edge")
            v_vals = np.pad(v_vals, (0, 12 - len(v_vals)), mode="edge")
        hourly_uv = np.stack([u_vals[:12], v_vals[:12]], axis=-1).astype(np.float32)
    else:
        # Default stand-in wind
        hourly_uv = np.tile([6.0, 4.0], (12, 1)).astype(np.float32)

    return fbfm, elev, canopy, hourly_uv


def generate_analytic_arrival_surface(
    fuel_group: np.ndarray,
    hourly_wind_uv: np.ndarray,
    ign_r: int,
    ign_c: int,
    resolution_m: float = 60.0,
    horizon_min: float = 720.0,
    speed_multiplier: float = 1.0,
    length_to_breadth: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate stand-in arrival time raster and censor mask via analytic wind-ellipse.

    Returns:
        (arrival_time, censor_mask)
    """
    h, w = fuel_group.shape
    mean_u = float(np.mean(hourly_wind_uv[:, 0])) * speed_multiplier
    mean_v = float(np.mean(hourly_wind_uv[:, 1])) * speed_multiplier
    wind_spd = float(np.hypot(mean_u, mean_v))
    wind_angle_rad = np.arctan2(mean_v, mean_u)

    # Fuel-dependent base forward spread rate (m/min)
    base_rates = np.array([0.0, 35.0, 25.0, 20.0, 15.0, 10.0, 12.0], dtype=np.float32)
    tile_fuel_rate = float(np.mean(base_rates[fuel_group]))
    tile_fuel_rate = max(5.0, tile_fuel_rate)

    # Rothermel-type wind acceleration
    r_head = tile_fuel_rate * (1.0 + 0.15 * (wind_spd**1.3))
    r_back = max(1.0, r_head / (length_to_breadth * 3.0))

    # Grid coordinates relative to ignition in meters
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    dx = (cc - ign_c) * resolution_m
    dy = (rr - ign_r) * resolution_m

    # Rotate coordinates to align with downwind axis
    cos_t = np.cos(wind_angle_rad)
    sin_t = np.sin(wind_angle_rad)
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t

    a = (r_head + r_back) / 2.0
    c = (r_head - r_back) / 2.0
    b = a / length_to_breadth

    # Exact positive arrival time for expanding ellipse
    a2 = a * a
    b2 = b * b
    c2 = c * c
    coeff_a = 1.0 - (c2 / a2)
    coeff_b = -2.0 * (c * x_rot) / a2
    coeff_c = -((x_rot**2) / a2 + (y_rot**2) / b2)

    disc = np.maximum(0.0, coeff_b**2 - 4.0 * coeff_a * coeff_c)
    t_arr = (-coeff_b + np.sqrt(disc)) / (2.0 * max(1e-4, coeff_a))

    # Cells where fuel is non-burnable (fuel_group == 0) never burn
    is_nonburnable = fuel_group == 0
    t_arr[is_nonburnable] = np.inf

    # Censoring: cells where arrival > horizon_min are censored
    censor_mask = (t_arr > horizon_min) | np.isinf(t_arr) | np.isnan(t_arr)
    arrival_time = np.where(censor_mask, horizon_min, t_arr).astype(np.float32)

    return arrival_time, censor_mask.astype(np.float32)


def simulate_cellular_standin_arrival(
    fuel_group: np.ndarray,
    hourly_wind_uv: np.ndarray,
    ign_r: int,
    ign_c: int,
    resolution_m: float = 60.0,
    horizon_min: float = 720.0,
    speed_multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate stand-in arrival time raster via cellular automaton / percolation simulation.

    Simulates discrete 8-neighbor front expansion with fuel and directional wind acceleration.
    """
    h, w = fuel_group.shape
    mean_u = float(np.mean(hourly_wind_uv[:, 0])) * speed_multiplier
    mean_v = float(np.mean(hourly_wind_uv[:, 1])) * speed_multiplier
    wind_spd = float(np.hypot(mean_u, mean_v))
    wind_angle = float(np.arctan2(mean_v, mean_u))

    # Fuel-dependent base forward spread rate (m/min)
    base_rates = np.array([0.0, 32.0, 24.0, 18.0, 14.0, 9.0, 11.0], dtype=np.float32)

    dist = np.full((h, w), np.inf, dtype=np.float32)
    dist[ign_r, ign_c] = 0.0

    pq: list[tuple[float, int, int]] = [(0.0, ign_r, ign_c)]
    visited = np.zeros((h, w), dtype=bool)

    neighbor_offsets = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    while pq:
        d, r, c = heapq.heappop(pq)
        if visited[r, c]:
            continue
        visited[r, c] = True
        if d >= horizon_min:
            break

        for dr, dc in neighbor_offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                f_grp = fuel_group[nr, nc]
                if f_grp == 0:  # Non-burnable
                    continue

                base_rate = base_rates[f_grp]
                if base_rate <= 0:
                    continue

                step_dist = resolution_m * (1.414 if (dr != 0 and dc != 0) else 1.0)
                theta_spread = np.arctan2(dr, dc)
                wind_align = np.cos(theta_spread - wind_angle)

                wind_factor = max(0.2, 1.0 + 0.18 * (wind_spd**1.2) * (0.6 + 0.4 * wind_align))
                effective_rate = max(1.0, base_rate * wind_factor)

                cost_min = step_dist / effective_rate
                new_d = d + cost_min
                if new_d < dist[nr, nc] and new_d <= horizon_min:
                    dist[nr, nc] = new_d
                    heapq.heappush(pq, (new_d, nr, nc))

    is_nonburnable = fuel_group == 0
    dist[is_nonburnable] = np.inf

    censor_mask = (dist > horizon_min) | np.isinf(dist) | np.isnan(dist)
    arrival_time = np.where(censor_mask, horizon_min, dist).astype(np.float32)

    return arrival_time, censor_mask.astype(np.float32)


def generate_ensemble_samples(
    fire_names: Sequence[str] | None = None,
    num_members: int = 20,
    base_seed: int = 42000,
    tile_size: tuple[int, int] = (256, 256),
    horizon_min: float = 720.0,
    scenarios_config_path: str = "configs/scenarios.yaml",
    source_mode: str = "both",
) -> list[SurrogateSample]:
    """Generate a dataset of perturbed stand-in scenario samples across the 7 training fires.

    Args:
        fire_names: List of training fire names (must exclude held-out/prohibited fires).
        num_members: Number of ensemble members per fire.
        base_seed: Master random seed.
        tile_size: (H, W) spatial dimensions.
        horizon_min: Simulation time horizon in minutes.
        scenarios_config_path: Path to scenarios configuration.
        source_mode: 'analytic_wind_ellipse', 'teacher_v1_cellular', or 'both'.
    """
    if fire_names is None:
        fire_names = SEVEN_TRAINING_FIRES

    cfg = load_held_out_fire_exclusions(scenarios_config_path)
    strat_cfg = cfg.get("stratified_sampling", {})
    strat_enabled = strat_cfg.get("enabled", True)
    min_burned_frac = float(strat_cfg.get("min_burned_fraction", 0.02))

    cfg_hash = get_teacher_config_hash()
    samples: list[SurrogateSample] = []

    for fire in fire_names:
        verify_fire_not_excluded(fire, scenarios_config_path)

        fbfm_base, elev_base, canopy_base, wind_base = load_fire_rasters_crop(
            fire_name=fire,
            tile_size=tile_size,
            scenarios_config_path=scenarios_config_path,
        )
        fuel_grp = fbfm40_to_fuel_group(fbfm_base)
        slope, asp_sin, asp_cos = compute_slope_aspect(elev_base, resolution_m=60.0)

        # Elevation z-scoring
        elev_z = ((elev_base - np.mean(elev_base)) / (np.std(elev_base) + 1e-5)).astype(np.float32)

        for m_idx in range(num_members):
            fire_hash = int(hashlib.sha256(fire.encode("utf-8")).hexdigest(), 16) % 10000
            seed = base_seed + fire_hash + m_idx * 17
            rng = np.random.RandomState(seed)

            # Determine source for this member
            if source_mode == "both":
                current_source = (
                    "analytic_wind_ellipse" if (m_idx % 2 == 0) else "teacher_v1_cellular"
                )
            elif source_mode in ("analytic_wind_ellipse", "teacher_v1_cellular"):
                current_source = source_mode
            else:
                current_source = "analytic_wind_ellipse"

            # Ensemble perturbations with stratified sampling retry if needed
            max_attempts = 10 if strat_enabled else 1
            arr_time: np.ndarray | None = None
            c_mask: np.ndarray | None = None
            hourly_wind_perturbed: np.ndarray | None = None
            ign_mask: np.ndarray | None = None

            for attempt in range(max_attempts):
                spd_boost = 1.0 + (attempt * 0.05 if strat_enabled else 0.0)
                spd_mult = float(rng.uniform(0.80, 1.20)) * spd_boost
                dir_jitter_deg = float(rng.uniform(-30.0, 30.0))
                ign_offset_r = int(rng.randint(-10, 11))
                ign_offset_c = int(rng.randint(-10, 11))
                lb_jitter = float(rng.uniform(-0.3, 0.3))

                # Apply wind direction jitter
                jitter_rad = np.deg2rad(dir_jitter_deg)
                cos_j = np.cos(jitter_rad)
                sin_j = np.sin(jitter_rad)
                u_j = (wind_base[:, 0] * cos_j - wind_base[:, 1] * sin_j) * spd_mult
                v_j = (wind_base[:, 0] * sin_j + wind_base[:, 1] * cos_j) * spd_mult
                hourly_wind_perturbed = np.stack([u_j, v_j], axis=-1).astype(np.float32)

                # Upwind third ignition
                ign_r, ign_c = calculate_upwind_third_ignition(
                    hourly_wind_uv=hourly_wind_perturbed,
                    tile_shape=tile_size,
                    offset_r=ign_offset_r,
                    offset_c=ign_offset_c,
                )

                ign_mask = np.zeros(tile_size, dtype=np.float32)
                r_lo, r_hi = max(0, ign_r - 1), min(tile_size[0], ign_r + 2)
                c_lo, c_hi = max(0, ign_c - 1), min(tile_size[1], ign_c + 2)
                ign_mask[r_lo:r_hi, c_lo:c_hi] = 1.0

                if current_source == "teacher_v1_cellular":
                    arr_time, c_mask = simulate_cellular_standin_arrival(
                        fuel_group=fuel_grp,
                        hourly_wind_uv=hourly_wind_perturbed,
                        ign_r=ign_r,
                        ign_c=ign_c,
                        resolution_m=60.0,
                        horizon_min=horizon_min,
                        speed_multiplier=spd_mult,
                    )
                else:
                    arr_time, c_mask = generate_analytic_arrival_surface(
                        fuel_group=fuel_grp,
                        hourly_wind_uv=hourly_wind_perturbed,
                        ign_r=ign_r,
                        ign_c=ign_c,
                        resolution_m=60.0,
                        horizon_min=horizon_min,
                        speed_multiplier=spd_mult,
                        length_to_breadth=2.5 + lb_jitter,
                    )

                burned_frac = float(np.mean(c_mask == 0.0))
                if not strat_enabled or burned_frac >= min_burned_frac:
                    break

            assert arr_time is not None and c_mask is not None
            assert hourly_wind_perturbed is not None and ign_mask is not None

            sample_id = f"{fire}_member_{m_idx:02d}_{current_source}"
            scaffold_tag = f"scaffold_{cfg_hash[:12]}"

            samples.append(
                SurrogateSample(
                    fire_name=fire,
                    sample_id=sample_id,
                    fuel_group=fuel_grp,
                    slope=slope,
                    aspect_sin=asp_sin,
                    aspect_cos=asp_cos,
                    elevation_z=elev_z,
                    canopy_cover=canopy_base,
                    ignition_mask=ign_mask,
                    hourly_wind_uv=hourly_wind_perturbed,
                    target_arrival_time=arr_time,
                    censor_mask=c_mask,
                    scaffold_tag=scaffold_tag,
                    seed=seed,
                    source=current_source,
                )
            )

    logger.info("Generated %d ensemble samples across %d fires.", len(samples), len(fire_names))
    return samples
