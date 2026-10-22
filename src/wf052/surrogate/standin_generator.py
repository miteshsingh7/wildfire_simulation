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
from dataclasses import dataclass
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

    # Weather: read HRRR hourly parquet with synthetic scaling
    weather_p = fire_dir / "hrrr_weather_hourly.parquet"
    if weather_p.exists():
        hourly_uv = sample_hrrr_wind_series(fire_dir, rng=None, window_hours=12)
    else:
        # Default stand-in wind
        hourly_uv = np.tile([6.0, 4.0], (12, 1)).astype(np.float32)

    return fbfm, elev, canopy, hourly_uv


def sample_hrrr_wind_series(
    fire_dir: Path,
    rng: np.random.RandomState | None = None,
    window_hours: int = 12,
    target_peak_speed_range: tuple[float, float] = (1.0, 20.0),
) -> np.ndarray:
    """Sample a 12-hour wind series from HRRR weather parquet with synthetic peak wind scaling.

    1. Reads hrrr_weather_hourly.parquet for the fire (containing 25 hourly observations).
    2. Randomly slices a 12-hour window.
    3. Synthetically scales wind vectors so peak sustained wind covers target_peak_speed_range ([1, 20] m/s).

    Returns:
        hourly_uv: (12, 2) float32 array of (u, v) wind vectors in m/s.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    weather_p = fire_dir / "hrrr_weather_hourly.parquet"
    if weather_p.exists():
        w_df = pd.read_parquet(weather_p)
        u_all = w_df["u10_ms"].values if "u10_ms" in w_df else np.full(24, 4.0)
        v_all = w_df["v10_ms"].values if "v10_ms" in w_df else np.full(24, 3.0)
        n_obs = len(u_all)
        if n_obs >= window_hours:
            start_idx = int(rng.randint(0, n_obs - window_hours + 1))
            u_win = u_all[start_idx : start_idx + window_hours].astype(np.float32)
            v_win = v_all[start_idx : start_idx + window_hours].astype(np.float32)
        else:
            u_win = np.pad(u_all, (0, max(0, window_hours - n_obs)), mode="edge")[:window_hours].astype(np.float32)
            v_win = np.pad(v_all, (0, max(0, window_hours - n_obs)), mode="edge")[:window_hours].astype(np.float32)
    else:
        # Default fallback
        u_win = np.full(window_hours, 5.0, dtype=np.float32)
        v_win = np.full(window_hours, 3.0, dtype=np.float32)

    # Base peak sustained wind speed across the 12-hour window
    base_speeds = np.hypot(u_win, v_win)
    base_peak = float(np.max(base_speeds))

    # Synthetic scaling to cover [1.0, 20.0] m/s peak sustained wind
    target_peak = float(rng.uniform(target_peak_speed_range[0], target_peak_speed_range[1]))
    scale = (target_peak / base_peak) if base_peak > 1e-4 else 1.0

    u_scaled = (u_win * scale).astype(np.float32)
    v_scaled = (v_win * scale).astype(np.float32)

    return np.stack([u_scaled, v_scaled], axis=-1).astype(np.float32)


def assign_burn_fraction_bin(bf: float) -> int:
    """Assign burned fraction in [0, 1] to one of 4 stratification bins:
    0: [0.0, 0.2)
    1: [0.2, 0.5)
    2: [0.5, 0.9)
    3: [0.9, 1.0]
    """
    if bf < 0.2:
        return 0
    if bf < 0.5:
        return 1
    if bf < 0.9:
        return 2
    return 3


def compute_stratified_histogram(
    burned_fractions: Sequence[float],
) -> dict[str, Any]:
    """Compute counts and percentages across the 4 burned fraction bins:
    - [0.0, 0.2)
    - [0.2, 0.5)
    - [0.5, 0.9)
    - [0.9, 1.0]
    """
    labels = ["0-0.2", "0.2-0.5", "0.5-0.9", ">0.9"]
    bins = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.9), (0.9, 1.0)]
    counts = [0, 0, 0, 0]
    for bf in burned_fractions:
        counts[assign_burn_fraction_bin(float(bf))] += 1
    total = len(burned_fractions)
    pcts = [(c / max(1, total)) * 100.0 for c in counts]
    return {
        "labels": labels,
        "bins": bins,
        "counts": counts,
        "percentages": pcts,
        "total": total,
    }


def stratify_samples_by_burned_fraction(
    samples: Sequence[SurrogateSample],
    target_proportions: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    total_samples: int | None = None,
    seed: int = 42,
) -> tuple[list[SurrogateSample], dict[str, Any], dict[str, Any]]:
    """Subsample / stratify a pool of samples to match target burned fraction bin proportions.

    Bins:
    - [0.0, 0.2)
    - [0.2, 0.5)
    - [0.5, 0.9)
    - [0.9, 1.0]

    Returns:
        (stratified_samples, hist_before, hist_after)
    """
    rng = np.random.RandomState(seed)

    bin_buckets: list[list[SurrogateSample]] = [[], [], [], []]
    bf_all: list[float] = []
    for s in samples:
        bf = float(np.mean(s.censor_mask < 0.5))
        bf_all.append(bf)
        bin_buckets[assign_burn_fraction_bin(bf)].append(s)

    hist_before = compute_stratified_histogram(bf_all)

    if total_samples is None:
        total_samples = len(samples)

    norm_props = np.array(target_proportions, dtype=np.float64)
    norm_props = norm_props / np.sum(norm_props)

    stratified: list[SurrogateSample] = []
    for bin_idx in range(4):
        target_k = int(round(total_samples * norm_props[bin_idx]))
        pool = bin_buckets[bin_idx]
        if not pool:
            continue
        if len(pool) >= target_k:
            chosen_indices = rng.choice(len(pool), size=target_k, replace=False)
            stratified.extend([pool[i] for i in chosen_indices])
        else:
            stratified.extend(pool)
            if target_k > len(pool):
                extra_indices = rng.choice(len(pool), size=target_k - len(pool), replace=True)
                stratified.extend([pool[i] for i in extra_indices])

    bf_after = [float(np.mean(s.censor_mask < 0.5)) for s in stratified]
    hist_after = compute_stratified_histogram(bf_after)

    return stratified, hist_before, hist_after


def format_stratification_report(hist_before: dict[str, Any], hist_after: dict[str, Any]) -> str:
    """Format before and after stratification histograms as a readable markdown table."""
    lines = [
        "| Burned Fraction Bin | Before Count | Before % | After Count | After % |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for lbl, b_cnt, b_pct, a_cnt, a_pct in zip(
        hist_before["labels"],
        hist_before["counts"],
        hist_before["percentages"],
        hist_after["counts"],
        hist_after["percentages"],
    ):
        lines.append(f"| `{lbl}` | {b_cnt} | {b_pct:.1f}% | {a_cnt} | {a_pct:.1f}% |")
    return "\n".join(lines)


@dataclass(frozen=True)
class BinRejectionStats:
    """Statistics for a single burned-fraction bin during rejection sampling."""

    bin_name: str
    bin_range: tuple[float, float]
    target: int
    generated: int
    rejected: int
    unique_count: int
    filled: bool


@dataclass(frozen=True)
class RejectionSamplingReport:
    """Comprehensive report across all bins from rejection sampling."""

    stats: list[BinRejectionStats]
    total_generated: int
    total_accepted: int
    total_rejected: int
    max_attempts: int
    cap_reached: bool
    unfilled_messages: list[str]

    def to_markdown(self) -> str:
        """Format report as a clean markdown table."""
        lines = [
            "| Bin | Range | Target Quota | Generated | Rejected | Unique Count | Status |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for s in self.stats:
            status_str = "**Filled**" if s.filled else f"**Unfilled** (deficit: {s.target - s.unique_count})"
            lines.append(
                f"| `{s.bin_name}` | `[{s.bin_range[0]}, {s.bin_range[1]})` | {s.target} | {s.generated} | {s.rejected} | {s.unique_count} | {status_str} |"
            )
        if self.unfilled_messages:
            lines.append("")
            for msg in self.unfilled_messages:
                lines.append(f"> [!WARNING] {msg}")
        return "\n".join(lines)


def sample_to_metadata_dict(sample: SurrogateSample, shard_id: str) -> dict[str, Any]:
    """Extract CSV metadata row for a single sample.

    Fields:
    - fire: fire name
    - source: generator source (e.g. teacher_v1_cellular)
    - seed: random seed
    - peak_wind: peak sustained wind speed in m/s across the 12h series
    - burned_fraction: fraction of cells where arrival < inf (censor_mask < 0.5)
    - censored_fraction: fraction of cells unburned (censor_mask >= 0.5)
    - bin: stratification bin index in 0..3
    - shard_id: identifier of the enclosing shard
    - sample_id: unique sample ID
    """
    speeds = np.hypot(sample.hourly_wind_uv[:, 0], sample.hourly_wind_uv[:, 1])
    peak_wind = float(np.max(speeds))
    burned_frac = float(np.mean(sample.censor_mask < 0.5))
    censored_frac = float(np.mean(sample.censor_mask >= 0.5))
    bin_idx = assign_burn_fraction_bin(burned_frac)

    return {
        "fire": sample.fire_name,
        "source": sample.source,
        "seed": int(sample.seed),
        "peak_wind": round(peak_wind, 3),
        "burned_fraction": round(burned_frac, 4),
        "censored_fraction": round(censored_frac, 4),
        "bin": bin_idx,
        "shard_id": shard_id,
        "sample_id": sample.sample_id,
    }


def save_shard_with_metadata(
    samples: Sequence[SurrogateSample],
    shard_id: str,
    shards_dir: Path,
) -> pd.DataFrame:
    """Save sample objects to a pickle file and metadata to a CSV file.

    Creates:
    - shards_dir / f"{shard_id}.pkl"
    - shards_dir / f"{shard_id}_metadata.csv"

    Returns:
    - pd.DataFrame containing the metadata rows.
    """
    import pickle

    shards_dir = Path(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = shards_dir / f"{shard_id}.pkl"
    tmp_pkl = shards_dir / f"tmp_{shard_id}.pkl"
    with open(tmp_pkl, "wb") as f:
        pickle.dump(list(samples), f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_pkl.rename(pkl_path)

    rows = [sample_to_metadata_dict(s, shard_id) for s in samples]
    df = pd.DataFrame(rows)
    csv_path = shards_dir / f"{shard_id}_metadata.csv"
    df.to_csv(csv_path, index=False)

    return df


def compute_wind_distribution_table(
    df_meta: pd.DataFrame,
    speed_bins: Sequence[tuple[float, float]] = (
        (1.0, 5.0),
        (5.0, 10.0),
        (10.0, 15.0),
        (15.0, 20.0),
    ),
) -> str:
    """Compute wind speed distribution table from metadata CSV DataFrame."""
    total = len(df_meta)
    lines = [
        "| Wind Speed Range (m/s) | Sample Count | Percentage | Mean Speed (m/s) | Min Speed (m/s) | Max Speed (m/s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    if total == 0:
        return "\n".join(lines)

    for lo, hi in speed_bins:
        sub = df_meta[
            (df_meta["peak_wind"] >= lo)
            & (df_meta["peak_wind"] < hi if hi < 20.0 else df_meta["peak_wind"] <= hi)
        ]
        cnt = len(sub)
        pct = (cnt / total) * 100.0 if total > 0 else 0.0
        mean_spd = float(sub["peak_wind"].mean()) if cnt > 0 else 0.0
        min_spd = float(sub["peak_wind"].min()) if cnt > 0 else 0.0
        max_spd = float(sub["peak_wind"].max()) if cnt > 0 else 0.0
        lines.append(
            f"| `[{lo:.1f}, {hi:.1f}]` | {cnt} | {pct:.1f}% | {mean_spd:.2f} | {min_spd:.2f} | {max_spd:.2f} |"
        )
    return "\n".join(lines)


def compute_burned_fraction_by_wind_bin(
    df_meta: pd.DataFrame,
    speed_bins: Sequence[tuple[float, float]] = (
        (1.0, 5.0),
        (5.0, 10.0),
        (10.0, 15.0),
        (15.0, 20.0),
    ),
) -> str:
    """Compute burned fraction statistics grouped by wind speed bins from metadata CSV DataFrame."""
    lines = [
        "| Wind Speed Range (m/s) | Sample Count | Mean Burned % | Std Burned % | Min Burned % | Max Burned % |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    if len(df_meta) == 0:
        return "\n".join(lines)

    for lo, hi in speed_bins:
        sub = df_meta[
            (df_meta["peak_wind"] >= lo)
            & (df_meta["peak_wind"] < hi if hi < 20.0 else df_meta["peak_wind"] <= hi)
        ]
        cnt = len(sub)
        if cnt > 0:
            mean_bf = float(sub["burned_fraction"].mean()) * 100.0
            std_bf = float(sub["burned_fraction"].std()) * 100.0 if cnt > 1 else 0.0
            min_bf = float(sub["burned_fraction"].min()) * 100.0
            max_bf = float(sub["burned_fraction"].max()) * 100.0
            lines.append(
                f"| `[{lo:.1f}, {hi:.1f}]` | {cnt} | {mean_bf:.1f}% | {std_bf:.1f}% | {min_bf:.1f}% | {max_bf:.1f}% |"
            )
        else:
            lines.append(f"| `[{lo:.1f}, {hi:.1f}]` | 0 | N/A | N/A | N/A | N/A |")
    return "\n".join(lines)


def generate_stratified_samples_rejection(
    fire_names: Sequence[str] | None = None,
    target_quotas: dict[str | int, int] | Sequence[int] | None = None,
    total_samples: int | None = None,
    max_attempts: int | None = None,
    base_seed: int = 42000,
    tile_size: tuple[int, int] = (256, 256),
    source_mode: str = "teacher_v1_cellular",
    horizon_min: float = 720.0,
    scenarios_config_path: str | Path = "configs/scenarios.yaml",
) -> tuple[list[SurrogateSample], RejectionSamplingReport]:
    """Generate training samples using rejection sampling against burned-fraction bin quotas.

    Each burned-fraction bin has a quota. The generator keeps sampling candidates with unique
    seeds until every bin is full or max_attempts is reached. Never duplicates a sample.

    Bins:
    - 0: [0.0, 0.2)
    - 1: [0.2, 0.5)
    - 2: [0.5, 0.9)
    - 3: [0.9, 1.0]

    Returns:
        (accepted_samples, rejection_report)
    """
    if fire_names is None:
        fire_names = SEVEN_TRAINING_FIRES

    cfg = load_held_out_fire_exclusions(scenarios_config_path)
    strat_cfg = cfg.get("stratified_sampling", {})

    # Determine bin quotas [q0, q1, q2, q3]
    bin_names = ["0-0.2", "0.2-0.5", "0.5-0.9", ">0.9"]
    bin_ranges = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.9), (0.9, 1.0)]

    quotas = [0, 0, 0, 0]
    if target_quotas is not None:
        if isinstance(target_quotas, Sequence):
            quotas = [int(q) for q in target_quotas[:4]]
        elif isinstance(target_quotas, dict):
            key_map = {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                "0": 0,
                "1": 1,
                "2": 2,
                "3": 3,
                "bin_0_0_2": 0,
                "bin_0_2_0_5": 1,
                "bin_0_5_0_9": 2,
                "bin_above_0_9": 3,
                "0-0.2": 0,
                "0.2-0.5": 1,
                "0.5-0.9": 2,
                ">0.9": 3,
            }
            for k, val in target_quotas.items():
                if k in key_map:
                    quotas[key_map[k]] = int(val)
    elif total_samples is not None:
        props = strat_cfg.get("target_proportions", [0.25, 0.25, 0.25, 0.25])
        quotas = [int(round(total_samples * p)) for p in props]
    else:
        cfg_quotas = strat_cfg.get("bin_quotas", {})
        quotas = [
            int(cfg_quotas.get("bin_0_0_2", 25)),
            int(cfg_quotas.get("bin_0_2_0_5", 25)),
            int(cfg_quotas.get("bin_0_5_0_9", 25)),
            int(cfg_quotas.get("bin_above_0_9", 25)),
        ]

    if max_attempts is None:
        max_attempts = int(strat_cfg.get("max_rejection_attempts", 2000))

    cfg_hash = get_teacher_config_hash()

    # Preload and cache rasters for training fires
    cached_fires: list[dict[str, Any]] = []
    for fire in fire_names:
        verify_fire_not_excluded(fire, scenarios_config_path)
        fbfm_base, elev_base, canopy_base, _ = load_fire_rasters_crop(
            fire_name=fire,
            tile_size=tile_size,
            scenarios_config_path=scenarios_config_path,
        )
        fuel_grp = fbfm40_to_fuel_group(fbfm_base)
        slope, asp_sin, asp_cos = compute_slope_aspect(elev_base, resolution_m=60.0)
        elev_z = ((elev_base - np.mean(elev_base)) / (np.std(elev_base) + 1e-5)).astype(np.float32)

        cached_fires.append(
            {
                "fire": fire,
                "fuel_grp": fuel_grp,
                "slope": slope,
                "asp_sin": asp_sin,
                "asp_cos": asp_cos,
                "elev_z": elev_z,
                "canopy": canopy_base,
                "fire_dir": Path("data/tiles/fires") / fire,
            }
        )

    accepted_samples: list[SurrogateSample] = []
    bin_accepted: list[list[SurrogateSample]] = [[], [], [], []]
    bin_generated: list[int] = [0, 0, 0, 0]
    bin_rejected: list[int] = [0, 0, 0, 0]

    for attempt in range(max_attempts):
        if all(len(bin_accepted[b]) >= quotas[b] for b in range(4) if quotas[b] > 0):
            break

        f_idx = attempt % len(cached_fires)
        f_info = cached_fires[f_idx]
        fire = f_info["fire"]

        seed = base_seed + attempt * 17
        rng = np.random.RandomState(seed)

        wind_base = sample_hrrr_wind_series(
            fire_dir=f_info["fire_dir"],
            rng=rng,
            window_hours=12,
            target_peak_speed_range=(1.0, 20.0),
        )

        # Perturbations
        spd_mult = float(rng.uniform(0.90, 1.10))
        dir_jitter_deg = float(rng.uniform(-20.0, 20.0))
        ign_offset_r = int(rng.randint(-10, 11))
        ign_offset_c = int(rng.randint(-10, 11))

        jitter_rad = np.deg2rad(dir_jitter_deg)
        cos_j, sin_j = np.cos(jitter_rad), np.sin(jitter_rad)
        u_j = (wind_base[:, 0] * cos_j - wind_base[:, 1] * sin_j) * spd_mult
        v_j = (wind_base[:, 0] * sin_j + wind_base[:, 1] * cos_j) * spd_mult
        hourly_wind_perturbed = np.stack([u_j, v_j], axis=-1).astype(np.float32)

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

        if source_mode == "teacher_v1_cellular":
            arr_time, c_mask = simulate_cellular_standin_arrival(
                fuel_group=f_info["fuel_grp"],
                hourly_wind_uv=hourly_wind_perturbed,
                ign_r=ign_r,
                ign_c=ign_c,
                resolution_m=60.0,
                horizon_min=horizon_min,
                speed_multiplier=spd_mult,
            )
        else:
            lb_jitter = float(rng.uniform(-0.3, 0.3))
            arr_time, c_mask = generate_analytic_arrival_surface(
                fuel_group=f_info["fuel_grp"],
                hourly_wind_uv=hourly_wind_perturbed,
                ign_r=ign_r,
                ign_c=ign_c,
                resolution_m=60.0,
                horizon_min=horizon_min,
                speed_multiplier=spd_mult,
                length_to_breadth=2.5 + lb_jitter,
            )

        bf = float(np.mean(c_mask < 0.5))
        bin_idx = assign_burn_fraction_bin(bf)
        bin_generated[bin_idx] += 1

        if len(bin_accepted[bin_idx]) < quotas[bin_idx]:
            sample_id = f"{fire}_member_{len(accepted_samples):04d}_{source_mode}"
            scaffold_tag = f"scaffold_{cfg_hash[:12]}"
            sample = SurrogateSample(
                fire_name=fire,
                sample_id=sample_id,
                fuel_group=f_info["fuel_grp"],
                slope=f_info["slope"],
                aspect_sin=f_info["asp_sin"],
                aspect_cos=f_info["asp_cos"],
                elevation_z=f_info["elev_z"],
                canopy_cover=f_info["canopy"],
                ignition_mask=ign_mask,
                hourly_wind_uv=hourly_wind_perturbed,
                target_arrival_time=arr_time,
                censor_mask=c_mask,
                scaffold_tag=scaffold_tag,
                seed=seed,
                source=source_mode,
                baseline_only=(source_mode == "analytic_wind_ellipse"),
            )
            bin_accepted[bin_idx].append(sample)
            accepted_samples.append(sample)
        else:
            bin_rejected[bin_idx] += 1

    stats: list[BinRejectionStats] = []
    unfilled_messages: list[str] = []
    for b in range(4):
        target = quotas[b]
        unique_cnt = len(bin_accepted[b])
        gen = bin_generated[b]
        rej = bin_rejected[b]
        filled = unique_cnt >= target
        if not filled and target > 0:
            msg = (
                f"Bin '{bin_names[b]}' could not be filled within the cap of {max_attempts} attempts "
                f"(target: {target}, generated in bin: {gen}, reached: {unique_cnt}, deficit: {target - unique_cnt})."
            )
            unfilled_messages.append(msg)
            logger.warning(msg)
        stats.append(
            BinRejectionStats(
                bin_name=bin_names[b],
                bin_range=bin_ranges[b],
                target=target,
                generated=gen,
                rejected=rej,
                unique_count=unique_cnt,
                filled=filled,
            )
        )

    report = RejectionSamplingReport(
        stats=stats,
        total_generated=sum(bin_generated),
        total_accepted=len(accepted_samples),
        total_rejected=sum(bin_rejected),
        max_attempts=max_attempts,
        cap_reached=any(not s.filled for s in stats if s.target > 0),
        unfilled_messages=unfilled_messages,
    )

    return accepted_samples, report


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

        fbfm_base, elev_base, canopy_base, _ = load_fire_rasters_crop(
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

            # Ellipse samples are baseline-only; teacher_v1 samples can be used for training
            is_baseline_only = current_source == "analytic_wind_ellipse"

            # Sample HRRR 12-hour wind window with synthetic scaling covering 1 to 20 m/s
            wind_base = sample_hrrr_wind_series(
                fire_dir=Path("data/tiles/fires") / fire,
                rng=rng,
                window_hours=12,
                target_peak_speed_range=(1.0, 20.0),
            )

            # Ensemble perturbations with stratified sampling retry if needed
            max_attempts = 10 if strat_enabled else 1
            arr_time: np.ndarray | None = None
            c_mask: np.ndarray | None = None
            hourly_wind_perturbed: np.ndarray | None = None
            ign_mask: np.ndarray | None = None

            for attempt in range(max_attempts):
                spd_boost = 1.0 + (attempt * 0.05 if strat_enabled else 0.0)
                spd_mult = float(rng.uniform(0.90, 1.10)) * spd_boost
                dir_jitter_deg = float(rng.uniform(-20.0, 20.0))
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
                    baseline_only=is_baseline_only,
                )
            )

    logger.info("Generated %d ensemble samples across %d fires.", len(samples), len(fire_names))
    return samples
