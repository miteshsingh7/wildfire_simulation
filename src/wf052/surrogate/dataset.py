"""Surrogate data contract, feature transformations, and PyTorch dataset.

Inputs:
- fuel_group (7 classes: NB, GR, GS, SH, TU, TL, SB)
- slope (continuous radians/gradient)
- aspect_sin, aspect_cos (trigonometric aspect)
- elevation (z-scored per tile)
- canopy_cover (normalized [0, 1])
- ignition_mask (binary {0, 1})
- hourly_wind_uv (shape (12, 2) in m/s)

Target:
- arrival_time (shape (256, 256) in minutes [0, 720])
- censor_mask (binary {0, 1}, 1 = unburned within horizon, 0 = burned)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

SCHEMA_VERSION: str = "2e"

FIRE_SPLITS: dict[str, list[str]] = {
    "train": ["kincade", "glass", "zogg", "creek"],
    "val": ["bobcat"],
    "test": ["caldor", "tamarack"],
}


def fbfm40_to_fuel_group(fbfm: np.ndarray) -> np.ndarray:
    """Map raw Scott & Burgan 40 fuel models (or standard Landfire codes) to 7 fuel groups.

    0: Non-burnable / Urban / Water (codes 91-99 or unburnable <= 100)
    1: Grass (GR1-GR9, codes 101-109)
    2: Grass-Shrub (GS1-GS4, codes 121-124)
    3: Shrub (SH1-SH9, codes 141-149)
    4: Timber-Understory (TU1-TU5, codes 161-165)
    5: Timber-Litter (TL1-TL9, codes 181-189)
    6: Slash-Blowdown (SB1-SB4, codes 201-204)
    """
    groups = np.zeros_like(fbfm, dtype=np.int64)
    # Grass: 101 to 109
    groups[(fbfm >= 101) & (fbfm <= 109)] = 1
    # Grass-Shrub: 121 to 124
    groups[(fbfm >= 121) & (fbfm <= 124)] = 2
    # Shrub: 141 to 149
    groups[(fbfm >= 141) & (fbfm <= 149)] = 3
    # Timber-Understory: 161 to 165
    groups[(fbfm >= 161) & (fbfm <= 165)] = 4
    # Timber-Litter: 181 to 189
    groups[(fbfm >= 181) & (fbfm <= 189)] = 5
    # Slash-Blowdown: 201 to 204
    groups[(fbfm >= 201) & (fbfm <= 204)] = 6
    # Standard 13 Anderson models fallback
    groups[(fbfm >= 1) & (fbfm <= 3)] = 1  # Grass
    groups[(fbfm >= 4) & (fbfm <= 7)] = 3  # Shrub
    groups[(fbfm >= 8) & (fbfm <= 10)] = 5  # Timber
    groups[(fbfm >= 11) & (fbfm <= 13)] = 6  # Slash
    return groups


def compute_slope_aspect(
    elevation: np.ndarray, resolution_m: float = 60.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute slope gradient, aspect sine, and aspect cosine from 2D elevation grid.

    Returns:
        (slope, aspect_sin, aspect_cos)
    """
    dy, dx = np.gradient(elevation, resolution_m, resolution_m)
    slope = np.arctan(np.hypot(dx, dy)).astype(np.float32)

    # Meteorological / cartographic aspect in radians: 0 = East, pi/2 = North
    aspect = np.arctan2(-dy, dx).astype(np.float32)
    aspect_sin = np.sin(aspect).astype(np.float32)
    aspect_cos = np.cos(aspect).astype(np.float32)
    return slope, aspect_sin, aspect_cos


def calculate_upwind_third_ignition(
    hourly_wind_uv: np.ndarray,
    tile_shape: tuple[int, int] = (256, 256),
    offset_r: int = 0,
    offset_c: int = 0,
) -> tuple[int, int]:
    """Calculate ignition location placed in the upwind third of a tile.

    Args:
        hourly_wind_uv: (T, 2) array of hourly wind vectors (u, v) in m/s.
        tile_shape: (H, W) tuple.
        offset_r: Row jitter perturbation.
        offset_c: Col jitter perturbation.

    Returns:
        (ign_row, ign_col) inside the tile.
    """
    mean_u = float(np.mean(hourly_wind_uv[:, 0]))
    mean_v = float(np.mean(hourly_wind_uv[:, 1]))

    # Wind blowing towards angle theta
    theta = np.arctan2(mean_v, mean_u)

    # Center of tile
    cr, cc = tile_shape[0] / 2.0, tile_shape[1] / 2.0
    # Displacement towards upwind (opposite to wind direction)
    # Target upwind third: ~0.55 of half-width from center
    disp = 0.55 * (min(tile_shape) / 2.0)

    ign_r = int(np.clip(cr - disp * np.sin(theta) + offset_r, 20, tile_shape[0] - 20))
    ign_c = int(np.clip(cc - disp * np.cos(theta) + offset_c, 20, tile_shape[1] - 20))
    return ign_r, ign_c


@dataclass(frozen=True)
class SurrogateSample:
    """Strongly typed container for a single surrogate data sample satisfying the contract."""

    fire_name: str
    sample_id: str
    fuel_group: np.ndarray  # (256, 256) int64 in [0, 6]
    slope: np.ndarray  # (256, 256) float32
    aspect_sin: np.ndarray  # (256, 256) float32
    aspect_cos: np.ndarray  # (256, 256) float32
    elevation_z: np.ndarray  # (256, 256) float32 (z-scored)
    canopy_cover: np.ndarray  # (256, 256) float32 in [0, 1]
    ignition_mask: np.ndarray  # (256, 256) float32 {0, 1}
    hourly_wind_uv: np.ndarray  # (12, 2) float32
    target_arrival_time: np.ndarray  # (256, 256) float32 in minutes (q50)
    censor_mask: np.ndarray  # (256, 256) float32 {0, 1} (1=unburned/censored at q50)
    scaffold_tag: str  # Tag: "scaffold" + teacher_config_hash
    seed: int
    source: str = "teacher_v2_ensemble"  # Source generator ("teacher_v2_ensemble", "teacher_v1_cellular", "analytic_wind_ellipse")
    baseline_only: bool = False  # Flag for baseline-only samples (e.g. analytic ellipse)
    # Multi-seed ensemble attributes (Phase 2D)
    burn_prob: np.ndarray | None = None
    arrival_q10: np.ndarray | None = None
    arrival_q50: np.ndarray | None = None
    arrival_q90: np.ndarray | None = None
    censor_mask_q10: np.ndarray | None = None
    censor_mask_q50: np.ndarray | None = None
    censor_mask_q90: np.ndarray | None = None
    k_ensemble: int = 16
    teacher_config_hash: str = "a6a92f81f7a9c72b5899d5d3173f86943a7f98648c4f210dcc5c646d2b4e85a5"
    canonical_sim_hash: str = "61c9baec36ce945a464e3d91a62e228841a3a7900e450a446f9a1e1e4a56eed3"
    # Scenario descriptor attributes (Phase 2E)
    fire_id: str = ""
    tile_id: str = ""
    crop_origin: tuple[int, int] = (0, 0)
    ignition_cell: tuple[int, int] = (128, 128)
    schema_version: str = SCHEMA_VERSION
    git_commit: str = ""


@dataclass(frozen=True)
class SurrogateBatch:
    """Strongly typed container for batched PyTorch surrogate tensors."""

    spatial_features: torch.Tensor  # (B, 13, 256, 256) float32
    fuel_group: torch.Tensor  # (B, 256, 256) int64
    wind_series: torch.Tensor  # (B, 12, 2) float32
    target_arrival: torch.Tensor  # (B, 256, 256) float32
    censor_mask: torch.Tensor  # (B, 256, 256) float32
    fire_names: list[str]
    sample_ids: list[str]
    burn_prob: torch.Tensor | None = None
    target_quantiles: torch.Tensor | None = None
    censor_quantiles: torch.Tensor | None = None

    def to(self, device: torch.device) -> "SurrogateBatch":
        return SurrogateBatch(
            spatial_features=self.spatial_features.to(device),
            fuel_group=self.fuel_group.to(device),
            wind_series=self.wind_series.to(device),
            target_arrival=self.target_arrival.to(device),
            censor_mask=self.censor_mask.to(device),
            fire_names=self.fire_names,
            sample_ids=self.sample_ids,
            burn_prob=self.burn_prob.to(device) if self.burn_prob is not None else None,
            target_quantiles=self.target_quantiles.to(device) if self.target_quantiles is not None else None,
            censor_quantiles=self.censor_quantiles.to(device) if self.censor_quantiles is not None else None,
        )


class SurrogateDataset(Dataset):
    """PyTorch Dataset implementation for wind-conditioned wildland fire surrogate modeling."""

    def __init__(self, samples: list[SurrogateSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]

        # One-hot encode fuel groups (7 channels)
        fg_tensor = torch.from_numpy(s.fuel_group).long()
        fg_onehot = torch.nn.functional.one_hot(fg_tensor, num_classes=7).permute(2, 0, 1).float()

        # Continuous terrain & ignition channels (6 channels)
        cont_channels = torch.from_numpy(
            np.stack(
                [
                    s.slope,
                    s.aspect_sin,
                    s.aspect_cos,
                    s.elevation_z,
                    s.canopy_cover,
                    s.ignition_mask,
                ],
                axis=0,
            )
        ).float()

        # Combined 13 spatial input channels
        spatial_inputs = torch.cat([fg_onehot, cont_channels], dim=0)

        item: dict[str, Any] = {
            "spatial_features": spatial_inputs,
            "fuel_group": fg_tensor,
            "wind_series": torch.from_numpy(s.hourly_wind_uv).float(),
            "target_arrival": torch.from_numpy(s.target_arrival_time).float(),
            "censor_mask": torch.from_numpy(s.censor_mask).float(),
            "fire_name": s.fire_name,
            "sample_id": s.sample_id,
        }

        if s.burn_prob is not None:
            item["burn_prob"] = torch.from_numpy(s.burn_prob).float()

        if s.arrival_q10 is not None and s.arrival_q50 is not None and s.arrival_q90 is not None:
            item["target_quantiles"] = torch.from_numpy(
                np.stack([s.arrival_q10, s.arrival_q50, s.arrival_q90], axis=0)
            ).float()

        if s.censor_mask_q10 is not None and s.censor_mask_q50 is not None and s.censor_mask_q90 is not None:
            item["censor_quantiles"] = torch.from_numpy(
                np.stack([s.censor_mask_q10, s.censor_mask_q50, s.censor_mask_q90], axis=0)
            ).float()

        return item


def collate_surrogate_batch(batch_list: list[dict[str, Any]]) -> SurrogateBatch:
    """Collate list of samples into typed SurrogateBatch."""
    spatial = torch.stack([b["spatial_features"] for b in batch_list], dim=0)
    fuel = torch.stack([b["fuel_group"] for b in batch_list], dim=0)
    wind = torch.stack([b["wind_series"] for b in batch_list], dim=0)
    arrival = torch.stack([b["target_arrival"] for b in batch_list], dim=0)
    censor = torch.stack([b["censor_mask"] for b in batch_list], dim=0)
    f_names = [b["fire_name"] for b in batch_list]
    s_ids = [b["sample_id"] for b in batch_list]

    burn_prob = (
        torch.stack([b["burn_prob"] for b in batch_list], dim=0)
        if "burn_prob" in batch_list[0]
        else None
    )
    t_quantiles = (
        torch.stack([b["target_quantiles"] for b in batch_list], dim=0)
        if "target_quantiles" in batch_list[0]
        else None
    )
    c_quantiles = (
        torch.stack([b["censor_quantiles"] for b in batch_list], dim=0)
        if "censor_quantiles" in batch_list[0]
        else None
    )

    return SurrogateBatch(
        spatial_features=spatial,
        fuel_group=fuel,
        wind_series=wind,
        target_arrival=arrival,
        censor_mask=censor,
        fire_names=f_names,
        sample_ids=s_ids,
        burn_prob=burn_prob,
        target_quantiles=t_quantiles,
        censor_quantiles=c_quantiles,
    )


_FIRE_RASTER_CACHE: dict[str, dict[str, np.ndarray]] = {}


def get_cached_fire_rasters(
    fire_name: str, fires_base_dir: str | Path = "data/tiles/fires"
) -> dict[str, np.ndarray]:
    """Retrieve or cache static full fire domain rasters in memory."""
    clean_name = fire_name.lower().strip()
    if clean_name in _FIRE_RASTER_CACHE:
        return _FIRE_RASTER_CACHE[clean_name]

    fire_dir = Path(fires_base_dir) / clean_name
    if not fire_dir.exists():
        raise FileNotFoundError(f"Fire directory {fire_dir} not found for raster caching.")

    with rasterio.open(fire_dir / "elevation.tif") as src:
        elev_full = src.read(1).astype(np.float32)
    with rasterio.open(fire_dir / "fbfm40.tif") as src:
        fbfm_full = src.read(1).astype(np.int64)
    with rasterio.open(fire_dir / "canopy_cover.tif") as src:
        canopy_full = src.read(1).astype(np.float32)
        if np.nanmax(canopy_full) > 1.0:
            canopy_full = np.clip(canopy_full / 100.0, 0.0, 1.0)

    fuel_grp_full = fbfm40_to_fuel_group(fbfm_full)

    cached = {
        "elevation": elev_full,
        "fbfm40": fbfm_full,
        "fuel_group": fuel_grp_full,
        "canopy": canopy_full,
    }
    _FIRE_RASTER_CACHE[clean_name] = cached
    return cached


def rebuild_spatial_inputs(
    fire_id: str,
    crop_origin: tuple[int, int],
    ignition_cell: tuple[int, int],
    fires_base_dir: str | Path = "data/tiles/fires",
    tile_shape: tuple[int, int] = (256, 256),
) -> torch.Tensor:
    """Rebuild the exact 13-channel spatial tensor on the fly from static fire rasters.

    Channels:
    0..6: Fuel group 1-hot (7 channels)
    7: Slope (radians)
    8: Aspect sine
    9: Aspect cosine
    10: Elevation z-score normalized per tile
    11: Canopy cover [0, 1]
    12: Ignition mask (3x3 box around ignition_cell)
    """
    cached = get_cached_fire_rasters(fire_id, fires_base_dir=fires_base_dir)
    r0, c0 = crop_origin
    h, w = tile_shape

    elev_crop = cached["elevation"][r0 : r0 + h, c0 : c0 + w].copy()
    fuel_grp_crop = cached["fuel_group"][r0 : r0 + h, c0 : c0 + w].copy()
    canopy_crop = cached["canopy"][r0 : r0 + h, c0 : c0 + w].copy()

    # One-hot fuel groups (7 channels)
    fg_tensor = torch.from_numpy(fuel_grp_crop).long()
    fg_onehot = torch.nn.functional.one_hot(fg_tensor, num_classes=7).permute(2, 0, 1).float()

    # Continuous terrain
    slope, asp_sin, asp_cos = compute_slope_aspect(elev_crop, resolution_m=60.0)
    elev_z = ((elev_crop - np.mean(elev_crop)) / (np.std(elev_crop) + 1e-5)).astype(np.float32)

    # Ignition mask (1 channel)
    ign_r, ign_c = ignition_cell
    ign_mask = np.zeros(tile_shape, dtype=np.float32)
    r_lo, r_hi = max(0, ign_r - 1), min(h, ign_r + 2)
    c_lo, c_hi = max(0, ign_c - 1), min(w, ign_c + 2)
    ign_mask[r_lo:r_hi, c_lo:c_hi] = 1.0
    ign_tensor = torch.from_numpy(ign_mask).unsqueeze(0).float()

    cont_tensor = torch.from_numpy(
        np.stack([slope, asp_sin, asp_cos, elev_z, canopy_crop], axis=0)
    ).float()

    spatial_features = torch.cat([fg_onehot, cont_tensor, ign_tensor], dim=0)
    return spatial_features


class CompactSurrogateDataset(Dataset):
    """PyTorch Dataset reading compact .npz shards with on-the-fly 13-channel reconstruction.

    Labels stored:
    - arrival_quantiles as float16 (minutes in [0, 720])
    - burn_prob as uint8 (value * 255.0)
    - censor masks as bit-packed uint8
    - dynamic wind series as float32

    Rebuilds 13-channel spatial tensor on the fly from cached fire tiles.
    """

    def __init__(
        self,
        shard_paths: list[Path | str] | Path | str,
        fires_base_dir: str | Path = "data/tiles/fires",
        expected_schema_version: str = SCHEMA_VERSION,
    ) -> None:
        if isinstance(shard_paths, (str, Path)):
            p = Path(shard_paths)
            if p.is_dir():
                self.shard_paths = sorted(p.glob("shard_*.npz"))
            else:
                self.shard_paths = [p]
        else:
            self.shard_paths = [Path(p) for p in shard_paths]

        self.fires_base_dir = Path(fires_base_dir)
        self.expected_schema_version = expected_schema_version

        self.samples_index: list[tuple[int, int]] = []
        self.loaded_shards: list[dict[str, np.ndarray]] = []

        for s_idx, sp in enumerate(self.shard_paths):
            if not sp.exists():
                raise FileNotFoundError(f"Shard file not found: {sp}")
            npz = np.load(sp, allow_pickle=True)
            schema_v = str(npz["schema_version"][0]) if "schema_version" in npz else "legacy"
            if schema_v != self.expected_schema_version:
                raise ValueError(
                    f"Shard '{sp.name}' has schema_version '{schema_v}', "
                    f"expected '{self.expected_schema_version}'!"
                )
            n_samples = len(npz["sample_ids"])
            self.loaded_shards.append({k: npz[k] for k in npz.files})
            for i in range(n_samples):
                self.samples_index.append((s_idx, i))

        self.fire_ids: list[str] = [
            str(self.loaded_shards[s_idx]["fire_ids"][i]) for s_idx, i in self.samples_index
        ]
        self.sample_ids: list[str] = [
            str(self.loaded_shards[s_idx]["sample_ids"][i]) for s_idx, i in self.samples_index
        ]

    def __len__(self) -> int:
        return len(self.samples_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s_idx, i = self.samples_index[idx]
        shard = self.loaded_shards[s_idx]

        fire_id = str(shard["fire_ids"][i])
        sample_id = str(shard["sample_ids"][i])
        crop_origin = (int(shard["crop_origins"][i, 0]), int(shard["crop_origins"][i, 1]))
        ign_cell = (int(shard["ignition_cells"][i, 0]), int(shard["ignition_cells"][i, 1]))

        # Rebuild 13-channel spatial tensor
        spatial_features = rebuild_spatial_inputs(
            fire_id=fire_id,
            crop_origin=crop_origin,
            ignition_cell=ign_cell,
            fires_base_dir=self.fires_base_dir,
            tile_shape=(256, 256),
        )

        # Decode labels
        arr_q = shard["arrival_quantiles"][i].astype(np.float32)
        bp = shard["burn_prob"][i].astype(np.float32) / 255.0

        c_packed = shard["censor_quantiles_packed"][i]  # (3, 8192) uint8
        c_unpacked = (
            np.unpackbits(c_packed, axis=-1)
            .reshape(3, 256, 256)
            .astype(np.float32)
        )

        wind = shard["wind_series"][i].astype(np.float32)

        cached = get_cached_fire_rasters(fire_id, fires_base_dir=self.fires_base_dir)
        r0, c0 = crop_origin
        fuel_grp_crop = cached["fuel_group"][r0 : r0 + 256, c0 : c0 + 256]
        fg_tensor = torch.from_numpy(fuel_grp_crop).long()

        return {
            "spatial_features": spatial_features,
            "fuel_group": fg_tensor,
            "wind_series": torch.from_numpy(wind).float(),
            "target_arrival": torch.from_numpy(arr_q[1]).float(),
            "censor_mask": torch.from_numpy(c_unpacked[1]).float(),
            "burn_prob": torch.from_numpy(bp).float(),
            "target_quantiles": torch.from_numpy(arr_q).float(),
            "censor_quantiles": torch.from_numpy(c_unpacked).float(),
            "fire_name": fire_id,
            "sample_id": sample_id,
        }
