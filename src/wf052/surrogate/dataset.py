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
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


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
    target_arrival_time: np.ndarray  # (256, 256) float32 in minutes
    censor_mask: np.ndarray  # (256, 256) float32 {0, 1} (1=unburned/censored)
    scaffold_tag: str  # Tag: "scaffold" + teacher_config_hash
    seed: int
    source: str = (
        "analytic_wind_ellipse"  # Source generator ("analytic_wind_ellipse", "teacher_v1_cellular")
    )
    baseline_only: bool = False  # Flag for baseline-only samples (e.g. analytic ellipse)


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

    def to(self, device: torch.device) -> "SurrogateBatch":
        return SurrogateBatch(
            spatial_features=self.spatial_features.to(device),
            fuel_group=self.fuel_group.to(device),
            wind_series=self.wind_series.to(device),
            target_arrival=self.target_arrival.to(device),
            censor_mask=self.censor_mask.to(device),
            fire_names=self.fire_names,
            sample_ids=self.sample_ids,
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

        return {
            "spatial_features": spatial_inputs,
            "fuel_group": fg_tensor,
            "wind_series": torch.from_numpy(s.hourly_wind_uv).float(),
            "target_arrival": torch.from_numpy(s.target_arrival_time).float(),
            "censor_mask": torch.from_numpy(s.censor_mask).float(),
            "fire_name": s.fire_name,
            "sample_id": s.sample_id,
        }


def collate_surrogate_batch(batch_list: list[dict[str, Any]]) -> SurrogateBatch:
    """Collate list of samples into typed SurrogateBatch."""
    spatial = torch.stack([b["spatial_features"] for b in batch_list], dim=0)
    fuel = torch.stack([b["fuel_group"] for b in batch_list], dim=0)
    wind = torch.stack([b["wind_series"] for b in batch_list], dim=0)
    arrival = torch.stack([b["target_arrival"] for b in batch_list], dim=0)
    censor = torch.stack([b["censor_mask"] for b in batch_list], dim=0)
    f_names = [b["fire_name"] for b in batch_list]
    s_ids = [b["sample_id"] for b in batch_list]

    return SurrogateBatch(
        spatial_features=spatial,
        fuel_group=fuel,
        wind_series=wind,
        target_arrival=arrival,
        censor_mask=censor,
        fire_names=f_names,
        sample_ids=s_ids,
    )
