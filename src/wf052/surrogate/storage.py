"""Compact storage layout and dataset for surrogate fire spread samples (Phase 2B).

Storage optimization principles:
1. Static tile layers are saved ONCE per fire landscape (float16 continuous, uint8 fuel).
2. Per-sample dynamic storage stores only:
   - wind_series: (12, 2) float32
   - ignition_center: (r, c) uint16
   - arrival_time: (256, 256) uint16 (tenths of a minute precision, scale factor = 10.0)
   - censor_mask: (256, 256) bit-packed uint8 (np.packbits, 8192 bytes)
3. Delivers ~28x compression over naive float32 tensor serialization.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from wf052.surrogate.dataset import (
    SurrogateSample,
)


@dataclass(frozen=True)
class StorageFootprintReport:
    """Report on storage footprints for dataset scaling."""

    per_sample_dynamic_bytes: int
    static_bytes_per_landscape: int
    num_landscapes: int
    total_static_bytes: int
    size_5k_samples_bytes: int
    size_5k_samples_mib: float
    size_20k_samples_bytes: int
    size_20k_samples_gib: float
    uncompressed_5k_gib: float
    uncompressed_20k_gib: float
    compression_ratio: float


def compute_storage_footprint(
    num_landscapes: int = 7,
    tile_shape: tuple[int, int] = (256, 256),
) -> StorageFootprintReport:
    """Compute exact bytes per sample and total size for 5k and 20k samples."""
    h, w = tile_shape
    num_pixels = h * w

    # Static per landscape:
    # 1. fuel_group uint8: num_pixels bytes
    # 2. 5 continuous layers (slope, aspect_sin, aspect_cos, elevation_z, canopy_cover) float16: 5 * num_pixels * 2 bytes
    static_fuel_bytes = num_pixels * 1
    static_continuous_bytes = 5 * num_pixels * 2
    static_per_landscape = (
        static_fuel_bytes + static_continuous_bytes
    )  # 65536 + 655360 = 720,896 bytes (704 KB)
    total_static = static_per_landscape * num_landscapes

    # Dynamic per sample:
    # 1. hourly_wind_uv: (12, 2) float32 -> 96 bytes
    # 2. ignition_center: (ign_r, ign_c) uint16 -> 4 bytes
    # 3. arrival_time: (256, 256) uint16 -> num_pixels * 2 = 131,072 bytes
    # 4. censor_mask: (256, 256) bit-packed -> num_pixels / 8 = 8,192 bytes
    wind_bytes = 12 * 2 * 4
    ign_bytes = 2 * 2
    arrival_bytes = num_pixels * 2
    censor_packed_bytes = num_pixels // 8

    dynamic_per_sample = (
        wind_bytes + ign_bytes + arrival_bytes + censor_packed_bytes
    )  # 139,364 bytes (~136.1 KB)

    # 5k samples
    dynamic_5k = 5000 * dynamic_per_sample
    total_5k = dynamic_5k + total_static
    size_5k_mib = total_5k / (1024 * 1024)

    # 20k samples
    dynamic_20k = 20000 * dynamic_per_sample
    total_20k = dynamic_20k + total_static
    size_20k_gib = total_20k / (1024 * 1024 * 1024)

    # Naive uncompressed float32 per sample:
    # 13 spatial channels (13 * num_pixels * 4) + target (num_pixels * 4) + censor (num_pixels * 4) + wind (96)
    uncompressed_per_sample = (13 + 1 + 1) * num_pixels * 4 + wind_bytes  # 3,932,256 bytes
    uncompressed_5k_gib = (5000 * uncompressed_per_sample) / (1024 * 1024 * 1024)
    uncompressed_20k_gib = (20000 * uncompressed_per_sample) / (1024 * 1024 * 1024)

    compression_ratio = uncompressed_per_sample / dynamic_per_sample

    return StorageFootprintReport(
        per_sample_dynamic_bytes=dynamic_per_sample,
        static_bytes_per_landscape=static_per_landscape,
        num_landscapes=num_landscapes,
        total_static_bytes=total_static,
        size_5k_samples_bytes=total_5k,
        size_5k_samples_mib=size_5k_mib,
        size_20k_samples_bytes=total_20k,
        size_20k_samples_gib=size_20k_gib,
        uncompressed_5k_gib=uncompressed_5k_gib,
        uncompressed_20k_gib=uncompressed_20k_gib,
        compression_ratio=compression_ratio,
    )


def save_compact_dataset(
    samples: Sequence[SurrogateSample],
    output_dir: Path | str,
    scale_factor: float = 10.0,
) -> dict[str, Any]:
    """Serialize stand-in samples into the compact static-once storage layout.

    Layout created in output_dir:
    - static_landscapes.npz: fuel_group (uint8) & continuous terrain (5, H, W float16) per fire.
    - samples.npz: dynamic wind, ignition centers, uint16 arrival, bit-packed censor masks.
    - metadata.json: metadata and scaling attributes.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    static_dict: dict[str, np.ndarray] = {}
    saved_fires: set[str] = set()

    sample_ids: list[str] = []
    fire_names: list[str] = []
    ignition_centers: list[list[int]] = []
    wind_series_list: list[np.ndarray] = []
    arrival_uint16_list: list[np.ndarray] = []
    censor_packed_list: list[np.ndarray] = []

    for s in samples:
        # Save static layer once per fire landscape
        if s.fire_name not in saved_fires:
            fg_uint8 = s.fuel_group.astype(np.uint8)
            cont_f16 = np.stack(
                [
                    s.slope.astype(np.float16),
                    s.aspect_sin.astype(np.float16),
                    s.aspect_cos.astype(np.float16),
                    s.elevation_z.astype(np.float16),
                    s.canopy_cover.astype(np.float16),
                ],
                axis=0,
            )
            static_dict[f"{s.fire_name}_fuel"] = fg_uint8
            static_dict[f"{s.fire_name}_continuous"] = cont_f16
            saved_fires.add(s.fire_name)

        # Extract ignition center coordinates from ignition_mask
        coords = np.argwhere(s.ignition_mask > 0.5)
        if len(coords) > 0:
            ign_r = int(np.round(np.mean(coords[:, 0])))
            ign_c = int(np.round(np.mean(coords[:, 1])))
        else:
            ign_r, ign_c = s.fuel_group.shape[0] // 2, s.fuel_group.shape[1] // 2

        # Scale arrival time to uint16
        arr_scaled = np.clip(
            np.round(s.target_arrival_time * scale_factor),
            0,
            65535,
        ).astype(np.uint16)

        # Bit-pack censor mask (boolean -> 1 bit per cell)
        censor_bool = s.censor_mask > 0.5
        censor_packed = np.packbits(censor_bool)

        sample_ids.append(s.sample_id)
        fire_names.append(s.fire_name)
        ignition_centers.append([ign_r, ign_c])
        wind_series_list.append(s.hourly_wind_uv.astype(np.float32))
        arrival_uint16_list.append(arr_scaled)
        censor_packed_list.append(censor_packed)

    # Save static archive
    np.savez_compressed(out_path / "static_landscapes.npz", **static_dict)

    # Save dynamic archive
    np.savez_compressed(
        out_path / "samples.npz",
        sample_ids=np.array(sample_ids),
        fire_names=np.array(fire_names),
        ignition_centers=np.array(ignition_centers, dtype=np.uint16),
        wind_series=np.stack(wind_series_list, axis=0),
        arrival_uint16=np.stack(arrival_uint16_list, axis=0),
        censor_packed=np.stack(censor_packed_list, axis=0),
    )

    meta = {
        "num_samples": len(samples),
        "scale_factor": scale_factor,
        "unique_fires": sorted(list(saved_fires)),
        "tile_shape": list(samples[0].fuel_group.shape),
        "teacher_config_hash": getattr(samples[0], "teacher_config_hash", "a6a92f81f7a9c72b5899d5d3173f86943a7f98648c4f210dcc5c646d2b4e85a5"),
        "canonical_sim_hash": getattr(samples[0], "canonical_sim_hash", "61c9baec36ce945a464e3d91a62e228841a3a7900e450a446f9a1e1e4a56eed3"),
        "source": getattr(samples[0], "source", "teacher_v2_ensemble"),
    }
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


class CompactSurrogateDataset(Dataset):
    """PyTorch Dataset reading the compact storage layout.

    Caches static tile landscape layers once in memory, dynamically unpacks
    bit-packed censor masks and uint16 arrival times, and generates the exact
    13-channel spatial tensor matching the standard SurrogateBatch contract.
    """

    def __init__(self, dataset_dir: Path | str) -> None:
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Compact dataset directory not found: {self.dataset_dir}")

        with open(self.dataset_dir / "metadata.json", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.scale_factor = float(self.metadata.get("scale_factor", 10.0))
        self.tile_shape = tuple(self.metadata.get("tile_shape", [256, 256]))

        # Load static layers
        static_data = np.load(self.dataset_dir / "static_landscapes.npz")
        self.static_cache: dict[str, dict[str, np.ndarray]] = {}
        for fire in self.metadata["unique_fires"]:
            self.static_cache[fire] = {
                "fuel_group": static_data[f"{fire}_fuel"],
                "continuous": static_data[f"{fire}_continuous"].astype(np.float32),
            }

        # Load dynamic samples
        samples_data = np.load(self.dataset_dir / "samples.npz")
        self.sample_ids = samples_data["sample_ids"]
        self.fire_names = samples_data["fire_names"]
        self.ignition_centers = samples_data["ignition_centers"]
        self.wind_series = samples_data["wind_series"]
        self.arrival_uint16 = samples_data["arrival_uint16"]
        self.censor_packed = samples_data["censor_packed"]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        fire = str(self.fire_names[idx])
        s_id = str(self.sample_ids[idx])

        # 1. Retrieve static layers
        static_entry = self.static_cache[fire]
        fg_raw = static_entry["fuel_group"]  # (H, W) uint8 in [0, 6]
        cont_5 = static_entry[
            "continuous"
        ]  # (5, H, W) float32 (slope, asp_sin, asp_cos, elev_z, canopy)

        # One-hot encode fuel group (7 channels)
        fg_tensor = torch.from_numpy(fg_raw).long()
        fg_onehot = torch.nn.functional.one_hot(fg_tensor, num_classes=7).permute(2, 0, 1).float()

        # 2. Reconstruct ignition mask from ignition_center (1 channel)
        ign_r, ign_c = int(self.ignition_centers[idx, 0]), int(self.ignition_centers[idx, 1])
        ign_mask = np.zeros(self.tile_shape, dtype=np.float32)
        r_lo, r_hi = max(0, ign_r - 1), min(self.tile_shape[0], ign_r + 2)
        c_lo, c_hi = max(0, ign_c - 1), min(self.tile_shape[1], ign_c + 2)
        ign_mask[r_lo:r_hi, c_lo:c_hi] = 1.0
        ign_tensor = torch.from_numpy(ign_mask).unsqueeze(0).float()

        # Combine: 7 fuel one-hot + 5 continuous terrain + 1 ignition mask = 13 channels
        cont_tensor = torch.from_numpy(cont_5).float()
        spatial_features = torch.cat([fg_onehot, cont_tensor, ign_tensor], dim=0)

        # 3. Dynamic target: arrival time and unpacked censor mask
        arrival_min = self.arrival_uint16[idx].astype(np.float32) / self.scale_factor
        censor_unpacked = (
            np.unpackbits(self.censor_packed[idx])[: self.tile_shape[0] * self.tile_shape[1]]
            .reshape(self.tile_shape)
            .astype(np.float32)
        )

        return {
            "spatial_features": spatial_features,
            "fuel_group": fg_tensor,
            "wind_series": torch.from_numpy(self.wind_series[idx]).float(),
            "target_arrival": torch.from_numpy(arrival_min).float(),
            "censor_mask": torch.from_numpy(censor_unpacked).float(),
            "fire_name": fire,
            "sample_id": s_id,
        }
