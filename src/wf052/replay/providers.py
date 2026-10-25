"""Pluggable fire arrival field providers for the causal replay harness.

Providers:
- Provider A (TeacherFieldProvider): Frozen teacher_v1 raster from Phase 1a.
- Provider B (ObservationInflatedProvider): Heuristic spot-fire inflation from
  empirically observed detections within R_spot downwind. (Labelled heuristic, not physics).
- Provider C (SurrogateFieldProvider): Stub for ML surrogate model to be trained in future.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr

from wf052.routing.contracts import ArrivalField, validate_arrival_field


class FieldProvider(ABC):
    """Abstract interface for causal arrival field providers."""

    @abstractmethod
    def get_field(
        self,
        current_step_min: float,
        visible_observations: gpd.GeoDataFrame | None = None,
    ) -> ArrivalField:
        """Return the arrival field valid at simulation clock time current_step_min."""
        ...


class TeacherFieldProvider(FieldProvider):
    """Provider A: Teacher v1 frozen raster field.

    Loads the frozen 60m teacher raster (arrival_time_teacher_v1_60m_seed42.tif).
    Note: As documented in Phase 1a, teacher_v1 stalls around 222 minutes post-ignition
    covering ~3,918 cells and does not reach the western Paradise evacuation corridors.
    """

    def __init__(
        self,
        tif_path: str | Path = "data/tiles/arrival_time_teacher_v1_60m_seed42.tif",
        horizon_minutes: float = 720.0,
    ) -> None:
        self.tif_path = Path(tif_path)
        self.horizon_minutes = horizon_minutes
        self._cached_field: ArrivalField | None = None
        self._load_raster()

    def _load_raster(self) -> None:
        if not self.tif_path.exists():
            raise FileNotFoundError(f"Teacher raster not found at {self.tif_path}")

        with rasterio.open(self.tif_path) as src:
            data = src.read(1).astype(np.float64)
            bounds = src.bounds
            width = src.width
            height = src.height

        dx = (bounds.right - bounds.left) / width
        dy = (bounds.top - bounds.bottom) / height
        gx = np.linspace(bounds.left + dx / 2.0, bounds.right - dx / 2.0, width)
        gy = np.linspace(bounds.top - dy / 2.0, bounds.bottom + dy / 2.0, height)

        data = np.where(np.isfinite(data) & (data >= 0.0), data, np.inf)

        ds = xr.Dataset(
            data_vars={
                "arrival_q10": (["y", "x"], data),
                "arrival_q50": (["y", "x"], data),
                "arrival_q90": (["y", "x"], data),
            },
            coords={"x": gx, "y": gy},
            attrs={
                "provider": "teacher_v1",
                "source_file": str(self.tif_path),
                "crs": "EPSG:32610",
                "ignition_time": "2018-11-08T14:33:00Z",
            },
        )
        validate_arrival_field(ds)
        self._cached_field = ds

    def get_field(
        self,
        current_step_min: float,
        visible_observations: gpd.GeoDataFrame | None = None,
    ) -> ArrivalField:
        if self._cached_field is None:
            self._load_raster()
        return self._cached_field


class TeacherV2FieldProvider(FieldProvider):
    """Provider A2: Teacher v2 frozen field on Camp Fire domain (ember kernel + causal assimilation).

    Physics: Contact spread (P_base=0.22) + non-local lognormal ember spotting
    (alpha=0.0040, beta=110.0 m/(m/s), p_urban_wui=0.15) calibrated on wind training fires.
    Causal Assimilation: At clock time t, merges observed burned extents up to t - latency
    with the physics forward simulation.
    """

    def __init__(
        self,
        tif_path: str | Path = "data/tiles/arrival_time_teacher_v2_60m_seed42.tif",
        assimilate: bool = True,
        patch_radius_m: float = 120.0,
    ) -> None:
        self.tif_path = Path(tif_path)
        self.assimilate = assimilate
        self.patch_radius_m = patch_radius_m
        self._cached_base_field: ArrivalField | None = None
        self._gx: np.ndarray | None = None
        self._gy: np.ndarray | None = None
        self._load_raster()

    def _load_raster(self) -> None:
        if not self.tif_path.exists():
            raise FileNotFoundError(f"Teacher v2 raster not found at {self.tif_path}")

        with rasterio.open(self.tif_path) as src:
            data = src.read(1).astype(np.float64)
            bounds = src.bounds
            width = src.width
            height = src.height

        dx = (bounds.right - bounds.left) / width
        dy = (bounds.top - bounds.bottom) / height
        gx = np.linspace(bounds.left + dx / 2.0, bounds.right - dx / 2.0, width)
        gy = np.linspace(bounds.top - dy / 2.0, bounds.bottom + dy / 2.0, height)
        self._gx = gx
        self._gy = gy

        data = np.where(np.isfinite(data) & (data >= 0.0), data, np.inf)

        ds = xr.Dataset(
            data_vars={
                "arrival_q10": (["y", "x"], data),
                "arrival_q50": (["y", "x"], data),
                "arrival_q90": (["y", "x"], data),
            },
            coords={"x": gx, "y": gy},
            attrs={
                "provider": "teacher_v2",
                "source_file": str(self.tif_path),
                "crs": "EPSG:32610",
                "ignition_time": "2018-11-08T14:33:00Z",
            },
        )
        validate_arrival_field(ds)
        self._cached_base_field = ds

    def get_field(
        self,
        current_step_min: float,
        visible_observations: gpd.GeoDataFrame | None = None,
    ) -> ArrivalField:
        if self._cached_base_field is None:
            self._load_raster()

        if not self.assimilate or visible_observations is None or len(visible_observations) == 0:
            return self._cached_base_field

        gx = self._gx
        gy = self._gy
        if gx is None or gy is None:
            return self._cached_base_field

        ny, nx = len(gy), len(gx)
        dx = float(gx[1] - gx[0])
        dy = float(gy[0] - gy[1])

        base_arr = self._cached_base_field["arrival_q50"].values.copy()
        assim_arr = base_arr.copy()

        for _, obs in visible_observations.iterrows():
            t_obs = float(obs.get("elapsed_minutes_adj", obs.get("elapsed_minutes", 0.0)))
            if t_obs > current_step_min:
                continue
            geom = obs.geometry
            if geom is None:
                continue
            ox = float(geom.x) if hasattr(geom, "x") else float(geom.centroid.x)
            oy = float(geom.y) if hasattr(geom, "y") else float(geom.centroid.y)

            # Local patch around observation
            c_min = max(0, int((ox - self.patch_radius_m - gx[0]) / dx))
            c_max = min(nx, int((ox + self.patch_radius_m - gx[0]) / dx) + 1)
            r_min = max(0, int((gy[0] - (oy + self.patch_radius_m)) / dy))
            r_max = min(ny, int((gy[0] - (oy - self.patch_radius_m)) / dy) + 1)

            if c_min >= c_max or r_min >= r_max:
                continue

            sub = assim_arr[r_min:r_max, c_min:c_max]
            sub[:] = np.minimum(sub, t_obs)

        ds = xr.Dataset(
            data_vars={
                "arrival_q10": (["y", "x"], assim_arr),
                "arrival_q50": (["y", "x"], assim_arr),
                "arrival_q90": (["y", "x"], assim_arr),
            },
            coords={"x": gx, "y": gy},
            attrs={
                "provider": "teacher_v2_causal_assimilated",
                "label": "teacher_v2 (embers + causal assimilation)",
                "crs": "EPSG:32610",
                "ignition_time": "2018-11-08T14:33:00Z",
            },
        )
        validate_arrival_field(ds)
        return ds


class ObservationInflatedProvider(FieldProvider):
    """Provider B: Heuristic observation-inflated field.

    Heuristic methodology (NOT physics):
    1. Arrival time at detected cells is set to the observation timestamp.
    2. Cells within R_spot (e.g. 1, 2, 4, 7 km) downwind of each detection receive:
         arrival = detection_time + distance / assumed_spread_speed
    3. The field is formed by taking the pointwise minimum with Provider A (teacher_v1 or teacher_v2).
    """

    def __init__(
        self,
        base_teacher_provider: FieldProvider | None = None,
        r_spot_km: float = 4.0,
        assumed_spread_speed_mps: float = 0.5833,  # 35.0 m/min = ~0.583 m/s (~2.1 km/h)
        wind_towards_deg: float = 240.0,  # Southwest downwind heading (azimuth)
        wind_cone_half_angle_deg: float = 60.0,
        horizon_minutes: float = 720.0,
        filter_leakage: bool = False,
        leakage_exclusion_indices: set[int] | None = None,
    ) -> None:
        self.base_provider = base_teacher_provider or TeacherFieldProvider()
        self.r_spot_km = float(r_spot_km)
        self.r_spot_m = self.r_spot_km * 1000.0
        self.spread_speed_m_per_min = assumed_spread_speed_mps * 60.0
        self.wind_towards_deg = wind_towards_deg
        self.wind_cone_rad = np.radians(wind_cone_half_angle_deg)
        self.horizon_minutes = horizon_minutes
        self.filter_leakage = filter_leakage
        self.leakage_exclusion_indices = leakage_exclusion_indices or set()

        # Precompute downwind unit vector
        wind_rad = np.radians(90.0 - wind_towards_deg)  # Math angle from east
        self.wind_u = np.cos(wind_rad)
        self.wind_v = np.sin(wind_rad)
        self.cos_cone = np.cos(self.wind_cone_rad)

        # Coordinate grid cache
        self._gx: np.ndarray | None = None
        self._gy: np.ndarray | None = None
        self._xx: np.ndarray | None = None
        self._yy: np.ndarray | None = None
        self._dx: float = 60.0
        self._dy: float = 60.0

    def get_field(
        self,
        current_step_min: float,
        visible_observations: gpd.GeoDataFrame | None = None,
    ) -> ArrivalField:
        base_field = self.base_provider.get_field(current_step_min, visible_observations)

        if visible_observations is None or len(visible_observations) == 0:
            return base_field

        if self._xx is None or self._gx is None or self._gy is None:
            self._gx = base_field["x"].values
            self._gy = base_field["y"].values
            self._xx, self._yy = np.meshgrid(self._gx, self._gy)
            self._dx = float(self._gx[1] - self._gx[0])
            self._dy = float(self._gy[0] - self._gy[1])

        gx = self._gx
        gy = self._gy
        ny, nx = len(gy), len(gx)

        base_arr = base_field["arrival_q50"].values.copy()
        inflated_arr = np.full((ny, nx), np.inf, dtype=np.float64)

        for obs_idx, obs in visible_observations.iterrows():
            if self.filter_leakage and obs_idx in self.leakage_exclusion_indices:
                continue

            t_obs = float(obs.get("elapsed_minutes_adj", obs.get("elapsed_minutes", 0.0)))
            if t_obs > current_step_min:
                continue

            geom = obs.geometry
            if geom is None:
                continue
            ox_val = float(geom.x) if hasattr(geom, "x") else float(geom.centroid.x)
            oy_val = float(geom.y) if hasattr(geom, "y") else float(geom.centroid.y)

            # Local bounding box within R_spot radius
            c_min = max(0, int((ox_val - self.r_spot_m - gx[0]) / self._dx))
            c_max = min(nx, int((ox_val + self.r_spot_m - gx[0]) / self._dx) + 1)
            r_min = max(0, int((gy[0] - (oy_val + self.r_spot_m)) / self._dy))
            r_max = min(ny, int((gy[0] - (oy_val - self.r_spot_m)) / self._dy) + 1)

            if c_min >= c_max or r_min >= r_max:
                continue

            sub_xx = self._xx[r_min:r_max, c_min:c_max]
            sub_yy = self._yy[r_min:r_max, c_min:c_max]

            dx = sub_xx - ox_val
            dy = sub_yy - oy_val
            dist = np.sqrt(dx * dx + dy * dy)

            within_radius = dist <= self.r_spot_m
            if not np.any(within_radius):
                continue

            dot = dx * self.wind_u + dy * self.wind_v
            is_downwind = (dot >= dist * self.cos_cone) | (dist < 120.0)

            valid_mask = within_radius & is_downwind
            if np.any(valid_mask):
                arrival_candidates = t_obs + (
                    dist[valid_mask] / max(1.0, self.spread_speed_m_per_min)
                )
                sub_inflated = inflated_arr[r_min:r_max, c_min:c_max]
                sub_inflated[valid_mask] = np.minimum(sub_inflated[valid_mask], arrival_candidates)

        # Pointwise minimum with base teacher field
        final_arr = np.minimum(base_arr, inflated_arr)

        ds = xr.Dataset(
            data_vars={
                "arrival_q10": (["y", "x"], final_arr),
                "arrival_q50": (["y", "x"], final_arr),
                "arrival_q90": (["y", "x"], final_arr),
            },
            coords={"x": gx, "y": gy},
            attrs={
                "provider": "observation_inflated",
                "r_spot_km": self.r_spot_km,
                "label": "heuristic, not physics",
                "crs": "EPSG:32610",
                "ignition_time": "2018-11-08T14:33:00Z",
            },
        )
        validate_arrival_field(ds)
        return ds


class SurrogateFieldProvider(FieldProvider):
    """Provider C: Stub for ML surrogate model to be trained in future phases."""

    def __init__(self, weights_path: str | Path | None = None) -> None:
        self.weights_path = Path(weights_path) if weights_path else None

    def get_field(
        self,
        current_step_min: float,
        visible_observations: gpd.GeoDataFrame | None = None,
    ) -> ArrivalField:
        raise NotImplementedError(
            "SurrogateFieldProvider stub: Surrogate model weights not trained in Phase 6A. "
            "Use Provider A (TeacherFieldProvider) or Provider B (ObservationInflatedProvider)."
        )
