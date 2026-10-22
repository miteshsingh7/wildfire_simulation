"""Adapter module bridging surrogate model predictions to the router's ArrivalField contract.

Contract specification:
- An ArrivalField is an xarray.Dataset with:
    - Variables: arrival_q10, arrival_q50, arrival_q90 (float2D, +inf where fire never arrives)
    - Coordinates: y (northing), x (easting)
    - Attributes: crs (str), ignition_time (ISO timestamp str)
- Unburned threshold: Cells with P(burned) < p_thr are strictly assigned +inf.
- Full-domain tile stitching: Overlapping tiles are stitched using min-arrival time
  across burned cells and max-burn probability.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import xarray as xr

from wf052.routing.contracts import validate_arrival_field


@dataclass(frozen=True)
class TilePrediction:
    """Prediction for a single tile in full-domain inference."""

    quantiles: np.ndarray | torch.Tensor  # (3, H, W) -> [q10, q50, q90] in minutes
    burn_prob: np.ndarray | torch.Tensor  # (H, W) or (1, H, W) in [0, 1]
    r0: int  # Starting row in domain grid
    c0: int  # Starting col in domain grid


def surrogate_output_to_arrival_field(
    quantiles: np.ndarray | torch.Tensor,
    burn_prob: np.ndarray | torch.Tensor,
    x_coords: np.ndarray | Sequence[float],
    y_coords: np.ndarray | Sequence[float],
    crs: str = "EPSG:32610",
    ignition_time: str = "2020-09-06T19:00:00Z",
    p_thr: float = 0.5,
) -> xr.Dataset:
    """Convert surrogate neural model outputs into an xarray ArrivalField satisfying router contract.

    Args:
        quantiles: (3, H, W) tensor/array of arrival quantiles [q10, q50, q90].
        burn_prob: (H, W) or (1, H, W) tensor/array of burn probability in [0, 1].
        x_coords: 1D array of easting coordinates.
        y_coords: 1D array of northing coordinates.
        crs: Coordinate reference system string.
        ignition_time: ISO timestamp string.
        p_thr: Probability threshold below which cells are treated as unburned (+inf).

    Returns:
        Validated xarray.Dataset conforming to ArrivalField contract.
    """
    if isinstance(quantiles, torch.Tensor):
        q_arr = quantiles.detach().cpu().float().numpy()
    else:
        q_arr = np.asarray(quantiles, dtype=np.float32)

    if isinstance(burn_prob, torch.Tensor):
        bp_arr = burn_prob.detach().cpu().float().numpy()
    else:
        bp_arr = np.asarray(burn_prob, dtype=np.float32)

    if bp_arr.ndim == 3 and bp_arr.shape[0] == 1:
        bp_arr = bp_arr[0]

    q10 = q_arr[0].copy()
    q50 = q_arr[1].copy()
    q90 = q_arr[2].copy()

    # Monotonic ordering enforcement: q10 <= q50 <= q90
    q50 = np.maximum(q50, q10)
    q90 = np.maximum(q90, q50)

    # Where P(burned) < p_thr, fire does not arrive -> assign +inf
    unburned_mask = bp_arr < p_thr
    q10[unburned_mask] = np.inf
    q50[unburned_mask] = np.inf
    q90[unburned_mask] = np.inf

    ds = xr.Dataset(
        data_vars={
            "arrival_q10": (("y", "x"), q10.astype(np.float64)),
            "arrival_q50": (("y", "x"), q50.astype(np.float64)),
            "arrival_q90": (("y", "x"), q90.astype(np.float64)),
        },
        coords={
            "y": np.asarray(y_coords, dtype=np.float64),
            "x": np.asarray(x_coords, dtype=np.float64),
        },
        attrs={
            "crs": str(crs),
            "ignition_time": str(ignition_time),
        },
    )

    validate_arrival_field(ds)
    return ds


def stitch_tiles_to_arrival_field(
    tile_predictions: Sequence[TilePrediction],
    domain_shape: tuple[int, int],
    origin_x: float,
    origin_y: float,
    resolution_m: float = 60.0,
    crs: str = "EPSG:32610",
    ignition_time: str = "2020-09-06T19:00:00Z",
    p_thr: float = 0.5,
) -> xr.Dataset:
    """Stitch multiple overlapping tile predictions into a seamless full-domain ArrivalField.

    For overlapping pixels across multiple tiles:
    - Arrival time is the minimum arrival time among tiles that predicted burn (P >= p_thr).
    - Burn probability is the maximum burn probability across all tiles.
    - Cells where max burn probability < p_thr are set to +inf.

    Args:
        tile_predictions: Sequence of TilePrediction instances with bounding coordinates.
        domain_shape: (H_full, W_full) full domain shape.
        origin_x: Westernmost easting coordinate (left edge center).
        origin_y: Northernmost northing coordinate (top edge center).
        resolution_m: Spatial resolution in meters.
        crs: Projected CRS string.
        ignition_time: ISO timestamp string.
        p_thr: Probability threshold for burned status.

    Returns:
        Full-domain validated ArrivalField xarray Dataset.
    """
    h_full, w_full = domain_shape

    full_q10 = np.full((h_full, w_full), np.inf, dtype=np.float64)
    full_q50 = np.full((h_full, w_full), np.inf, dtype=np.float64)
    full_q90 = np.full((h_full, w_full), np.inf, dtype=np.float64)
    full_prob = np.zeros((h_full, w_full), dtype=np.float64)

    for tp in tile_predictions:
        if isinstance(tp.quantiles, torch.Tensor):
            q_tile = tp.quantiles.detach().cpu().float().numpy()
        else:
            q_tile = np.asarray(tp.quantiles, dtype=np.float64)

        if isinstance(tp.burn_prob, torch.Tensor):
            bp_tile = tp.burn_prob.detach().cpu().float().numpy()
        else:
            bp_tile = np.asarray(tp.burn_prob, dtype=np.float64)

        if bp_tile.ndim == 3 and bp_tile.shape[0] == 1:
            bp_tile = bp_tile[0]

        th, tw = q_tile.shape[1], q_tile.shape[2]
        r0 = tp.r0
        c0 = tp.c0
        r1 = min(h_full, r0 + th)
        c1 = min(w_full, c0 + tw)

        # Slice sub-regions
        dr = r1 - r0
        dc = c1 - c0
        if dr <= 0 or dc <= 0:
            continue

        tile_q10 = q_tile[0, :dr, :dc]
        tile_q50 = q_tile[1, :dr, :dc]
        tile_q90 = q_tile[2, :dr, :dc]
        tile_bp = bp_tile[:dr, :dc]

        target_slice = (slice(r0, r1), slice(c0, c1))

        # Update max probability
        full_prob[target_slice] = np.maximum(full_prob[target_slice], tile_bp)

        # In fire front propagation, earliest arrival time wins
        burned_here = tile_bp >= p_thr
        full_q10[target_slice] = np.where(
            burned_here,
            np.minimum(full_q10[target_slice], tile_q10),
            full_q10[target_slice],
        )
        full_q50[target_slice] = np.where(
            burned_here,
            np.minimum(full_q50[target_slice], tile_q50),
            full_q50[target_slice],
        )
        full_q90[target_slice] = np.where(
            burned_here,
            np.minimum(full_q90[target_slice], tile_q90),
            full_q90[target_slice],
        )

    # Clean unburned cells
    unburned = full_prob < p_thr
    full_q10[unburned] = np.inf
    full_q50[unburned] = np.inf
    full_q90[unburned] = np.inf

    # Construct coordinates
    # In GIS convention, y decreases from north to south
    x_coords = origin_x + np.arange(w_full) * resolution_m
    y_coords = origin_y - np.arange(h_full) * resolution_m

    ds = xr.Dataset(
        data_vars={
            "arrival_q10": (("y", "x"), full_q10),
            "arrival_q50": (("y", "x"), full_q50),
            "arrival_q90": (("y", "x"), full_q90),
        },
        coords={
            "y": y_coords,
            "x": x_coords,
        },
        attrs={
            "crs": str(crs),
            "ignition_time": str(ignition_time),
        },
    )

    validate_arrival_field(ds)
    return ds
