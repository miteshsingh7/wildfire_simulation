"""Tests for wildfire simulation engines and evaluation metrics."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from wf052.eval.comparison import EvaluationMetrics, evaluate_simulation
from wf052.sim.cell2fire_engine import FBFM40_TO_FBP
from wf052.sim.pytorchfire_engine import FBFM40_TO_P_VEG, load_environmental_tensors


def test_fuel_mappings_consistency() -> None:
    """Verify non-burnable and burnable fuel models are consistently mapped."""
    # Non-burnable models must be strictly non-burnable in both engines
    non_burnable_codes = [91, 92, 93, 98, 99]
    for code in non_burnable_codes:
        assert FBFM40_TO_P_VEG[code] < 0.0, f"Code {code} must have negative p_veg in PyTorchFire"
        gval, fcode = FBFM40_TO_FBP[code]
        assert fcode.strip() == "NF", f"Code {code} must map to NF in Cell2Fire"
        assert gval == 101

    # Key Sierra foothill flammable fuels must be present
    assert 102 in FBFM40_TO_P_VEG  # GR2
    assert 122 in FBFM40_TO_P_VEG  # GS2
    assert 145 in FBFM40_TO_P_VEG  # SH5
    assert 165 in FBFM40_TO_P_VEG  # TU5
    assert 186 in FBFM40_TO_P_VEG  # TL6


def test_environmental_tensors_loading() -> None:
    """Verify environmental tensor shapes and bounds."""
    tiles_dir = Path("data/tiles")
    if not (tiles_dir / "fbfm40.tif").exists():
        pytest.skip("Tiles directory not populated")

    device = torch.device("cpu")
    tensors, shape = load_environmental_tensors(tiles_dir, device)

    assert shape == (1128, 1124)
    assert tensors["p_veg"].shape == (1128, 1124)
    assert tensors["p_den"].shape == (1128, 1124)
    assert tensors["slope"].shape == (1128, 1124, 3, 3)
    assert tensors["elev"].shape == (1128, 1124)


def test_evaluation_metrics_types() -> None:
    """Verify evaluation metrics calculation on synthetic arrival raster."""
    with open("configs/phase0.yaml") as f:
        cfg = yaml.safe_load(f)

    # Synthetic raster: 1128x1124 with only ignition area burned
    synth_arr = np.full((1128, 1124), np.nan, dtype=np.float32)
    synth_arr[530:545, 865:875] = 60.0  # burned at 60 min

    metrics = evaluate_simulation(
        arrival_raster=synth_arr,
        config=cfg,
        engine_name="TestEngine",
        seed=42,
        wall_clock_seconds=1.23,
    )

    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.engine == "TestEngine"
    assert metrics.seed == 42
    assert metrics.wall_clock_seconds == 1.23
    assert 0.0 <= metrics.unburned_fraction_pct <= 100.0
    assert metrics.horizon_3h.horizon_hours == 3.0
    assert metrics.horizon_6h.horizon_hours == 6.0
    assert metrics.horizon_12h.horizon_hours == 12.0
