"""Unit tests for Phase 3C surrogate enhancements and guards."""

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from wf052.surrogate.evaluate import (
    aggregate_per_bin_and_fire_metrics,
    check_evaluation_guards,
    compute_sample_metrics,
)
from wf052.surrogate.model import WindConditionedUNet


def test_base_rate_initialization_math_and_param_count():
    model = WindConditionedUNet()
    param_count_orig = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count_orig == 4094852

    mean_bp = 0.123928
    applied_bias = model.init_burn_prob_bias(mean_bp)
    expected_bias = math.log(mean_bp / (1.0 - mean_bp))
    assert math.isclose(applied_bias, expected_bias, rel_tol=1e-6)

    # Bias in layer matches applied bias
    layer_bias = float(model.burn_logit_head[2].bias.item())
    assert math.isclose(layer_bias, expected_bias, rel_tol=1e-6)

    # Parameter count strictly preserved
    param_count_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count_after == 4094852


def test_base_rate_initialization_clipping():
    model = WindConditionedUNet()

    # Extreme low clip to 1e-4
    b_low = model.init_burn_prob_bias(1e-6)
    expected_low = math.log(1e-4 / (1.0 - 1e-4))
    assert math.isclose(b_low, expected_low, rel_tol=1e-6)

    # Extreme high clip to 0.5
    b_high = model.init_burn_prob_bias(0.99)
    expected_high = math.log(0.5 / (1.0 - 0.5))  # logit(0.5) == 0.0
    assert math.isclose(b_high, 0.0, abs_tol=1e-6)


def test_evaluation_guards_warnings(capsys):
    # Case 1: Healthy predictions - no warnings
    df_healthy = pd.DataFrame([{
        "pred_bp_std": 0.05,
        "pred_area_p01": 0.20,
        "target_area_mean": 0.15,
    }])
    w_healthy = check_evaluation_guards(df_healthy)
    assert len(w_healthy) == 0

    # Case 2: Flat predictions (std < 1e-3)
    df_flat = pd.DataFrame([{
        "pred_bp_std": 5e-4,
        "pred_area_p01": 0.10,
        "target_area_mean": 0.10,
    }])
    w_flat = check_evaluation_guards(df_flat)
    assert len(w_flat) == 1
    assert "Mean std of predicted burn_prob" in w_flat[0]

    # Case 3: Over-prediction (p>=0.1 area > 5x target area)
    df_over = pd.DataFrame([{
        "pred_bp_std": 0.05,
        "pred_area_p01": 0.80,
        "target_area_mean": 0.10,
    }])
    w_over = check_evaluation_guards(df_over)
    assert len(w_over) == 1
    assert "exceeds 5x target area" in w_over[0]

    captured = capsys.readouterr()
    assert "LOUD EVALUATION GUARD WARNING" in captured.out


def test_aggregate_per_bin_and_fire_metrics():
    df_records = pd.DataFrame([
        {
            "model": "Surrogate (UNet)",
            "fire": "caldor",
            "bin_kmean": 1,
            "brier": 0.02,
            "iou": 0.5,
            "mae_q10": 100.0,
            "mae_q50": 150.0,
            "mae_q90": 200.0,
            "false_unburned_share": 0.1,
            "ordering_violation_rate": 0.0,
            "pred_bp_std": 0.05,
            "pred_area_p01": 0.1,
            "target_area_mean": 0.1,
        },
        {
            "model": "Surrogate (UNet)",
            "fire": "tamarack",
            "bin_kmean": 2,
            "brier": 0.03,
            "iou": 0.6,
            "mae_q10": 120.0,
            "mae_q50": 160.0,
            "mae_q90": 210.0,
            "false_unburned_share": 0.05,
            "ordering_violation_rate": 0.0,
            "pred_bp_std": 0.06,
            "pred_area_p01": 0.12,
            "target_area_mean": 0.12,
        },
    ])

    agg = aggregate_per_bin_and_fire_metrics(df_records, fires=["caldor", "tamarack"])
    # 2 fires + 'ALL' = 3 fire subsets * 5 bins = 15 rows
    assert len(agg) == 15
    assert set(agg["fire"].unique()) == {"caldor", "tamarack", "ALL"}
    assert set(agg["bin"].unique()) == {0, 1, 2, 3, 4}
    assert "mae_q10" in agg.columns
    assert "mae_q50" in agg.columns
    assert "mae_q90" in agg.columns
    assert "flag" in agg.columns


def test_kaggle_today_notebook_structure():
    nb_path = Path("notebooks/kaggle_today.ipynb")
    assert nb_path.exists(), f"Notebook {nb_path} must exist!"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert len(nb["cells"]) == 6, f"Expected exactly 6 cells, got {len(nb['cells'])}"
    for idx, c in enumerate(nb["cells"]):
        assert c["cell_type"] == "code", f"Cell {idx+1} must be code cell"
        assert len(c["source"]) > 0, f"Cell {idx+1} source must not be empty"
