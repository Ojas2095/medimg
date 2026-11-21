"""Unit tests for Probability Calibration (Temperature Scaling) and Calibration Metrics."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

from medimg.calibration import TemperatureScaler, compute_calibration_metrics


def test_temperature_scaler_initialization():
    scaler = TemperatureScaler(initial_temperature=1.5)
    assert abs(float(scaler.temperature.item()) - 1.5) < 1e-4

    # Test scaling
    logits = torch.tensor([1.5, -1.5])
    scaled = scaler(logits)
    assert torch.allclose(scaled, torch.tensor([1.0, -1.0]), atol=1e-4)


def test_calibrate_probability_types():
    scaler = TemperatureScaler(initial_temperature=2.0)

    # Float logit
    p_float = scaler.calibrate_probability(0.0)
    assert abs(p_float - 0.5) < 1e-5

    # NumPy array
    arr = np.array([0.0, 2.0, -2.0])
    p_arr = scaler.calibrate_probability(arr)
    assert isinstance(p_arr, np.ndarray)
    assert abs(p_arr[0] - 0.5) < 1e-5
    assert p_arr[1] > 0.5
    assert p_arr[2] < 0.5

    # PyTorch Tensor
    t = torch.tensor([0.0, 2.0, -2.0])
    p_t = scaler.calibrate_probability(t)
    assert isinstance(p_t, torch.Tensor)
    assert abs(p_t[0].item() - 0.5) < 1e-5


def test_temperature_scaler_fitting():
    torch.manual_seed(42)
    labels = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    logits = torch.tensor([1.5, 2.0, -0.5, -1.2, -1.8, 0.8, 0.6, -1.5])

    criterion = nn.BCEWithLogitsLoss()
    loss_before = criterion(logits, labels).item()

    scaler = TemperatureScaler(initial_temperature=1.0)
    opt_t = scaler.fit(logits, labels, max_iter=30, lr=0.05)

    loss_after = criterion(logits / opt_t, labels).item()

    assert not torch.isnan(scaler.temperature)
    assert 0.05 < opt_t < 10.0
    assert loss_after <= loss_before + 1e-4


def test_calibration_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        calib_file = Path(tmpdir) / "calib.json"
        scaler = TemperatureScaler(initial_temperature=1.85)
        scaler.save(calib_file, metadata={"model": "resnet18", "val_brier": 0.042})

        assert calib_file.exists()
        with open(calib_file, "r") as f:
            data = json.load(f)
        assert data["method"] == "temperature_scaling"
        assert abs(data["temperature"] - 1.85) < 1e-4
        assert data["metadata"]["model"] == "resnet18"

        loaded = TemperatureScaler.load(calib_file)
        assert abs(float(loaded.temperature.item()) - 1.85) < 1e-4


def test_calibration_metrics_computation():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.85, 0.15])

    metrics = compute_calibration_metrics(y_true, y_prob, n_bins=5)

    assert "brier_score" in metrics
    assert "ece" in metrics
    assert "mce" in metrics
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["mce"] <= 1.0
    assert metrics["num_samples"] == 8

    # Test length mismatch error
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_calibration_metrics([1, 0], [0.5])
