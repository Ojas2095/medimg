"""Probability calibration module implementing empirical Temperature Scaling.

Calibrates raw neural network logits using an optimization pass on held-out
validation data to align predicted confidence with observed empirical frequency.
Computes Expected Calibration Error (ECE), Brier Score, and Maximum Calibration Error (MCE).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for binary classification logits:

    p_calibrated = sigmoid(logit / T)
    where T > 0 is optimized strictly on held-out validation data via NLL minimization.
    """

    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([float(initial_temperature)]))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        t = torch.clamp(self.temperature, min=1e-2, max=50.0)
        return logits / t

    def calibrate_probability(self, logit: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Convert logit to calibrated probability."""
        t_val = float(torch.clamp(self.temperature, min=1e-2, max=50.0).item())
        if isinstance(logit, (float, int)):
            scaled = logit / t_val
            return 1.0 / (1.0 + np.exp(-scaled))
        elif isinstance(logit, np.ndarray):
            scaled = logit / t_val
            return 1.0 / (1.0 + np.exp(-scaled))
        else:
            with torch.no_grad():
                return torch.sigmoid(self.forward(logit))

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50, lr: float = 0.05) -> float:
        """Fit optimal temperature on validation set via NLL (BCE) minimization."""
        val_logits = val_logits.detach().view(-1)
        val_labels = val_labels.detach().view(-1).float()

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            with torch.no_grad():
                self.temperature.clamp_(min=1e-2, max=50.0)
            scaled = self.forward(val_logits)
            loss = criterion(scaled, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        with torch.no_grad():
            self.temperature.clamp_(min=1e-2, max=50.0)
        return float(self.temperature.item())

    def save(self, path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save fitted calibration parameters to JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "method": "temperature_scaling",
            "temperature": float(self.temperature.item()),
            "metadata": metadata or {}
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> TemperatureScaler:
        """Load fitted temperature scaler from JSON file."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Calibration parameter file not found: {target}")
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        scaler = cls(initial_temperature=data.get("temperature", 1.0))
        return scaler


def compute_calibration_metrics(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    n_bins: int = 10
) -> Dict[str, Any]:
    """Compute binned calibration metrics: ECE, MCE, Brier score, and bin-level points."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    if len(y_true) != len(y_prob):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)}, len(y_prob)={len(y_prob)}")

    brier_score = float(np.mean((y_prob - y_true) ** 2))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_accs = []
    bin_confs = []
    bin_counts = []
    ece = 0.0
    mce = 0.0
    N = len(y_true)

    for i in range(n_bins):
        if i == n_bins - 1:
            idx = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            idx = (y_prob >= bins[i]) & (y_prob < bins[i + 1])

        count = int(np.sum(idx))
        bin_counts.append(count)

        if count > 0:
            acc = float(np.mean(y_true[idx]))
            conf = float(np.mean(y_prob[idx]))
            bin_accs.append(acc)
            bin_confs.append(conf)
            bin_error = abs(acc - conf)
            ece += (count / N) * bin_error
            mce = max(mce, bin_error)
        else:
            bin_accs.append(float("nan"))
            bin_confs.append(float(bin_centers[i]))

    return {
        "brier_score": brier_score,
        "ece": float(ece),
        "mce": float(mce),
        "bin_centers": bin_centers.tolist(),
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "num_samples": N
    }
