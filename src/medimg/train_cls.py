"""Train ResNet or GoogLeNet classifier (lesion present / absent).

Selects best epoch on VAL AUROC; fits temperature scaling calibration on validation split;
evaluates held-out TEST split strictly once at the end using the validation-calibrated operating threshold.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .calibration import TemperatureScaler, compute_calibration_metrics
from .checkpoints import CheckpointMetadata, clone_state_dict, save_checkpoint
from .common import device, run_meta, save_json, seed_all
from .config import CONFIG
from .datasets import MedDataset, collate
from .metrics import auroc, bootstrap_ci, confusion, sens_spec, threshold_at_sensitivity, time_fn
from .models import classifier


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, dev: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract raw logits and true binary ground truth labels."""
    model.eval()
    ys, logits = [], []
    for b in loader:
        out = model(b["x"].to(dev)).squeeze(-1)
        logits.extend(out.cpu().numpy().tolist())
        ys.extend(b["label"].numpy().tolist())
    return np.asarray(ys, dtype=np.int64), np.asarray(logits, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Train Medical Imaging Lesion Classifier")
    ap.add_argument("--root", default="data/synth", help="Dataset root directory")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50", "googlenet"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--balance", action="store_true", help="Oversample minority melanoma class")
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/cls")
    a = ap.parse_args()

    seed_all(a.seed)
    dev = device()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = MedDataset(a.root, "train", a.size, augment=not a.no_aug)
    va = MedDataset(a.root, "val", a.size, augment=False)
    te = MedDataset(a.root, "test", a.size, augment=False)

    sampler = None
    if a.balance:
        y = tr.df.label.values
        w = np.where(y == 1, 1.0 / max(1, y.sum()), 1.0 / max(1, (1 - y).sum()))
        sampler = WeightedRandomSampler(w, len(y), replacement=True)

    make_loader = lambda d, **k: DataLoader(d, batch_size=a.bs, collate_fn=collate, **k)
    trl = make_loader(tr, shuffle=(sampler is None), sampler=sampler)
    val = make_loader(va, shuffle=False)
    tel = make_loader(te, shuffle=False)

    model = classifier(a.arch, pretrained=a.pretrained, n_out=1).to(dev)

    # Class-weighted loss
    pos_count = max(1, int(tr.df.label.sum()))
    neg_count = max(1, len(tr) - pos_count)
    pos_weight = torch.tensor([neg_count / pos_count], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    best_val_auroc = -1.0
    best_state = None
    history = []

    print(f"Beginning training {a.arch} on {dev} for {a.epochs} epochs (pos_weight={pos_weight.item():.2f})...")
    for ep in range(a.epochs):
        model.train()
        train_loss_total = 0.0
        for b in trl:
            opt.zero_grad()
            logits = model(b["x"].to(dev)).squeeze(-1)
            loss = lossf(logits, b["label"].to(dev).float())
            loss.backward()
            opt.step()
            train_loss_total += loss.item() * len(b["label"])

        y_val, logit_val = predict_logits(model, val, dev)
        prob_val = 1.0 / (1.0 + np.exp(-logit_val))
        val_auc = auroc(y_val, prob_val)

        epoch_rec = {
            "epoch": ep + 1,
            "train_loss": train_loss_total / len(tr),
            "val_auroc": val_auc
        }
        history.append(epoch_rec)
        print(f"Epoch {ep+1:02d}/{a.epochs:02d} | Train Loss: {epoch_rec['train_loss']:.4f} | Val AUROC: {val_auc:.4f}")

        if val_auc > best_val_auroc:
            best_val_auroc = val_auc
            best_state = clone_state_dict(model)

    if best_state is None:
        best_state = clone_state_dict(model)

    # Load best state
    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    # Fit Probability Calibration on Validation Logits
    y_val, logit_val = predict_logits(model, val, dev)
    scaler = TemperatureScaler(initial_temperature=1.0)
    best_temp = scaler.fit(torch.from_numpy(logit_val), torch.from_numpy(y_val))
    calib_params_path = out_dir / f"{a.arch}_calibration.json"
    scaler.save(calib_params_path, metadata={"val_auroc": best_val_auroc})
    print(f"Fitted Temperature Scaling Calibration: T = {best_temp:.4f} (Saved to {calib_params_path.name})")

    # Operating Point: Select threshold on validation set strictly to achieve Sensitivity >= 95%
    prob_val_calibrated = scaler.calibrate_probability(logit_val)
    thr = threshold_at_sensitivity(y_val, prob_val_calibrated, target_sens=CONFIG.target_sensitivity)
    print(f"Selected Validation Operating Threshold: {thr:.4f} (Target Sens >= {CONFIG.target_sensitivity*100:.0f}%)")

    # Evaluate on held-out TEST set (Evaluated once)
    y_test, logit_test = predict_logits(model, tel, dev)
    prob_test_calibrated = scaler.calibrate_probability(logit_test)

    test_auc = auroc(y_test, prob_test_calibrated)
    test_ci = bootstrap_ci(auroc, y_test, prob_test_calibrated, n=1000, seed=a.seed)
    se, sp = sens_spec(y_test, prob_test_calibrated >= thr)
    cm = confusion(y_test, prob_test_calibrated >= thr)
    calib_metrics = compute_calibration_metrics(y_test, prob_test_calibrated)

    # Save Checkpoint with full provenance metadata
    ckpt_path = out_dir / "best.pt"
    meta = CheckpointMetadata(
        architecture=a.arch,
        sha256="",
        num_parameters=sum(p.numel() for p in model.parameters()),
        input_size=a.size,
        calibration_method="temperature_scaling",
        calibration_temperature=best_temp,
        operating_threshold=thr,
        target_sensitivity=CONFIG.target_sensitivity,
        dataset_provenance=str(a.root),
        notes=f"Trained for {a.epochs} epochs with pos_weight={pos_weight.item():.2f}."
    )
    save_checkpoint(best_state, ckpt_path, metadata=meta)

    # Inference Latency
    x1 = te[0]["x"][None].to(dev)
    lat = time_fn(lambda: model(x1), warmup=3, runs=25)

    report = {
        "architecture": a.arch,
        "test_auroc": test_auc,
        "test_auroc_95ci": test_ci,
        "operating_threshold": thr,
        "test_sensitivity": se,
        "test_specificity": sp,
        "confusion_matrix": cm,
        "calibration": calib_metrics,
        "temperature": best_temp,
        "latency_ms_median_p95": lat,
        "history": history,
        "meta": run_meta(a)
    }
    save_json(report, str(out_dir / "report.json"))

    print("\n========================================================")
    print(f"Final Test Evaluation for {a.arch}:")
    print(f"  AUROC:       {test_auc:.4f} [95% CI: {test_ci[0]:.4f}, {test_ci[1]:.4f}]")
    print(f"  Sensitivity: {se:.4f} (at val-selected threshold {thr:.3f})")
    print(f"  Specificity: {sp:.4f}")
    print(f"  ECE:         {calib_metrics['ece']:.4f} | Brier: {calib_metrics['brier_score']:.4f}")
    print("========================================================")


if __name__ == "__main__":
    main()
