"""Train torchvision RetinaNet for lesion detection.

Selects best epoch on validation AP50; evaluates test split once at the end.
Saves verified checkpoint with metadata and independent state snapshots.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoints import CheckpointMetadata, clone_state_dict, save_checkpoint
from .common import device, run_meta, save_json, seed_all
from .config import CONFIG
from .datasets import MedDataset, collate
from .metrics import ap50, time_fn
from .models import retinanet


@torch.no_grad()
def evaluate_det(model: torch.nn.Module, loader: DataLoader, dev: str) -> float:
    """Evaluate mean Average Precision at IoU=0.50 (AP50)."""
    model.eval()
    preds, gts = [], []
    for b in loader:
        outs = model(list(b["x"].to(dev)))
        for o, g in zip(outs, b["boxes"]):
            preds.append((o["boxes"].cpu().numpy(), o["scores"].cpu().numpy()))
            gts.append(g.numpy())
    return float(ap50(preds, gts))


def main():
    ap = argparse.ArgumentParser(description="Train RetinaNet Lesion Detector")
    ap.add_argument("--root", default="data/synth", help="Dataset root directory")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--pretrained_backbone", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/det")
    a = ap.parse_args()

    seed_all(a.seed)
    dev = device()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    make_loader = lambda split, **kwargs: DataLoader(
        MedDataset(a.root, split, a.size, **kwargs),
        batch_size=a.bs,
        collate_fn=collate,
        shuffle=(split == "train")
    )
    trl = make_loader("train", augment=True)
    val = make_loader("val")
    tel = make_loader("test")

    model = retinanet(num_classes=2, pretrained_backbone=a.pretrained_backbone).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    best_val_ap = -1.0
    best_state = None
    history = []

    print(f"Beginning training RetinaNet on {dev} for {a.epochs} epochs...")
    for ep in range(a.epochs):
        model.train()
        train_loss_total = 0.0
        for b in trl:
            targets = [
                {
                    "boxes": bx.to(dev),
                    "labels": torch.ones(len(bx), dtype=torch.int64, device=dev)
                }
                for bx in b["boxes"]
            ]
            loss_dict = model(list(b["x"].to(dev)), targets)
            loss = sum(loss_dict.values())

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_total += loss.item()

        val_ap = evaluate_det(model, val, dev)
        epoch_rec = {
            "epoch": ep + 1,
            "train_loss": train_loss_total / max(1, len(trl)),
            "val_ap50": val_ap
        }
        history.append(epoch_rec)
        print(f"Epoch {ep+1:02d}/{a.epochs:02d} | Train Loss: {epoch_rec['train_loss']:.4f} | Val AP50: {val_ap:.4f}")

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = clone_state_dict(model)

    if best_state is None:
        best_state = clone_state_dict(model)

    # Save best checkpoint with metadata
    ckpt_path = out_dir / "best.pt"
    meta = CheckpointMetadata(
        architecture="retinanet_resnet50_fpn",
        sha256="",
        num_parameters=sum(p.numel() for p in model.parameters()),
        input_size=a.size,
        notes=f"Trained for {a.epochs} epochs on {a.root}."
    )
    save_checkpoint(best_state, ckpt_path, metadata=meta)

    # Evaluate on held-out test split strictly once
    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    test_ap = evaluate_det(model, tel, dev)

    x1 = next(iter(tel))["x"][:1].to(dev)
    lat = time_fn(lambda: model(list(x1)), warmup=3, runs=20)

    report = {
        "architecture": "retinanet_resnet50_fpn",
        "test_ap50": test_ap,
        "latency_ms_median_p95": lat,
        "history": history,
        "meta": run_meta(a)
    }
    save_json(report, str(out_dir / "report.json"))
    print(f"\nFinal Test Results for RetinaNet: AP50 = {test_ap:.4f}, Latency = {lat[0]:.2f} ms")


if __name__ == "__main__":
    main()
