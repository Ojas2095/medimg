"""Train U-Net or ResNetUNet segmentation models.

Selects best epoch on validation Dice; evaluates test split once at the end.
Saves verified checkpoint with architecture metadata and independent state snapshots.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoints import CheckpointMetadata, clone_state_dict, save_checkpoint
from .common import device, run_meta, save_json, seed_all
from .config import CONFIG
from .datasets import MedDataset, collate
from .metrics import dice, iou, time_fn
from .models import ResNetUNet, UNet, dice_bce_loss


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, dev: str) -> Tuple[float, float]:
    """Evaluate mean Dice score and mean IoU across a DataLoader."""
    model.eval()
    d_list, j_list = [], []
    for b in loader:
        logits = model(b["x"].to(dev))
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        gts = b["mask"].numpy()
        for pi, gi in zip(preds, gts):
            d_list.append(dice(pi, gi))
            j_list.append(iou(pi, gi))
    return float(np.mean(d_list)), float(np.mean(j_list))


def build_segmentor_model(arch: str = "resnet_unet", pretrained: bool = True, base: int = 16) -> torch.nn.Module:
    """Instantiate segmentation architecture based on chosen identifier."""
    if arch in ("resnet_unet", "resnetunet"):
        return ResNetUNet(pretrained=pretrained)
    elif arch == "unet":
        return UNet(in_ch=3, base=base)
    else:
        raise ValueError(f"Unsupported segmentation architecture: {arch}. Choose 'resnet_unet' or 'unet'.")


def main():
    ap = argparse.ArgumentParser(description="Train Medical Image Segmentation Model")
    ap.add_argument("--root", default="data/synth", help="Dataset root directory")
    ap.add_argument("--arch", default="resnet_unet", choices=["resnet_unet", "unet"], help="Segmentation architecture")
    ap.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained encoder weights")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--base", type=int, default=16, help="Base feature channels for standard UNet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/seg")
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

    model = build_segmentor_model(arch=a.arch, pretrained=a.pretrained, base=a.base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    best_val_dice = -1.0
    best_state = None
    history = []

    print(f"Beginning training {a.arch} on {dev} for {a.epochs} epochs...")
    for ep in range(a.epochs):
        model.train()
        train_loss_total = 0.0
        for b in trl:
            opt.zero_grad()
            loss = dice_bce_loss(model(b["x"].to(dev)), b["mask"].to(dev))
            loss.backward()
            opt.step()
            train_loss_total += loss.item() * len(b["mask"])

        val_dice, val_iou = evaluate(model, val, dev)
        epoch_rec = {
            "epoch": ep + 1,
            "train_loss": train_loss_total / len(trl.dataset),
            "val_dice": val_dice,
            "val_iou": val_iou
        }
        history.append(epoch_rec)
        print(f"Epoch {ep+1:02d}/{a.epochs:02d} | Train Loss: {epoch_rec['train_loss']:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            # Use independent detached clone to prevent CPU in-memory mutation
            best_state = clone_state_dict(model)

    if best_state is None:
        best_state = clone_state_dict(model)

    # Save best checkpoint with metadata
    ckpt_path = out_dir / "best.pt"
    meta = CheckpointMetadata(
        architecture=a.arch,
        sha256="",
        num_parameters=sum(p.numel() for p in model.parameters()),
        input_size=a.size,
        notes=f"Trained for {a.epochs} epochs on {a.root}."
    )
    save_checkpoint(best_state, ckpt_path, metadata=meta)

    # Load best state for held-out test evaluation
    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    test_dice, test_iou = evaluate(model, tel, dev)

    x1 = next(iter(tel))["x"][:1].to(dev)
    lat = time_fn(lambda: model(x1), warmup=3, runs=20)

    report = {
        "architecture": a.arch,
        "test_dice": test_dice,
        "test_iou": test_iou,
        "latency_ms_median_p95": lat,
        "history": history,
        "meta": run_meta(a)
    }
    save_json(report, str(out_dir / "report.json"))
    print(f"\nFinal Test Results for {a.arch}: Dice = {test_dice:.4f}, IoU = {test_iou:.4f}")


if __name__ == "__main__":
    main()
