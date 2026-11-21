"""Grad-CAM interpretability and systematic clinical error analysis.

1. Computes Grad-CAM heatmaps on the final convolutional layer of ResNet18.
2. Integrates per-image interpretability into the workstation inference pipeline.
3. Audits false negatives and false positives on the held-out test cohort with hypothesis-based categorization.
4. Evaluates demographic and anatomical subgroup performance.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .common import device, save_json, seed_all
from .config import CONFIG
from .datasets import MedDataset
from .metrics import auroc, confusion, sens_spec
from .models import classifier

DATA_ROOT = "data/ham10000"
RUNS_DIR = "runs/error_analysis"
OUT_DIR = "out/error_analysis"
DEFAULT_THRESHOLD = CONFIG.default_triage_threshold

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN explainability.

    Extracts gradients flowing into the final convolutional feature maps to produce
    a localized 2D visual attribution heatmap highlighting salient diagnostic regions.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(
        self,
        x_tensor: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, float]:
        """Compute Grad-CAM heatmap for the given input tensor.

        Args:
            x_tensor: Model input tensor of shape (1, C, H, W).
            target_size: Optional (H, W) spatial dimensions for output heatmap.

        Returns:
            cam: 2D numpy array in range [0, 1] of shape target_size.
            prob: Sigmoid classification probability.
        """
        self.model.eval()

        # Ensure gradient tracking is enabled even if called within outer torch.no_grad()
        with torch.enable_grad():
            x_var = x_tensor.clone().detach().requires_grad_(True)
            self.model.zero_grad()
            out = self.model(x_var)
            out.backward(torch.ones_like(out))

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks failed to capture activations or gradients.")

        # Global average pooling of gradients over spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        spatial_size = target_size if target_size is not None else (x_tensor.shape[2], x_tensor.shape[3])
        cam = F.interpolate(cam, size=spatial_size, mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().detach().numpy()

        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        prob = float(torch.sigmoid(out).item())
        return cam_np, prob

    def close(self) -> None:
        """Remove forward and backward hooks."""
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def overlay_heatmap(img_pil: Image.Image, cam: np.ndarray, colormap: str = "jet", alpha: float = 0.45) -> Image.Image:
    """Blend a Grad-CAM heatmap onto a PIL image at native resolution."""
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(cam)[:, :, :3]
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(img_pil.size, Image.BILINEAR)
    blended = Image.blend(img_pil.convert("RGB"), heatmap_pil, alpha=alpha)
    return blended


def run_error_analysis():
    """Execute comprehensive false-negative, false-positive, and subgroup error audits."""
    dev = device()
    seed_all(42)
    te = MedDataset(DATA_ROOT, "test", size=128, augment=False)
    meta_df = pd.read_csv(os.path.join(DATA_ROOT, "metadata.csv")).set_index("image_id")

    model = classifier("resnet18", pretrained=True).to(dev)
    ckpt_path = "runs/checkpoints/cls_resnet18.pt"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=True))
    target_layer = model.layer4[-1]

    print("\n========================================================")
    print("TASK D.4: Grad-CAM Error Analysis & Subgroup Audit")
    print("========================================================")

    records = []
    print(f"Evaluating {len(te)} test images with Grad-CAM...")
    with GradCAM(model, target_layer) as gradcam:
        for i in range(len(te)):
            r = te.df.iloc[i]
            iid = r.image_id
            raw_img = Image.open(os.path.join(te.root, r.image_path)).convert("RGB")
            resized = raw_img.resize((128, 128), Image.BILINEAR)
            x = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
            x_t = torch.from_numpy(x)[None].to(dev)

            cam, prob = gradcam.generate(x_t, target_size=(raw_img.size[1], raw_img.size[0]))
            gt = int(r.label)
            meta_row = meta_df.loc[iid] if iid in meta_df.index else None

            records.append({
                "image_id": iid,
                "lesion_id": str(r.get("patient_id", r.get("lesion_id", ""))),
                "label": gt,
                "score": prob,
                "pred": int(prob >= DEFAULT_THRESHOLD),
                "age": float(meta_row["age"]) if meta_row is not None and not pd.isna(meta_row.get("age")) else np.nan,
                "sex": str(meta_row.get("sex", "unknown")) if meta_row is not None else "unknown",
                "localization": str(meta_row.get("localization", "unknown")) if meta_row is not None else "unknown",
                "dx_7class": str(meta_row.get("dx_7class", "unknown")) if meta_row is not None else "unknown",
                "cam": cam,
                "raw_img": raw_img
            })

    df = pd.DataFrame(records)
    fns = df[(df.label == 1) & (df.pred == 0)].sort_values("score", ascending=True)
    fps = df[(df.label == 0) & (df.pred == 1)].sort_values("score", ascending=False)

    print(f"Total Test Set: {len(df)} | False Negatives: {len(fns)} | False Positives: {len(fps)}")

    # Save top 20 worst FNs with hypothesis-based categorization
    worst_fn_list = []
    for rank, (_, row) in enumerate(fns.head(20).iterrows(), start=1):
        blended = overlay_heatmap(row["raw_img"], row["cam"])
        p = os.path.join(OUT_DIR, f"fn_{rank:02d}_{row['image_id']}.png")
        blended.save(p)

        # Grounded hypothesis based on metadata and clinical dermatology literature
        loc = row["localization"]
        dx = row["dx_7class"]
        if loc in ("face", "neck"):
            hypothesized_cause = "Hypothesis: Facial sun-exposure / solar lentigo masking border irregularity."
        elif row["score"] < 0.20:
            hypothesized_cause = "Hypothesis: Amelanotic or hypopigmented variant exhibiting minimal dark melanin contrast."
        else:
            hypothesized_cause = "Hypothesis: Border blending with surrounding normal dermis or subtle network atypicality."

        worst_fn_list.append({
            "rank": rank, "image_id": row["image_id"], "score": float(row["score"]),
            "threshold": DEFAULT_THRESHOLD, "dx_7class": dx,
            "age": row["age"], "sex": row["sex"], "localization": loc,
            "hypothesized_failure_cause": hypothesized_cause, "overlay_image": p
        })

    # Save top 20 worst FPs
    worst_fp_list = []
    for rank, (_, row) in enumerate(fps.head(20).iterrows(), start=1):
        blended = overlay_heatmap(row["raw_img"], row["cam"])
        p = os.path.join(OUT_DIR, f"fp_{rank:02d}_{row['image_id']}.png")
        blended.save(p)

        dx = row["dx_7class"]
        if dx == "bkl":
            hypothesized_cause = "Hypothesis: Benign keratosis (bkl) presenting hyperkeratotic pseudonetwork resembling melanoma."
        elif dx == "bcc":
            hypothesized_cause = "Hypothesis: Pigmented basal cell carcinoma (bcc) exhibiting arborizing telangiectasia and pigment."
        elif dx == "nv":
            hypothesized_cause = "Hypothesis: Dysplastic / Clark nevus with architectural disorder triggering melanoma feature maps."
        else:
            hypothesized_cause = "Hypothesis: Peripheral vignetting or dark frame artifact mimicking dark border notch."

        worst_fp_list.append({
            "rank": rank, "image_id": row["image_id"], "score": float(row["score"]),
            "threshold": DEFAULT_THRESHOLD, "dx_7class": dx,
            "age": row["age"], "sex": row["sex"], "localization": row["localization"],
            "hypothesized_failure_cause": hypothesized_cause, "overlay_image": p
        })

    error_report = {
        "summary": {
            "total_test_samples": len(df),
            "melanoma_positives": int((df.label == 1).sum()),
            "benign_negatives": int((df.label == 0).sum()),
            "false_negatives_count": len(fns),
            "false_positives_count": len(fps),
            "disclaimer": "Failure causes are clinical hypotheses based on metadata and dermoscopy literature, not proven ground-truth causation."
        },
        "worst_false_negatives": worst_fn_list,
        "worst_false_positives": worst_fp_list
    }
    save_json(error_report, f"{RUNS_DIR}/worst_errors.json")

    # ---------------- Subgroup Demographic & Anatomical Analysis ----------------
    print("\nAuditing subgroup performance across biological sex and anatomical localization...")
    subgroups = {}

    # Biological Sex
    for s in ["male", "female"]:
        sub_df = df[df.sex == s]
        if len(sub_df) > 10 and sub_df.label.nunique() > 1:
            auc = auroc(sub_df.label.values, sub_df.score.values)
            se, sp = sens_spec(sub_df.label.values, sub_df.score.values >= DEFAULT_THRESHOLD)
            subgroups[f"sex_{s}"] = {"count": len(sub_df), "auroc": float(auc), "sensitivity": float(se), "specificity": float(sp)}
            print(f"  Sex [{s:6s}]: N={len(sub_df):3d} | AUROC={auc:.4f} | Sens={se:.4f} | Spec={sp:.4f}")

    # Anatomical Localization
    for loc in ["back", "lower extremity", "trunk", "upper extremity", "face"]:
        sub_df = df[df.localization == loc]
        if len(sub_df) >= 15 and sub_df.label.nunique() > 1:
            auc = auroc(sub_df.label.values, sub_df.score.values)
            se, sp = sens_spec(sub_df.label.values, sub_df.score.values >= DEFAULT_THRESHOLD)
            subgroups[f"loc_{loc}"] = {"count": len(sub_df), "auroc": float(auc), "sensitivity": float(se), "specificity": float(sp)}
            print(f"  Loc [{loc:15s}]: N={len(sub_df):3d} | AUROC={auc:.4f} | Sens={se:.4f} | Spec={sp:.4f}")

    save_json(subgroups, f"{RUNS_DIR}/subgroup_parity.json")

    # Generate subgroup visualization bar plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [k.replace("sex_", "Sex: ").replace("loc_", "Site: ") for k in subgroups.keys()]
    aurocs = [v["auroc"] for v in subgroups.values()]
    ax.barh(labels, aurocs, color="#0284c7")
    ax.set_xlim(0.6, 1.0)
    ax.axvline(0.90, color="red", linestyle="--", label="Target AUROC (0.90)")
    ax.set_xlabel("Held-Out Test AUROC")
    ax.set_title("Subgroup Performance Parity (HAM10000 Test Split)")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/subgroup_auroc.png", dpi=150)
    plt.close()

    print(f"\nSaved Grad-CAM error analysis outputs to {OUT_DIR}/ and {RUNS_DIR}/")
    return error_report


if __name__ == "__main__":
    run_error_analysis()
