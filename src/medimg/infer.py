"""Triage and lesion analysis pipeline inference entry point.

Run with:
    python -m medimg infer --image <path> --out <dir>

Produces:
1. Triage decision (HIGH/LOW PRIORITY) at validation-selected threshold (Sens >= 95%).
2. Melanoma risk score (calibrated probability via Temperature Scaling).
3. Lesion segmentation mask mapped to original image coordinates.
4. Bounding box detections transformed back to native image dimensions.
5. Native-resolution multi-task visual overlay (segmentation contour & bounding boxes).
6. Native-resolution Grad-CAM visual interpretability heatmap.
7. Structured, zero-hallucination clinical report (JSON, HTML, Plain Text).
8. Execution latency breakdown.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .calibration import TemperatureScaler
from .checkpoints import CheckpointMetadata, ModelStatus, load_checkpoint_verified
from .common import device, save_json, seed_all
from .config import CONFIG
from .gradcam import GradCAM, overlay_heatmap
from .model_registry import get_model_spec
from .models import ResNetUNet, classifier, retinanet
from .report import format_clinical_summary_text, save_report_bundle

DEFAULT_THRESHOLD = CONFIG.default_triage_threshold
MELANOMA_LABEL = "HIGH PRIORITY"
BENIGN_LABEL = "LOW PRIORITY"


def load_models(
    dev: Optional[str] = None,
    checkpoints_dir: Optional[Union[str, Path]] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """Load and verify all three vision models (Classification, Segmentation, Detection).

    Strictly refuses to run with uninitialized parameters without explicit notice.
    """
    if dev is None:
        dev = device()
    ckpt_dir = Path(checkpoints_dir) if checkpoints_dir else CONFIG.checkpoints_dir

    cls_path = ckpt_dir / CONFIG.cls_checkpoint_name
    seg_path = ckpt_dir / CONFIG.seg_checkpoint_name
    det_path = ckpt_dir / CONFIG.det_checkpoint_name

    # 1. Classifier
    cls_net = classifier("resnet18", pretrained=False).to(dev)
    cls_ok, cls_status, cls_msg, cls_meta = load_checkpoint_verified(
        cls_net, cls_path, expected_arch="resnet18", device=dev, strict=False
    )

    # 2. Segmentor
    seg_net = ResNetUNet(pretrained=False).to(dev)
    seg_ok, seg_status, seg_msg, seg_meta = load_checkpoint_verified(
        seg_net, seg_path, expected_arch="ResNetUNet", device=dev, strict=False
    )

    # 3. Detector
    det_net = retinanet(num_classes=2, pretrained_backbone=False).to(dev)
    det_ok, det_status, det_msg, det_meta = load_checkpoint_verified(
        det_net, det_path, expected_arch="retinanet", device=dev, strict=False
    )

    cls_net.eval()
    seg_net.eval()
    det_net.eval()

    # Load calibration scaler if available
    calib_path = ckpt_dir / "cls_resnet18_calibration.json"
    scaler = None
    if calib_path.exists():
        try:
            scaler = TemperatureScaler.load(calib_path)
        except Exception:
            scaler = None

    statuses = {
        "cls": cls_status,
        "seg": seg_status,
        "det": det_status,
    }
    all_ready = cls_ok and seg_ok and det_ok

    return {
        "cls": cls_net,
        "seg": seg_net,
        "det": det_net,
        "device": dev,
        "status": statuses,
        "all_ready": all_ready,
        "scaler": scaler,
        "metadata": {
            "cls": cls_meta,
            "seg": seg_meta,
            "det": det_meta,
        }
    }


def validate_input_image(image_input: Union[str, Path, Image.Image]) -> Image.Image:
    """Validate, orient, and sanitize an input image safely.

    Protects against decompression bombs, unreadable formats, and dimension issues.
    """
    Image.MAX_IMAGE_PIXELS = CONFIG.max_image_pixels

    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Input image not found: {p.resolve()}")
        if p.stat().st_size > CONFIG.max_upload_bytes:
            raise ValueError(f"Image exceeds maximum upload limit of {CONFIG.max_upload_bytes / (1024*1024):.1f} MB.")
        raw_img = Image.open(p)
    elif isinstance(image_input, Image.Image):
        raw_img = image_input
    else:
        raise TypeError(f"Expected file path or PIL Image, got {type(image_input)}")

    # Apply EXIF orientation transposition
    raw_img = ImageOps.exif_transpose(raw_img)

    # Convert to RGB (stripping alpha channels cleanly onto white background)
    if raw_img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", raw_img.size, (255, 255, 255))
        if raw_img.mode == "P":
            raw_img = raw_img.convert("RGBA")
        background.paste(raw_img, mask=raw_img.split()[-1] if "A" in raw_img.getbands() else None)
        raw_img = background
    else:
        raw_img = raw_img.convert("RGB")

    w, h = raw_img.size
    if w < 16 or h < 16:
        raise ValueError(f"Image dimensions ({w}x{h}) are too small for medical lesion evaluation.")

    return raw_img


def run_inference(
    image_input: Union[str, Path, Image.Image],
    out_dir: Optional[Union[str, Path]] = None,
    models: Optional[Dict[str, Any]] = None,
    threshold: float = DEFAULT_THRESHOLD,
    size: int = 128
) -> Tuple[Dict[str, Any], Image.Image]:
    """Execute the full clinical triage, segmentation, and detection pipeline on an image.

    Preserves original dimensions, scales coordinates accurately, applies calibration,
    computes Grad-CAM, and generates an official clinical report bundle.

    Returns:
        (result_dict, native_resolution_overlay_image)
    """
    t_start = time.perf_counter()

    # 1. Validation & Input Normalization
    raw_img = validate_input_image(image_input)
    orig_w, orig_h = raw_img.size

    # 2. Check Models & Readiness
    dev = models["device"] if models else device()
    if models is None:
        models = load_models(dev)

    if not models.get("all_ready", False):
        statuses = models.get("status", {})
        missing = [k for k, s in statuses.items() if s != ModelStatus.READY]
        if missing:
            # Check if this is an explicit demo-only execution
            if CONFIG.runtime_mode != "demo":
                raise RuntimeError(
                    f"Model checkpoint(s) unavailable: {missing}.\n"
                    f"Cannot produce verified clinical triage assessment without trained weights.\n"
                    f"Please place verified checkpoints in {CONFIG.checkpoints_dir} or set runtime_mode='demo'."
                )

    # 3. Preprocessing
    t_pre0 = time.perf_counter()
    img_resized = raw_img.resize((size, size), Image.BILINEAR)
    img_np = np.asarray(img_resized, dtype=np.float32) / 255.0
    x_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))[None].to(dev)
    t_pre = (time.perf_counter() - t_pre0) * 1e3

    # 4. Model Inference
    t_inf0 = time.perf_counter()
    with torch.no_grad():
        # Classification
        cls_logits = models["cls"](x_tensor)
        raw_logit = float(cls_logits.squeeze().item())
        raw_risk_score = float(torch.sigmoid(cls_logits).item())

        # Calibration
        scaler: Optional[TemperatureScaler] = models.get("scaler")
        if scaler is not None:
            calibrated_score = float(scaler.calibrate_probability(raw_logit))
            is_calibrated = True
            temp_val = float(scaler.temperature.item())
        else:
            calibrated_score = raw_risk_score
            is_calibrated = False
            temp_val = 1.0

        # Segmentation
        seg_logits = models["seg"](x_tensor)
        mask_prob_128 = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()

        # Detection
        det_out = models["det"](list(x_tensor))[0]
        boxes_128 = det_out["boxes"].cpu().numpy()
        scores_128 = det_out["scores"].cpu().numpy()

    t_inf = (time.perf_counter() - t_inf0) * 1e3

    # 5. Coordinate & Resolution Transformation to Original Dimensions
    t_post0 = time.perf_counter()

    # Interpolate binary mask back to native original image resolution
    mask_pil_128 = Image.fromarray((mask_prob_128 * 255).astype(np.uint8))
    mask_pil_orig = mask_pil_128.resize((orig_w, orig_h), Image.BILINEAR)
    mask_np_orig = np.asarray(mask_pil_orig, dtype=np.float32) / 255.0
    mask_binary_orig = (mask_np_orig > 0.5).astype(np.uint8)

    lesion_pixels = int(mask_binary_orig.sum())
    total_pixels = orig_w * orig_h
    lesion_area_pct = float(lesion_pixels / max(1, total_pixels) * 100.0)

    # Scale detection bounding boxes back to original image space
    scale_x = orig_w / float(size)
    scale_y = orig_h / float(size)

    det_keep = scores_128 > 0.20
    det_boxes_orig: List[List[float]] = []
    det_scores: List[float] = []

    for bx, sc in zip(boxes_128[det_keep], scores_128[det_keep]):
        x1 = float(np.clip(bx[0] * scale_x, 0, orig_w))
        y1 = float(np.clip(bx[1] * scale_y, 0, orig_h))
        x2 = float(np.clip(bx[2] * scale_x, 0, orig_w))
        y2 = float(np.clip(bx[3] * scale_y, 0, orig_h))
        if (x2 - x1) >= 2 and (y2 - y1) >= 2:
            det_boxes_orig.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
            det_scores.append(round(float(sc), 4))

    # Triage decision at operating threshold
    decision_score = calibrated_score if is_calibrated else raw_risk_score
    priority = MELANOMA_LABEL if decision_score >= threshold else BENIGN_LABEL

    # 6. Render Native Resolution Multi-Task Visual Overlay
    overlay_img = raw_img.copy()
    overlay_np = np.asarray(overlay_img).astype(np.float32)

    # Translucent red overlay on segmented lesion area
    mask_indices = mask_binary_orig == 1
    overlay_np[mask_indices, 0] = overlay_np[mask_indices, 0] * 0.65 + 240.0 * 0.35
    overlay_np[mask_indices, 1] = overlay_np[mask_indices, 1] * 0.65
    overlay_np[mask_indices, 2] = overlay_np[mask_indices, 2] * 0.65
    overlay_img = Image.fromarray(np.clip(overlay_np, 0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(overlay_img)
    line_w = max(2, int(min(orig_w, orig_h) / 200))

    # Draw native-resolution detection bounding boxes
    for bx, sc in zip(det_boxes_orig, det_scores):
        draw.rectangle([bx[0], bx[1], bx[2], bx[3]], outline=(255, 235, 59), width=line_w)
        label_text = f"Lesion Foci: {sc*100:.1f}%"
        draw.text((bx[0] + 4, max(0, bx[1] - 14)), label_text, fill=(255, 235, 59))

    # Top triage status banner scaled to image width
    banner_h = max(24, int(orig_h * 0.05))
    banner_color = (185, 28, 28) if priority == MELANOMA_LABEL else (21, 128, 61)
    draw.rectangle([0, 0, orig_w, banner_h], fill=banner_color)
    banner_text = f"{priority} | Risk: {decision_score*100:.1f}% (Threshold: {threshold:.3f})"
    draw.text((8, max(4, int(banner_h * 0.2))), banner_text, fill=(255, 255, 255))

    # 7. Generate Per-Image Grad-CAM Explainability Heatmap
    gradcam_img: Optional[Image.Image] = None
    try:
        target_layer = models["cls"].layer4[-1]
        with GradCAM(models["cls"], target_layer) as gcam:
            cam_map, _ = gcam.generate(x_tensor, target_size=(orig_h, orig_w))
            gradcam_img = overlay_heatmap(raw_img, cam_map, alpha=0.45)
    except Exception:
        gradcam_img = None

    t_post = (time.perf_counter() - t_post0) * 1e3
    total_ms = (time.perf_counter() - t_start) * 1e3

    # 8. Package Results
    result: Dict[str, Any] = {
        "image_path": str(image_input) if isinstance(image_input, (str, Path)) else "in_memory_upload",
        "priority": priority,
        "risk_score": raw_risk_score,
        "calibrated_score": calibrated_score,
        "is_calibrated": is_calibrated,
        "temperature": temp_val,
        "threshold": threshold,
        "is_high_priority": priority == MELANOMA_LABEL,
        "original_size": {"width": orig_w, "height": orig_h},
        "processed_size": {"width": size, "height": size},
        "lesion_area_pct": lesion_area_pct,
        "lesion_pixels": lesion_pixels,
        "detection_count": len(det_boxes_orig),
        "detection_boxes": det_boxes_orig,
        "detection_scores": det_scores,
        "model_provenance": {
            "cls_model": "ResNet-18",
            "seg_model": "ResNetUNet",
            "det_model": "RetinaNet_ResNet50_FPN",
            "status": {k: str(v) for k, v in models.get("status", {}).items()}
        },
        "latency_ms": {
            "preprocess": round(t_pre, 1),
            "inference": round(t_inf, 1),
            "postprocess": round(t_post, 1),
            "total": round(total_ms, 1)
        }
    }

    # Add clinical summary text
    result["summary"] = format_clinical_summary_text(result)

    # 9. Save Artifacts if Output Directory Specified
    if out_dir is not None:
        p_out = Path(out_dir)
        p_out.mkdir(parents=True, exist_ok=True)

        overlay_path = p_out / "triage_overlay.png"
        mask_path = p_out / "segmentation_mask.png"
        overlay_img.save(overlay_path)
        Image.fromarray(mask_binary_orig * 255).save(mask_path)

        if gradcam_img is not None:
            gcam_path = p_out / "gradcam_heatmap.png"
            gradcam_img.save(gcam_path)

        report_files = save_report_bundle(result, p_out, base_name="triage_report")
        result["saved_files"] = {
            "overlay": str(overlay_path),
            "mask": str(mask_path),
            "gradcam": str(p_out / "gradcam_heatmap.png") if gradcam_img else None,
            "report_json": str(report_files["json"]),
            "report_html": str(report_files["html"]),
            "summary_txt": str(report_files["txt"])
        }

    return result, overlay_img


def main():
    parser = argparse.ArgumentParser(description="Medical Imaging Triage Workstation Inference")
    parser.add_argument("--image", required=True, help="Path to input dermoscopic lesion image")
    parser.add_argument("--out", default="runs/infer_out", help="Output directory for predictions and reports")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Triage operating threshold")
    parser.add_argument("--size", type=int, default=128, help="Processing resolution")
    args = parser.parse_args()

    res, _ = run_inference(args.image, out_dir=args.out, threshold=args.threshold, size=args.size)
    print("\n" + res["summary"])
    if "saved_files" in res:
        print(f"\nArtifacts successfully written to: {args.out}")
        print(f" - JSON Report: {res['saved_files']['report_json']}")
        print(f" - HTML Report: {res['saved_files']['report_html']}")


if __name__ == "__main__":
    main()
