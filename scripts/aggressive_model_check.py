"""
Aggressive Model & Pipeline Stress-Test Suite
Executes end-to-end audits of checkpoints, numerical stability, edge cases,
clinical ground truth predictions, ONNX parity, and latency SLAs.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import onnx
import onnxruntime as ort

from medimg.models import classifier, ResNetUNet, retinanet
from medimg.infer import run_inference, load_models, DEFAULT_THRESHOLD

def audit_checkpoints():
    print("=" * 70)
    print("1. CHECKPOINT INTEGRITY & PARAMETER STABILITY AUDIT")
    print("=" * 70)

    # Classification
    cls_path = "runs/checkpoints/cls_resnet18.pt"
    assert os.path.exists(cls_path), f"Missing {cls_path}"
    cls_ckpt = torch.load(cls_path, map_location="cpu", weights_only=True)
    n_params = sum(p.numel() for p in cls_ckpt.values())
    has_nan = any(torch.isnan(p).any().item() for p in cls_ckpt.values())
    has_inf = any(torch.isinf(p).any().item() for p in cls_ckpt.values())
    print(f"[*] ResNet-18 Checkpoint: {n_params:,} parameters | Has NaN: {has_nan} | Has Inf: {has_inf}")
    assert not has_nan and not has_inf, "Classification checkpoint has NaN or Inf!"

    # Segmentation
    seg_path = "runs/checkpoints/seg_resnetunet.pt"
    assert os.path.exists(seg_path), f"Missing {seg_path}"
    seg_ckpt = torch.load(seg_path, map_location="cpu", weights_only=True)
    n_params_seg = sum(p.numel() for p in seg_ckpt.values())
    has_nan_seg = any(torch.isnan(p).any().item() for p in seg_ckpt.values())
    has_inf_seg = any(torch.isinf(p).any().item() for p in seg_ckpt.values())
    print(f"[*] ResNetUNet Checkpoint: {n_params_seg:,} parameters | Has NaN: {has_nan_seg} | Has Inf: {has_inf_seg}")
    assert not has_nan_seg and not has_inf_seg, "Segmentation checkpoint has NaN or Inf!"

    # Detection
    det_path = "runs/checkpoints/det_retinanet.pt"
    assert os.path.exists(det_path), f"Missing {det_path}"
    det_ckpt = torch.load(det_path, map_location="cpu", weights_only=True)
    n_params_det = sum(p.numel() for p in det_ckpt.values())
    has_nan_det = any(torch.isnan(p).any().item() for p in det_ckpt.values())
    has_inf_det = any(torch.isinf(p).any().item() for p in det_ckpt.values())
    print(f"[*] RetinaNet Checkpoint: {n_params_det:,} parameters | Has NaN: {has_nan_det} | Has Inf: {has_inf_det}")
    assert not has_nan_det and not has_inf_det, "Detection checkpoint has NaN or Inf!"

    # ONNX
    onnx_path = "runs/onnx/cls_resnet18.onnx"
    assert os.path.exists(onnx_path), f"Missing {onnx_path}"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"[*] ONNX Checkpoint: {os.path.getsize(onnx_path)/(1024*1024):.2f} MB | Graph valid: True")
    print(" Checkpoints integrity verified successfully.\n")


def audit_clinical_predictions():
    print("=" * 70)
    print("2. CLINICAL GROUND TRUTH DISCRIMINATION AUDIT")
    print("=" * 70)
    import pandas as pd
    df = pd.read_csv("data/ham10000/metadata.csv")
    mel_samples = df[df["label"] == 1].head(15)["image_path"].tolist()
    ben_samples = df[df["label"] == 0].head(15)["image_path"].tolist()

    cls_model = classifier("resnet18", pretrained=False)
    cls_model.load_state_dict(torch.load("runs/checkpoints/cls_resnet18.pt", map_location="cpu", weights_only=True))
    cls_model.eval()

    def get_prob(rel_path):
        full_path = os.path.join("data/ham10000", rel_path)
        img = Image.open(full_path).convert("RGB").resize((128, 128))
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        x_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))[None]
        with torch.no_grad():
            logit = cls_model(x_tensor).squeeze().item()
            return float(torch.sigmoid(torch.tensor(logit)).item())

    mel_probs = [get_prob(p) for p in mel_samples]
    ben_probs = [get_prob(p) for p in ben_samples]

    mean_mel = np.mean(mel_probs)
    mean_ben = np.mean(ben_probs)
    print(f"[*] Mean Predicted Risk for True Melanomas (N={len(mel_samples)}): {mean_mel:.4f} (min={np.min(mel_probs):.4f}, max={np.max(mel_probs):.4f})")
    print(f"[*] Mean Predicted Risk for True Benign Nevi  (N={len(ben_samples)}): {mean_ben:.4f} (min={np.min(ben_probs):.4f}, max={np.max(ben_probs):.4f})")
    print(f"[*] Discrimination Gap (Melanoma Mean - Benign Mean): {mean_mel - mean_ben:+.4f}")
    assert mean_mel > mean_ben, "Model fails to discriminate between Melanoma and Benign!"
    print(" Ground truth discrimination verified.\n")


def audit_adversarial_and_edge_cases():
    print("=" * 70)
    print("3. ADVERSARIAL & DEGENERATE EDGE-CASE STRESS-TESTING")
    print("=" * 70)
    out_dir = "out/stress_test_edge_cases"
    os.makedirs(out_dir, exist_ok=True)
    models = load_models("cpu")

    edge_cases = {
        "pure_black": Image.new("RGB", (256, 256), (0, 0, 0)),
        "pure_white": Image.new("RGB", (256, 256), (255, 255, 255)),
        "uniform_noise": Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)),
        "gaussian_noise": Image.fromarray(np.clip(np.random.normal(128, 40, (256, 256, 3)), 0, 255).astype(np.uint8)),
        "extreme_aspect_wide": Image.new("RGB", (1000, 60), (180, 100, 90)),
        "extreme_aspect_tall": Image.new("RGB", (60, 1000), (90, 100, 180)),
        "tiny_image_8x8": Image.new("RGB", (8, 8), (120, 120, 120)),
        "grayscale_single_channel": Image.new("L", (256, 256), 128),
        "rgba_alpha_channel": Image.new("RGBA", (256, 256), (200, 100, 100, 128)),
    }

    all_passed = True
    for name, img in edge_cases.items():
        img_path = os.path.join(out_dir, f"{name}.png")
        img.save(img_path)
        try:
            t0 = time.perf_counter()
            res, _ = run_inference(img_path, out_dir=os.path.join(out_dir, name), models=models)
            elapsed = (time.perf_counter() - t0) * 1000
            score = res["risk_score"]
            priority = res["priority"]
            n_det = res.get("detection_count", len(res.get("detection_boxes", [])))
            assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"
            assert priority in ["HIGH PRIORITY", "LOW PRIORITY"], f"Invalid priority: {priority}"
            print(f"  [PASS] {name:26s} -> Risk: {score:.4f} | Priority: {priority:13s} | Detections: {n_det} | Time: {elapsed:.1f}ms")
        except ValueError as ve:
            if "too small" in str(ve) or "dimension" in str(ve):
                print(f"  [PASS] {name:26s} -> Correctly rejected: {ve}")
            else:
                print(f"  [FAIL] {name:26s} -> Unexpected ValueError: {ve}")
                all_passed = False
        except Exception as e:
            print(f"  [FAIL] {name:26s} -> Exception raised: {e}")
            all_passed = False

    assert all_passed, "One or more edge cases crashed the pipeline!"
    print(" Adversarial & edge cases successfully handled with zero crashes.\n")


def audit_onnx_parity():
    print("=" * 70)
    print("4. ONNX RUNTIME PARITY & SPEEDUP AUDIT")
    print("=" * 70)
    import pandas as pd
    df = pd.read_csv("data/ham10000/metadata.csv").head(25)
    
    cls_model = classifier("resnet18", pretrained=False)
    cls_model.load_state_dict(torch.load("runs/checkpoints/cls_resnet18.pt", map_location="cpu", weights_only=True))
    cls_model.eval()

    ort_session = ort.InferenceSession("runs/onnx/cls_resnet18.onnx", providers=["CPUExecutionProvider"])

    py_times = []
    onnx_times = []
    diffs = []

    for _, row in df.iterrows():
        full_path = os.path.join("data/ham10000", row["image_path"])
        img = Image.open(full_path).convert("RGB").resize((128, 128))
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        x_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))[None]

        # PyTorch
        t0 = time.perf_counter()
        with torch.no_grad():
            py_logit = cls_model(x_tensor).squeeze().item()
        py_times.append((time.perf_counter() - t0) * 1000)

        # ONNX
        t1 = time.perf_counter()
        onnx_out = ort_session.run(None, {"image": x_tensor.numpy()})[0]
        onnx_logit = float(onnx_out.squeeze())
        onnx_times.append((time.perf_counter() - t1) * 1000)

        diffs.append(abs(py_logit - onnx_logit))

    max_diff = max(diffs)
    mean_diff = np.mean(diffs)
    med_py = np.median(py_times)
    med_onnx = np.median(onnx_times)
    speedup = med_py / med_onnx if med_onnx > 0 else 1.0

    print(f"[*] Evaluated on {len(df)} real dermoscopy images:")
    print(f"[*] Max Absolute Logit Difference: {max_diff:.2e}")
    print(f"[*] Mean Absolute Logit Difference: {mean_diff:.2e}")
    print(f"[*] PyTorch CPU Latency (median):  {med_py:.2f} ms")
    print(f"[*] ONNX Runtime CPU Latency (med): {med_onnx:.2f} ms ({speedup:.2f}x speedup)")

    assert max_diff < 1e-4, f"ONNX parity violation! Max difference: {max_diff}"
    print(" ONNX Runtime parity strictly within 1e-4 tolerance.\n")


def audit_latency_sla():
    print("=" * 70)
    print("5. CLINICAL WORKSTATION END-TO-END LATENCY SLA AUDIT")
    print("=" * 70)
    test_img = "data/isic_seg_det/images/ISIC_0000000.png"
    out_dir = "out/stress_test_latency"
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(dev)

    # Warmup
    res, _ = run_inference(test_img, out_dir=out_dir, models=models)

    latencies = []
    for i in range(25):
        t0 = time.perf_counter()
        res, _ = run_inference(test_img, out_dir=out_dir, models=models)
        latencies.append((time.perf_counter() - t0) * 1000)

    med_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)

    print(f"[*] Total Full-Pipeline Inferences: {len(latencies)}")
    print(f"[*] Device:         {dev}")
    print(f"[*] Min Latency:    {min_lat:.1f} ms")
    print(f"[*] Median Latency: {med_lat:.1f} ms")
    print(f"[*] p95 Latency:   {p95_lat:.1f} ms")
    print(f"[*] Max Latency:   {max_lat:.1f} ms")
    target_sla = 150.0 if dev == "cuda" else 350.0
    print(f"[*] Target SLA:    <= {target_sla:.1f} ms/image ({dev})")

    sla_pass = med_lat <= target_sla
    print(f"[*] SLA Compliance: {'PASS' if sla_pass else 'FAIL'}")
    assert sla_pass, f"Median latency {med_lat:.1f} ms exceeds {target_sla:.1f} ms SLA!"
    print(" Workstation latency SLA successfully met.\n")


def audit_factual_consistency():
    print("=" * 70)
    print("6. FACTUAL CONSISTENCY & ZERO-HALLUCINATION AUDIT")
    print("=" * 70)
    test_img = "data/ham10000/images/ISIC_0032182.png"
    models = load_models("cpu")
    res, _ = run_inference(test_img, out_dir="out/stress_test_summary", models=models)
    summary = res["summary"]
    score = res["risk_score"]
    pct_str = f"{score * 100:.2f}%"

    print(f"[*] Triage Score: {score:.4f} ({pct_str})")
    print(f"[*] Generated Summary:\n{summary}")
    print(f"[*] Verifying that {pct_str} is accurately stated in text...")
    assert pct_str in summary, f"Factual inconsistency! {pct_str} not found in summary: '{summary}'"
    print(" Zero-hallucination factual consistency verified.\n")


if __name__ == "__main__":
    audit_checkpoints()
    audit_clinical_predictions()
    audit_adversarial_and_edge_cases()
    audit_onnx_parity()
    audit_latency_sla()
    audit_factual_consistency()
    print("=" * 70)
    print(">>> ALL AGGRESSIVE AUDITS PASSED WITH ZERO ERRORS <<<")
    print("=" * 70)
