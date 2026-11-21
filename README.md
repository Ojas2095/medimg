# Medical Imaging Triage & Lesion Analysis Workstation

[![Tests](https://img.shields.io/badge/pytest-63%20passed%20(100%25)-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)]()
[![SaMD](https://img.shields.io/badge/SaMD-Human--in--the--Loop%20Triage-red.svg)]()
[![CI](https://github.com/Ojas2095/medimg/actions/workflows/ci.yml/badge.svg)]()

A comprehensive, production-grade Software-as-a-Medical-Device (SaMD) workstation prototype for dermatological lesion triage, semantic segmentation, lesion detection, probability calibration, and explainable AI reporting. Evaluated on the HAM10000 and ISIC 2018 datasets with rigorous scientific integrity, zero-hallucination reporting, and complete CPU/GPU reproducibility.

---

## Key Features

1. **Multi-Task Vision Architecture:**
   - **Classification:** ResNet-18 (primary, held-out AUROC 0.8953), ResNet-50, and GoogLeNet for melanoma triage.
   - **Segmentation:** ResNetUNet (primary, held-out Dice 0.7711) and classical U-Net for native-resolution lesion footprint delineation.
   - **Detection:** RetinaNet (held-out mAP@0.5 0.5822) and Faster R-CNN for localized lesion and suspicious foci detection.
2. **Post-Hoc Probability Calibration:**
   - Empirical Temperature Scaling ($T=1.2251$) fitted strictly on validation logits via L-BFGS NLL minimization.
   - Binned Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score evaluation.
3. **Verified Checkpoint Architecture:**
   - Strict model parameter validation, SHA-256 cryptographic hashing, sidecar JSON metadata provenance, and safe CPU-isolated tensor snapshotting.
   - Refuses to run inference on missing or incompatible weights without explicit fallback or false claims.
4. **Interactive Clinical Workstation (Gradio UI):**
   - In-memory image processing, EXIF orientation handling, and native-resolution mask and bounding box scaling.
   - Explanatory Grad-CAM attention heatmaps and one-click downloadable standalone HTML/JSON/TXT clinical decision-support report bundles.
5. **Rigorous Experimental Evaluation:**
   - 63 comprehensive unit and integration tests (`pytest -v`).
   - Group-stratified train/val/test splits preventing lesion-level data leakage (`lesion_id`).
   - Subgroup demographic parity audits (biological sex, anatomical site, Fitzpatrick phototype).

---

## Quickstart

### 1. Installation
Clone the repository and install all dependencies in editable mode:
```bash
# Editable install with all vision, ONNX, UI, and testing dependencies:
pip install -e ".[all]"

# Or install from requirements.txt:
pip install -r requirements.txt
```

### 2. Verify System Integrity
Run the complete automated test suite (63 tests covering NumPy fundamentals, vision architectures, checkpoints, calibration, reporting, and Gradio):
```bash
pytest -v
# or
make test
```

Execute the 6-pillar aggressive model and stress-testing audit:
```bash
python scripts/aggressive_model_check.py
# or
make verify
```

### 3. Launch the Interactive Web Workstation (Gradio UI)
Launch the browser-based clinical workstation:
```bash
python -m medimg gui --port 7860
# or
make gui
```
Open **[http://127.0.0.1:7860](http://127.0.0.1:7860)** in your browser to inspect lesions, view segmentation boundaries, bounding boxes, triage decisions, and Grad-CAM activation heatmaps.

### 4. Single-Image CLI Inference
Run the end-to-end inference pipeline on an input lesion image:
```bash
python -m medimg infer --image data/isic_seg_det/images/ISIC_0000000.png --out out/triage_demo
# or
make infer IMAGE=data/isic_seg_det/images/ISIC_0000000.png OUT=out/triage_demo
```
This generates:
- `triage_overlay.png`: Visual lesion boundary overlay scaled to native resolution.
- `segmentation_mask.png`: Binary segmentation mask at original image dimensions.
- `gradcam_heatmap.png`: High-resolution Grad-CAM activation heatmap.
- `triage_report.html`: Interactive, standalone HTML clinical decision-support report.
- `triage_report.json`: Structured machine-readable JSON triage report with model provenance.
- `triage_report.txt` / `triage_summary.txt`: Plain-text clinical summary for electronic health records (EHR).

---

## Workstation CLI Commands

| Command | Makefile Target | Description |
|---|---|---|
| `pytest -v` | `make test` | Runs all 63 unit and integration tests |
| `python scripts/aggressive_model_check.py` | `make verify` | Runs 6-pillar adversarial, parameter, and SLA verification |
| `python -m medimg gui --port 7860` | `make gui` | Starts the interactive Gradio web workstation |
| `python -m medimg infer --image <img.png> --out <dir>` | `make infer IMAGE=... OUT=...` | Runs single-image triage, segmentation, detection, and reporting |
| `python -m medimg evaluate` | `make evaluate` | Evaluates models on held-out test set with bootstrap 95% CIs |
| `python -m medimg robustness` | `make robustness` | Runs clinical stress tests (blur, noise, lighting, downscaling) |
| `python -m medimg gradcam` | `make gradcam` | Generates Grad-CAM visual heatmaps & failure mode audits |
| `python -m medimg fundamentals all --out out` | `make fundamentals` | Executes from-scratch NumPy demonstrations & ablation sweeps |
| `python scripts/run_benchmarks.py` | `make benchmarks` | Re-executes the full benchmark matrix across all models |

---

## Configuration & Environment Variables

The workstation is configured via `src/medimg/config.py` and supports zero-code configuration via environment variables:

| Environment Variable | Default Value | Description |
|---|---|---|
| `MEDIMG_DEVICE` | Auto (`cuda` if available else `cpu`) | Execution device for neural networks |
| `MEDIMG_TARGET_SENSITIVITY` | `0.95` (95%) | Target triage sensitivity for high-risk operating threshold |
| `MEDIMG_CHECKPOINT_DIR` | `runs/checkpoints` | Centralized checkpoint storage directory |
| `MEDIMG_DATA_DIR` | `data` | Root dataset directory |
| `MEDIMG_MAX_IMAGE_PIXELS` | `25000000` (25 MP) | Security limit against image decompression bombs |
| `MEDIMG_ENABLE_GRADCAM` | `1` (Enabled) | Whether to compute Grad-CAM heatmaps during inference |

---

## Repository Layout

```
medimg/
├── .github/workflows/
│   └── ci.yml                # Automated GitHub Actions CI test pipeline
├── docs/
│   ├── ARCHITECTURE.md       # Complete system architecture and dataflow diagrams
│   ├── CHANGELOG_REMEDIATION.md # Exhaustive audit of all remediation actions
│   ├── PBE_VIVA_GUIDE.md     # Defensible technical viva preparation guide
│   ├── REMEDIATION_BASELINE.md # Phase 0 baseline reconnaissance and defect audit
│   ├── REQUIREMENTS.md       # Clinical stakeholder specifications & acceptance criteria
│   ├── DATASET_CARD.md       # Empirical dataset documentation (HAM10000 & ISIC 2018)
│   ├── MODEL_CARD.md         # Production model specifications, ethics, and fairness audits
│   ├── TECHNICAL_REPORT.md   # Complete experimental benchmark report & Go/No-Go scorecard
│   └── USER_GUIDE.md         # Step-by-step user, clinician, and developer documentation
├── src/medimg/
│   ├── fundamentals/         # Pure NumPy tensor ops, SVD, backprop, MLP, optimizers, ablations
│   ├── app_gradio.py         # Multi-tab interactive Gradio clinical workstation
│   ├── calibration.py        # Post-hoc Temperature Scaling and ECE/MCE/Brier metrics
│   ├── checkpoints.py        # Centralized checkpoint validation, hashing, and metadata
│   ├── common.py             # Shared project constants, seeds, and device management
│   ├── config.py             # Centralized dataclass configuration & env overrides
│   ├── data_ham10000.py      # HAM10000 & ISIC 2018 preprocessor with lesion-level grouping
│   ├── datasets.py           # PyTorch multi-task medical dataset loaders & augmentations
│   ├── evaluate.py           # Comprehensive evaluation suite (AUROC, CI, calibration, latency)
│   ├── export_onnx.py        # ONNX export, shape verification, and ONNX Runtime speedup
│   ├── gradcam.py            # Grad-CAM interpretability engine & failure mode analysis
│   ├── infer.py              # Unified single-image inference pipeline entrypoint
│   ├── metrics.py            # High-precision metrics (sens/spec/AUROC, Dice/IoU, box AP)
│   ├── model_registry.py     # Registry of supported vision model architectures
│   ├── models.py             # ResNet-18/50, GoogLeNet, UNet, ResNetUNet, RetinaNet, Faster R-CNN
│   ├── prepare_splits.py     # Leakage-safe lesion-grouped stratification
│   ├── report.py             # Structured JSON, HTML, and TXT clinical report generator
│   ├── robustness.py         # Environmental degradation stress-testing suite
│   ├── synth_data.py         # Deterministic synthetic lesion generator
│   ├── train_cls.py          # Classification training pipeline (ResNet-18/50, GoogLeNet)
│   ├── train_det.py          # Detection training pipeline (RetinaNet, Faster R-CNN)
│   ├── train_seg.py          # Segmentation training pipeline (UNet, ResNetUNet)
│   ├── transformer_summary.py# Clinical verbalizer (deterministic rule-based + transformer audit)
│   └── vae_augment.py        # ConvVAE minority-class augmentation engine & copy audit
├── tests/                    # 63 unit and integration tests across 8 test suites
└── scripts/
    ├── aggressive_model_check.py # 6-pillar adversarial, parameter, and SLA verification
    ├── run_benchmarks.py         # Multi-architecture benchmark runner
    └── train_final_checkpoints.py # Deterministic checkpoint trainer
```

---

## Regulatory and Clinical Disclaimer

> **INVESTIGATIONAL USE ONLY — Software-as-a-Medical-Device (SaMD) Research Prototype**  
> This system is designed and evaluated strictly as an academic research prototype and clinical decision-support tool. It is **NOT** FDA-cleared, CE-marked, or approved for autonomous medical diagnosis. All classification decisions, segmentation boundaries, and detection boxes must be reviewed and confirmed by a licensed medical practitioner.
