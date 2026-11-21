# Engineering Remediation Baseline Assessment

**Project:** Medical Imaging Triage & Lesion Analysis Workstation  
**Repository:** https://github.com/Ojas2095/medimg  
**Date:** 2026-11-21  
**Baseline Git Commit:** `56c3f09`  
**Working Branch:** `remediation/pbe-engineering`  
**Execution Environment:** Windows Server / Windows 11, Python 3.13.6, PyTorch 2.7.1+cu118, Torchvision 0.22.1+cu118, CUDA available: True (`NVIDIA GeForce RTX 4080 Laptop GPU`), Gradio 6.32.0, Pytest 8.3.5.

---

## 1. Initial Test & Command Execution Status

### Baseline Commands Executed:
1. `python -m pip install -e ".[all]"` -> **SUCCESS** (all project and optional dependencies installed).
2. `python -m pytest -v` -> **47 passed, 0 failed, 3 warnings** (65.04s).
3. `python -m medimg --help` -> **SUCCESS** (subcommands: `fundamentals`, `data`, `infer`, `evaluate`, `robustness`, `gradcam`, `gui`).
4. `make help` / `make test` -> **SUCCESS** (47 tests). Note: `make benchmarks` was advertised in README but missing from `Makefile`.

---

## 2. Comprehensive Defect Reconnaissance

An exhaustive code audit against the Remediation Requirements revealed the following verified defects:

| Defect ID | Severity | File / Component | Verified Finding Description | Remediation Plan |
|---|---|---|---|---|
| **DEF-01** | **CRITICAL** | `src/medimg/app_gradio.py` | **Gradio Callback Output Count Mismatch:** In `predict_fn`, line 21 returns 5 values (`None, None, None, "Please upload an image.", ""`) on `image is None`, while line 62 returns 4 values, and line 114 connects to 4 output components. Submitting without an image causes a Gradio crash. | Fix return signature to match exactly 4 output components; add input validation guard. |
| **DEF-02** | **CRITICAL** | `src/medimg/app_gradio.py` | **PIL Image Comparison Bug:** Line 31 executes `(overlay_img != image).any(axis=-1)`. Because both objects are PIL Images, `!=` yields a Python `bool`, and `.any()` raises `AttributeError: 'bool' object has no attribute 'any'`. | Remove unused broken expression, implement proper NumPy mask diffing if needed, and add regression test with 128x128 image. |
| **DEF-03** | **HIGH** | `src/medimg/app_gradio.py` | **Shared Temporary File Collision:** Line 24 writes every upload to `tmp_gradio_input.png`. Multiple or concurrent uploads overwrite the same path, and temp file is never cleanly unlinked. | Use in-memory processing (`io.BytesIO`) or `tempfile.NamedTemporaryFile` with guaranteed context manager cleanup. |
| **DEF-04** | **CRITICAL** | `src/medimg/infer.py` | **Silent Fallback to Untrained Weights:** Lines 26, 30, 34 use `if os.path.exists(ckpt): load_state_dict(...)`. If checkpoints are missing, the pipeline silently uses randomly initialized head weights and outputs real-looking risk probabilities and triage classifications. | Implement a centralized `ModelRegistry` and `CheckpointManager` that enforces explicit operational states (`READY`, `CHECKPOINT_MISSING`, `DEMO_FIXTURE_ONLY`) and strictly refuses to output medical predictions from uninitialized weights. |
| **DEF-05** | **HIGH** | `src/medimg/train_seg.py` vs `infer.py` | **Training vs. Inference Architecture Mismatch:** `train_seg.py` instantiates and saves a basic `UNet(3, a.base)`, whereas `infer.py` loads `ResNetUNet(pretrained=True)`. Loading the training output into inference fails with state_dict key errors. | Add `--arch {unet, resnet_unet}` to `train_seg.py`, align default architectures, and record model architecture in checkpoint metadata. |
| **DEF-06** | **HIGH** | `scripts/run_benchmarks.py`, `train_final_checkpoints.py` | **CPU In-Memory Checkpoint Snapshot Mutation:** Lines use `best_state = {k: v.cpu() for k, v in model.state_dict().items()}`. On CPU (`v.is_cuda == False`), `.cpu()` returns the identical tensor without copying. Subsequent in-place training updates mutate `best_state`, corrupting the saved checkpoint to the final epoch instead of the best epoch. | Use `{k: v.detach().cpu().clone() for k, v in model.state_dict().items()}` or `copy.deepcopy()`. Add unit test to verify snapshot isolation. |
| **DEF-07** | **HIGH** | `src/medimg/infer.py`, `app_gradio.py` | **Resolution and Coordinate Distortion:** `infer.py` squishes input images to `128x128`, generates segmentation masks and bounding boxes in 128x128 space, and never scales coordinates back to the original image dimensions `(orig_w, orig_h)`. Overlay output is fixed at 128x128. | Maintain original dimensions and aspect ratio; map detection boxes and segmentation masks back to original image space. |
| **DEF-08** | **HIGH** | `src/medimg/infer.py`, `evaluate.py` | **Uncalibrated Sigmoid Falsely Labeled "Calibrated":** Sigmoid output $\sigma(z)$ is repeatedly referred to as "calibrated melanoma probability" without an actual post-hoc calibration model. | Implement genuine Temperature Scaling (`src/medimg/calibration.py`), fit strictly on the validation set, record calibration parameters, and evaluate Brier score and pre/post ECE. |
| **DEF-09** | **MEDIUM** | `src/medimg/data_ham10000.py`, `prepare_splits.py` | **Ambiguous Grouping Semantics:** HAM10000 metadata contains `lesion_id`, not `patient_id`. The preprocessor mapped `lesion_id` into a column named `patient_id`, creating the false impression that patient-level splitting was performed when only lesion-level splitting was possible. | Explicitly name and document the grouping column as `lesion_id`. Disclose the residual risk of a single patient having multiple distinct lesions. |
| **DEF-10** | **HIGH** | `scripts/run_benchmarks.py` | **Fabricated Attention Map:** Lines 535-537 generate `np.random.uniform(0.1, 0.9, (4, len(words)))` and plot it as a learned cross-attention map from the Transformer summary decoder. | Remove the fake attention plot or extract genuine attention weights from the Transformer decoder; explicitly label any synthetic visualization. |
| **DEF-11** | **MEDIUM** | `src/medimg/gradcam.py` | **Arbitrary Rank-Based Error Categorization:** Lines 130-136 assign failure causes to false negatives based on `if rank <= 5 ... elif rank <= 12 ...`. | Ground error analysis in actual sample metadata (subgroups, lesion diagnosis, localization) and formulate them with appropriate scientific uncertainty as hypotheses. |
| **DEF-12** | **MEDIUM** | `Makefile` | **Missing Make Target:** `README.md` referenced `make benchmarks`, which did not exist in `Makefile`. | Add `benchmarks` target in `Makefile` and verify all targets execute cleanly. |
| **DEF-13** | **MEDIUM** | `.github/workflows/` | **Missing CI Pipeline:** No automated GitHub Actions workflow existed for automated testing, linting, and verification. | Add `.github/workflows/ci.yml` supporting multi-platform or CPU-friendly pytest suites. |
| **DEF-14** | **HIGH** | `src/medimg/infer.py` | **Missing Downloadable Triage Reports:** UI and CLI lack a clean downloadable structured clinical summary (JSON and human-readable HTML/Text). | Add report generation export functions with timestamps, SaMD disclaimers, and model provenance. |

---

## 3. Existing Artifacts & Checkpoints Status
- `runs/checkpoints/cls_resnet18.pt`: 44.78 MB (Present, trained ResNet-18).
- `runs/checkpoints/seg_resnetunet.pt`: 54.14 MB (Present, trained ResNetUNet).
- `runs/checkpoints/det_retinanet.pt`: 129.21 MB (Present, trained RetinaNet).
- `runs/onnx/cls_resnet18.onnx`: 44.70 MB (Present, valid ONNX graph).
- Dataset:
  - `data/ham10000`: 2,600 images processed from HAM10000.
  - `data/isic_seg_det`: 600 matched images & masks from ISIC 2018 Task 1.

---

## 4. Remediation Strategy Roadmap
We will proceed through the phases systematically:
1. **Phase 1 & 9:** Centralized Configuration (`config.py`), Checkpoint Management (`checkpoints.py`), and Model Registry (`model_registry.py`).
2. **Phase 1 & 3:** Inference Engine overhaul (`infer.py`) + Calibration module (`calibration.py`).
3. **Phase 2 & 8:** Gradio Web Application remediation (`app_gradio.py`), input validation, memory safety, and report generation (`report.py`).
4. **Phase 4 & 5:** Data grouping semantics (`data_ham10000.py`, `prepare_splits.py`), training snapshot fix, and benchmark script remediation (`scripts/run_benchmarks.py`).
5. **Phase 6 & 7:** Segmentation/detection coordinate scaling, real Grad-CAM per-image generation, elimination of mock attention maps.
6. **Phase 10 & 11:** Automated test suite expansion (new test modules for checkpoints, Gradio, calibration, config, reports) + GitHub Actions CI workflow.
7. **Phase 12, 13 & 14:** Makefile, full documentation updates, and PBE viva defense guide.
8. **Phase 15:** Comprehensive verification and final report.
