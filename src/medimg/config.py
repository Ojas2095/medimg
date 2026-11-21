"""Centralized configuration management for medimg workstation.

Provides structured, immutable, environment-aware configuration with sensible defaults
for medical imaging triage, training, evaluation, and user interface workflows.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

@dataclass(frozen=True)
class MedImgConfig:
    """Project-wide configuration settings."""

    # Project directories
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(default_factory=lambda: Path("data"))
    checkpoints_dir: Path = field(default_factory=lambda: Path("runs/checkpoints"))
    output_dir: Path = field(default_factory=lambda: Path("out"))
    runs_dir: Path = field(default_factory=lambda: Path("runs"))
    onnx_dir: Path = field(default_factory=lambda: Path("runs/onnx"))

    # Runtime & device settings
    device: str = field(default_factory=lambda: "cuda" if os.environ.get("MEDIMG_DEVICE") == "cuda" else ("cpu" if os.environ.get("MEDIMG_DEVICE") == "cpu" else ("cuda" if _is_cuda_available() else "cpu")))
    seed: int = 42
    runtime_mode: Literal["production", "demo"] = "production"

    # Clinical Operating Thresholds (selected strictly on validation cohorts)
    # Default threshold tuned on HAM10000 validation split to achieve Sensitivity >= 95%
    default_triage_threshold: float = 0.518
    target_sensitivity: float = 0.95

    # Input image specifications & safety limits
    classifier_input_size: int = 128
    segmentor_input_size: int = 128
    detector_input_size: int = 128
    max_upload_bytes: int = 20 * 1024 * 1024  # 20 MB max image upload
    max_image_pixels: int = 25_000_000         # 25 Megapixels (decompression bomb protection)

    # Web UI settings
    gradio_server_name: str = "127.0.0.1"      # Localhost only by default for security
    gradio_server_port: int = 7860
    gradio_share: bool = False

    # Model filenames
    cls_checkpoint_name: str = "cls_resnet18.pt"
    seg_checkpoint_name: str = "seg_resnetunet.pt"
    det_checkpoint_name: str = "det_retinanet.pt"
    onnx_classifier_name: str = "cls_resnet18.onnx"

    def get_checkpoint_path(self, model_key: Literal["cls", "seg", "det"]) -> Path:
        """Resolve full path to a model checkpoint."""
        mapping = {
            "cls": self.cls_checkpoint_name,
            "seg": self.seg_checkpoint_name,
            "det": self.det_checkpoint_name,
        }
        return self.checkpoints_dir / mapping[model_key]

    def get_onnx_path(self) -> Path:
        """Resolve full path to the ONNX classifier export."""
        return self.onnx_dir / self.onnx_classifier_name


def _is_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Shared default configuration instance
CONFIG = MedImgConfig()
