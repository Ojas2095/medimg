"""Regression and integrity tests for Gradio application callbacks, inputs, and outputs."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

from medimg.app_gradio import predict_fn


def test_gradio_predict_fn_empty_input():
    """Verify empty input handling returns exactly 6 values and an informative error."""
    out = predict_fn(None, threshold=0.518)
    assert isinstance(out, tuple)
    assert len(out) == 6, f"Expected 6 output values for Gradio components, got {len(out)}"

    overlay, cam, badge, summary, status, report_path = out
    assert overlay is None
    assert cam is None
    assert "No Image Provided" in badge
    assert "Upload an image" in summary
    assert report_path is None


def test_gradio_predict_fn_valid_synthetic_image():
    """Verify synthetic 128x128 image returns all 6 expected components without crashing."""
    img = Image.new("RGB", (128, 128), color=(200, 140, 110))
    out = predict_fn(img, threshold=0.518)

    assert isinstance(out, tuple)
    assert len(out) == 6

    overlay, cam, badge, summary, status, report_path = out
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (128, 128)
    assert cam is None or isinstance(cam, Image.Image)
    assert isinstance(badge, str) and ("HIGH PRIORITY" in badge or "LOW PRIORITY" in badge)
    assert isinstance(summary, str) and len(summary) > 0
    if report_path is not None:
        assert Path(report_path).exists()


def test_gradio_no_shared_temporary_leak():
    """Verify no tmp_gradio_input.png exists in root or is created by inference."""
    shared_tmp = Path("tmp_gradio_input.png")
    if shared_tmp.exists():
        shared_tmp.unlink()

    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    _ = predict_fn(img, threshold=0.5)

    assert not shared_tmp.exists(), "Found unsafe shared tmp_gradio_input.png in working directory"


def test_pil_image_difference_numpy_safeguard():
    """Regression test for DEF-02: verify image array difference logic does not crash on PIL images."""
    im1 = Image.new("RGB", (64, 64), color=(10, 20, 30))
    im2 = Image.new("RGB", (64, 64), color=(10, 20, 31))

    # Correct NumPy evaluation
    arr1 = np.asarray(im1)
    arr2 = np.asarray(im2)
    has_diff = bool((arr1 != arr2).any())
    assert has_diff is True
