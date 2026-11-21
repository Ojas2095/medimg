"""Unit tests for clinical triage report generation (JSON, HTML, and TXT)."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from medimg.report import (
    generate_html_report,
    format_clinical_summary_text,
    save_report_bundle
)


def _make_dummy_inference_result():
    return {
        "status": "OPERATIONAL",
        "priority": "HIGH PRIORITY",
        "risk_score": 0.875,
        "calibrated_score": 0.875,
        "is_calibrated": False,
        "temperature": 1.45,
        "threshold": 0.518,
        "lesion_detected": True,
        "lesion_area_pct": 8.6,
        "detection_boxes": [[24.0, 30.0, 110.0, 125.0]],
        "detection_scores": [0.94],
        "image_path": "sample.png",
        "original_size": {
            "width": 600,
            "height": 450
        },
        "latency_ms": {
            "preprocess": 4.2,
            "classification": 12.5,
            "segmentation": 18.1,
            "detection": 15.3,
            "gradcam": 22.0,
            "total": 72.1
        },
        "model_provenance": {
            "cls_model": "ResNet-18",
            "seg_model": "ResNetUNet",
            "det_model": "RetinaNet"
        }
    }


def test_format_clinical_summary_text():
    res = _make_dummy_inference_result()
    text = format_clinical_summary_text(res)

    assert "CLINICAL TRIAGE & LESION ANALYSIS REPORT" in text
    assert "HIGH PRIORITY" in text
    assert "87.50%" in text
    assert "SaMD prototype" in text
    assert "NOT an autonomous diagnostic device" in text


def test_generate_html_report():
    res = _make_dummy_inference_result()
    html = generate_html_report(res)

    assert "<!DOCTYPE html>" in html
    assert "<title>Clinical Triage & Lesion Analysis Report</title>" in html
    assert "HIGH PRIORITY" in html
    assert "INVESTIGATIONAL USE ONLY" in html
    assert "600" in html
    assert "450" in html


def test_save_report_bundle():
    res = _make_dummy_inference_result()
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = save_report_bundle(res, tmpdir, base_name="test_report")

        assert "json" in bundle and bundle["json"].exists()
        assert "html" in bundle and bundle["html"].exists()
        assert "txt" in bundle and bundle["txt"].exists()
        assert "summary" in bundle and bundle["summary"].exists()

        with open(bundle["json"], "r", encoding="utf-8") as f:
            loaded_json = json.load(f)
        assert loaded_json["priority"] == "HIGH PRIORITY"
        assert loaded_json["risk_score"] == 0.875
        assert "timestamp" in loaded_json

        html_text = bundle["html"].read_text(encoding="utf-8")
        assert "Clinical Triage & Lesion Analysis Report" in html_text

        txt_text = bundle["txt"].read_text(encoding="utf-8")
        assert "HIGH PRIORITY" in txt_text
