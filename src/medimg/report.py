"""Standardized clinical report generator for triage and lesion analysis.

Exports structured machine-readable JSON and human-readable HTML/Text decision-support
reports with model provenance, calibration status, image metrics, and SaMD disclaimers.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def format_clinical_summary_text(result: Dict[str, Any]) -> str:
    """Format a deterministic, zero-hallucination clinical summary in plain text."""
    priority = result.get("priority", "UNKNOWN")
    raw_score = result.get("risk_score", 0.0)
    calibrated_score = result.get("calibrated_score", raw_score)
    threshold = result.get("threshold", 0.518)
    lesion_area = result.get("lesion_area_pct", 0.0)
    det_boxes = result.get("detection_boxes", [])
    latency = result.get("latency_ms", {}).get("total", 0.0)
    is_calibrated = result.get("is_calibrated", False)
    calib_str = f"Calibrated Risk:   {calibrated_score * 100:.2f}%" if is_calibrated else f"Estimated Risk:    {raw_score * 100:.2f}% (Raw Sigmoid)"

    lines = [
        "CLINICAL TRIAGE & LESION ANALYSIS REPORT",
        "=========================================",
        "LEGAL & REGULATORY NOTICE:",
        "Educational and research prototype (SaMD prototype).",
        "This tool is NOT an autonomous diagnostic device and does NOT diagnose melanoma.",
        "All predictions provide decision support and must be evaluated by a certified clinician.",
        "",
        f"Triage Decision:   {priority}",
        calib_str,
        f"Operating Point:   Threshold = {threshold:.3f} (Val-calibrated target Sensitivity >= 95%)",
        f"Lesion Footprint:  {lesion_area:.2f}% of field of view",
        f"Suspicious Foci:   {len(det_boxes)} detected bounding box(es)",
        f"Inference Latency: {latency:.1f} ms end-to-end",
        "",
        "Clinical Pathway Recommendation:",
        (
            "-> HIGH PRIORITY: Fast-track priority review and formal dermatoscopy consultation indicated."
            if priority == "HIGH PRIORITY"
            else "-> LOW PRIORITY: Routine clinical follow-up; advise patient to monitor for changes (ABCDE criteria)."
        ),
        "========================================="
    ]
    return "\n".join(lines)


def generate_html_report(result: Dict[str, Any]) -> str:
    """Generate a clean, standalone HTML clinical report."""
    priority = result.get("priority", "UNKNOWN")
    is_high = priority == "HIGH PRIORITY"
    status_color = "#dc2626" if is_high else "#16a34a"
    bg_color = "#fef2f2" if is_high else "#f0fdf4"
    border_color = "#f87171" if is_high else "#86efac"

    raw_score = result.get("risk_score", 0.0)
    calibrated_score = result.get("calibrated_score", raw_score)
    is_calibrated = result.get("is_calibrated", False)
    score_display = f"{calibrated_score * 100:.2f}%" + (" (Calibrated)" if is_calibrated else " (Raw Sigmoid)")

    det_boxes = result.get("detection_boxes", [])
    det_scores = result.get("detection_scores", [])
    box_rows = ""
    for idx, (b, s) in enumerate(zip(det_boxes, det_scores), start=1):
        box_rows += f"<tr><td>Box #{idx}</td><td>[{b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f}, {b[3]:.0f}]</td><td>{s*100:.1f}%</td></tr>"

    if not box_rows:
        box_rows = "<tr><td colspan='3' style='text-align:center; color:#6b7280;'>No suspicious foci detected above threshold.</td></tr>"

    timestamp = result.get("timestamp", datetime.now(timezone.utc).isoformat())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Clinical Triage & Lesion Analysis Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; line-height: 1.5; color: #1f2937; margin: 40px auto; max-width: 800px; padding: 0 20px; }}
    .header {{ border-bottom: 2px solid #e5e7eb; padding-bottom: 16px; margin-bottom: 24px; }}
    .header h1 {{ margin: 0; font-size: 24px; color: #111827; }}
    .header p {{ margin: 4px 0 0; color: #6b7280; font-size: 13px; }}
    .disclaimer {{ background: #fffbeb; border-left: 4px solid #f59e0b; padding: 12px 16px; margin-bottom: 24px; font-size: 13px; color: #92400e; }}
    .triage-card {{ background: {bg_color}; border: 2px solid {border_color}; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
    .triage-title {{ font-size: 22px; font-weight: bold; color: {status_color}; margin-top: 0; margin-bottom: 8px; }}
    .score {{ font-size: 28px; font-weight: 800; color: {status_color}; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px; }}
    th, td {{ padding: 8px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
    th {{ background: #f9fafb; font-weight: 600; color: #4b5563; }}
    .footer {{ margin-top: 32px; border-top: 1px solid #e5e7eb; padding-top: 12px; font-size: 12px; color: #9ca3af; text-align: center; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>Medical Imaging Triage & Lesion Analysis Report</h1>
    <p>Software-as-a-Medical-Device (SaMD) Research Prototype &bull; Report Generated: {timestamp}</p>
  </div>

  <div class="disclaimer">
    <strong>INVESTIGATIONAL USE ONLY:</strong> This report is generated by an automated deep learning research workstation. It is NOT an autonomous medical diagnosis. All findings, risk probabilities, and segmentations must be independently evaluated by a qualified medical professional.
  </div>

  <div class="triage-card">
    <div class="triage-title">{priority}</div>
    <div>Estimated Melanoma Risk: <span class="score">{score_display}</span></div>
    <p style="margin: 8px 0 0; font-size: 14px;"><strong>Clinical Pathway:</strong> {
        "Fast-track priority referral for comprehensive dermatoscopy inspection."
        if is_high else "Routine clinical monitoring. Advise patient to monitor for changes in asymmetry, border, color, or diameter."
    }</p>
  </div>

  <h3>Input Image & Model Specifications</h3>
  <table>
    <tr><th>Parameter</th><th>Value</th></tr>
    <tr><td>Source Image</td><td>{result.get('image_path', 'N/A')}</td></tr>
    <tr><td>Original Dimensions</td><td>{result.get('original_size', {}).get('width', 'N/A')} &times; {result.get('original_size', {}).get('height', 'N/A')} px</td></tr>
    <tr><td>Triage Operating Threshold</td><td>{result.get('threshold', 0.518):.3f} (Val target Sens &ge; 95%)</td></tr>
    <tr><td>Lesion Area Coverage</td><td>{result.get('lesion_area_pct', 0.0):.2f}% of field of view</td></tr>
    <tr><td>Classification Model</td><td>{result.get('model_provenance', {}).get('cls_model', 'ResNet-18')}</td></tr>
    <tr><td>Total Pipeline Latency</td><td>{result.get('latency_ms', {}).get('total', 0.0):.1f} ms</td></tr>
  </table>

  <h3>Suspicious Foci Detections (RetinaNet)</h3>
  <table>
    <tr><th>Detection</th><th>Bounding Box [x1, y1, x2, y2] (Original Image Space)</th><th>Confidence Score</th></tr>
    {box_rows}
  </table>

  <div class="footer">
    MedImg Triage Workstation &bull; Open-source SaMD Research Prototype &bull; Version 0.2.0
  </div>
</body>
</html>"""
    return html


def save_report_bundle(
    result: Dict[str, Any],
    out_dir: Union[str, Path],
    base_name: str = "triage_report"
) -> Dict[str, Path]:
    """Save machine-readable JSON, human-readable HTML, and plain text reports to out_dir."""
    p_out = Path(out_dir)
    p_out.mkdir(parents=True, exist_ok=True)

    json_path = p_out / f"{base_name}.json"
    html_path = p_out / f"{base_name}.html"
    txt_path = p_out / f"{base_name}.txt"

    # Add timestamp if missing
    if "timestamp" not in result:
        result["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Machine-readable JSON
    with open(json_path, "w", encoding="utf-8") as f:
        # Convert non-serializable items
        clean_res = json.loads(json.dumps(result, default=str))
        json.dump(clean_res, f, indent=2)

    # Human-readable HTML
    html_content = generate_html_report(result)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Plain Text
    txt_content = format_clinical_summary_text(result)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)

    # Also write triage_summary.txt for backwards compatibility
    summary_path = p_out / "triage_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(txt_content)

    return {
        "json": json_path,
        "html": html_path,
        "txt": txt_path,
        "summary": summary_path
    }
