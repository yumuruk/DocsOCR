"""Evaluation harness.

Phase 1 of the plan in ``FINDINGS.md``: before any experiment can be believed,
there has to be something that measures it. Nothing in this package reuses
``BoundingBox.intersects()`` (which is not IoU) or the old prefix split.

Layout::

    geometry.py     IoU and exact set-area arithmetic on boxes
    detection.py    COCO-style AP/mAP and a class-agnostic confusion matrix
    operational.py  reduction / loss / contamination, vs full-page OCR and oracle
    harness.py      loading ground truth, generating predictions, orchestration
    report.py       Markdown + JSON rendering, and the cross-run comparison table
"""

from pdf_layout_parser.evaluation.detection import (
    GtBox,
    IOU_THRESHOLDS,
    PredBox,
    average_precision,
    confusion_matrix,
    evaluate_detections,
)
from pdf_layout_parser.evaluation.geometry import (
    clamp_to_page,
    difference_area,
    intersection_area,
    iou,
    overlap_area,
    union_area,
)
from pdf_layout_parser.evaluation.harness import (
    GroundTruth,
    GroundTruthError,
    dataset_fingerprint,
    evaluate,
    generate_predictions,
    image_id_for,
    load_ground_truth,
    load_predictions,
    predictions_from_annotations,
    save_predictions,
)
from pdf_layout_parser.evaluation.operational import (
    EVAL_CONFIDENCE,
    OPERATING_CONFIDENCE,
    PageGeometry,
    evaluate_operational,
    full_page_baseline,
    oracle_baseline,
)
from pdf_layout_parser.evaluation.report import (
    render_comparison,
    render_markdown,
    write_report,
)

__all__ = [
    "EVAL_CONFIDENCE",
    "GroundTruth",
    "GroundTruthError",
    "GtBox",
    "IOU_THRESHOLDS",
    "OPERATING_CONFIDENCE",
    "PageGeometry",
    "PredBox",
    "average_precision",
    "clamp_to_page",
    "confusion_matrix",
    "dataset_fingerprint",
    "difference_area",
    "evaluate",
    "evaluate_detections",
    "evaluate_operational",
    "full_page_baseline",
    "generate_predictions",
    "image_id_for",
    "intersection_area",
    "iou",
    "load_ground_truth",
    "load_predictions",
    "oracle_baseline",
    "overlap_area",
    "predictions_from_annotations",
    "render_comparison",
    "render_markdown",
    "save_predictions",
    "union_area",
    "write_report",
]
