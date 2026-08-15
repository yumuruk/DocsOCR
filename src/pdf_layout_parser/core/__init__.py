"""Core OCR/layout pipeline facade.

The frontend and backend should treat this package as the reusable Python
module that owns PDF rendering, layout detection, OCR dispatch, dataset export,
and DocLayout-YOLO training helpers.
"""

from pdf_layout_parser.dataset import export_yolo_dataset
from pdf_layout_parser.detectors import build_detector
from pdf_layout_parser.evaluation import (
    evaluate,
    generate_predictions,
    load_ground_truth,
    write_report,
)
from pdf_layout_parser.experiment_log import append_note, append_run
from pdf_layout_parser.labels import CANONICAL_LABELS, canonical_id
from pdf_layout_parser.ocr import OcrEngine
from pdf_layout_parser.pdf import extract_native_text_blocks, render_pdf_pages
from pdf_layout_parser.pipeline import parse_document
from pdf_layout_parser.splits import make_splits, sample_train_pages
from pdf_layout_parser.training import run_doclayout_yolo_training

__all__ = [
    "CANONICAL_LABELS",
    "OcrEngine",
    "append_note",
    "append_run",
    "build_detector",
    "canonical_id",
    "evaluate",
    "export_yolo_dataset",
    "extract_native_text_blocks",
    "generate_predictions",
    "load_ground_truth",
    "make_splits",
    "parse_document",
    "render_pdf_pages",
    "run_doclayout_yolo_training",
    "sample_train_pages",
    "write_report",
]

