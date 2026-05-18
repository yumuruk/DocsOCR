"""Core OCR/layout pipeline facade.

The frontend and backend should treat this package as the reusable Python
module that owns PDF rendering, layout detection, OCR dispatch, dataset export,
and DocLayout-YOLO training helpers.
"""

from pdf_layout_parser.dataset import export_yolo_dataset
from pdf_layout_parser.detectors import build_detector
from pdf_layout_parser.ocr import OcrEngine
from pdf_layout_parser.pdf import extract_native_text_blocks, render_pdf_pages
from pdf_layout_parser.pipeline import parse_document
from pdf_layout_parser.training import run_doclayout_yolo_training

__all__ = [
    "OcrEngine",
    "build_detector",
    "export_yolo_dataset",
    "extract_native_text_blocks",
    "parse_document",
    "render_pdf_pages",
    "run_doclayout_yolo_training",
]

