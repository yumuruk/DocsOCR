"""Prototype package for annotation-driven PDF layout parsing."""

from pdf_layout_parser.models import (
    BoundingBox,
    DocumentParseResult,
    LabelPolicy,
    PageParseResult,
    ParsedRegion,
)

__all__ = [
    "BoundingBox",
    "DocumentParseResult",
    "LabelPolicy",
    "PageParseResult",
    "ParsedRegion",
]

