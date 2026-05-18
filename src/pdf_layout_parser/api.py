"""Compatibility shim for the legacy API import path.

New code should run the backend with:

    uvicorn backend.main:app

The OCR/layout pipeline itself lives in the pdf_layout_parser package.
"""

from backend.main import app

__all__ = ["app"]

