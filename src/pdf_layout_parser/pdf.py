from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pdf_layout_parser.models import BoundingBox


@dataclass(frozen=True)
class RenderedPage:
    page_number: int
    image_path: Path
    width: int
    height: int


@dataclass(frozen=True)
class NativeTextBlock:
    page_number: int
    bbox: BoundingBox
    text: str


def require_pymupdf():
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for PDF rendering. Install with `pip install -e .[pdf]`."
        ) from exc
    return fitz


def render_pdf_pages(pdf_path: Path, output_dir: Path, dpi: int = 180, max_pages: int | None = None) -> list[RenderedPage]:
    fitz = require_pymupdf()
    output_dir.mkdir(parents=True, exist_ok=True)

    pages: list[RenderedPage] = []
    doc = fitz.open(pdf_path)
    try:
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        limit = min(len(doc), max_pages) if max_pages else len(doc)
        for index in range(limit):
            page = doc[index]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output_dir / f"page_{index + 1:04d}.png"
            pix.save(image_path)
            pages.append(
                RenderedPage(
                    page_number=index + 1,
                    image_path=image_path,
                    width=pix.width,
                    height=pix.height,
                )
            )
    finally:
        doc.close()
    return pages


def extract_native_text_blocks(pdf_path: Path, dpi: int = 180, max_pages: int | None = None) -> dict[int, list[NativeTextBlock]]:
    fitz = require_pymupdf()
    blocks_by_page: dict[int, list[NativeTextBlock]] = {}
    doc = fitz.open(pdf_path)
    try:
        scale = dpi / 72
        limit = min(len(doc), max_pages) if max_pages else len(doc)
        for index in range(limit):
            page = doc[index]
            page_number = index + 1
            page_blocks: list[NativeTextBlock] = []
            for block in page.get_text("blocks"):
                x1, y1, x2, y2, text, *_ = block
                text = str(text).strip()
                if not text:
                    continue
                bbox = BoundingBox.from_xyxy(x1 * scale, y1 * scale, x2 * scale, y2 * scale)
                page_blocks.append(NativeTextBlock(page_number=page_number, bbox=bbox, text=text))
            blocks_by_page[page_number] = page_blocks
    finally:
        doc.close()
    return blocks_by_page


def native_text_for_region(blocks: list[NativeTextBlock], region_bbox: BoundingBox) -> str:
    selected = [block.text for block in blocks if block.bbox.intersects(region_bbox, min_overlap_ratio=0.15)]
    return "\n".join(selected).strip()

