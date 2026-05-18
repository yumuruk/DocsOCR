from __future__ import annotations

from pathlib import Path

from PIL import Image

from pdf_layout_parser.models import BoundingBox


class OcrEngine:
    name = "pytesseract"

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def extract_text(self, image_path: Path, bbox: BoundingBox) -> tuple[str, list[str]]:
        if not self.enabled:
            return "", ["OCR disabled"]

        try:
            import pytesseract  # type: ignore
        except ImportError:
            return "", ["pytesseract is not installed; native text or empty text was used"]

        with Image.open(image_path) as image:
            clamped = bbox.clamp(*image.size)
            crop = image.crop(clamped.to_xyxy())
            text = pytesseract.image_to_string(crop)
        return text.strip(), []

