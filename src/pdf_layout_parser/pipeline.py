from __future__ import annotations

import json
from pathlib import Path

from pdf_layout_parser.detectors import build_detector
from pdf_layout_parser.models import (
    DEFAULT_LABEL_POLICIES,
    DocumentParseResult,
    ExtractionAction,
    LabelPolicy,
    PageParseResult,
    ParseRequestOptions,
    ParsedRegion,
)
from pdf_layout_parser.ocr import OcrEngine
from pdf_layout_parser.pdf import extract_native_text_blocks, native_text_for_region, render_pdf_pages


def parse_document(pdf_path: Path, output_dir: Path, options: ParseRequestOptions) -> DocumentParseResult:
    pdf_path = pdf_path.resolve()
    output_dir = output_dir.resolve()
    pages_dir = output_dir / "pages"
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    rendered_pages = render_pdf_pages(pdf_path, pages_dir, dpi=options.dpi, max_pages=options.max_pages)
    try:
        native_blocks = extract_native_text_blocks(pdf_path, dpi=options.dpi, max_pages=options.max_pages)
    except RuntimeError as exc:
        native_blocks = {}
        warnings.append(str(exc))

    detector = build_detector(
        detector_name=options.detector,
        model_path=options.model_path,
        device=options.device,
        image_size=options.image_size,
        confidence=options.confidence,
    )
    policies = build_policies(options.policy_overrides)
    ocr = OcrEngine(enabled=options.run_ocr)

    result = DocumentParseResult(
        input_path=str(pdf_path),
        output_dir=str(output_dir),
        detector=detector.name,
        model_path=str(options.model_path) if options.model_path else None,
        warnings=warnings,
    )

    for rendered_page in rendered_pages:
        detections = detector.detect(rendered_page.image_path, rendered_page.page_number)
        page_result = PageParseResult(
            page_number=rendered_page.page_number,
            image_path=str(rendered_page.image_path),
            width=rendered_page.width,
            height=rendered_page.height,
        )
        blocks = native_blocks.get(rendered_page.page_number, [])
        for detection in detections:
            policy = policies.get(detection.label, LabelPolicy(label=detection.label, action=ExtractionAction.OCR))
            text = ""
            table_data = None
            notes: list[str] = []

            if policy.action == ExtractionAction.SKIP:
                notes.append("Skipped by region policy")
            elif policy.action in {ExtractionAction.OCR, ExtractionAction.NATIVE_TEXT}:
                native_text = native_text_for_region(blocks, detection.bbox)
                if native_text:
                    text = native_text
                    notes.append("Used native PDF text")
                elif policy.action == ExtractionAction.OCR:
                    text, ocr_notes = ocr.extract_text(rendered_page.image_path, detection.bbox)
                    notes.extend(ocr_notes)
            elif policy.action == ExtractionAction.TABLE:
                native_text = native_text_for_region(blocks, detection.bbox)
                if native_text:
                    rows = [line.split() for line in native_text.splitlines() if line.strip()]
                    table_data = rows if rows else None
                    text = native_text
                    notes.append("Prototype table extraction used native text line splitting")
                else:
                    text, ocr_notes = ocr.extract_text(rendered_page.image_path, detection.bbox)
                    notes.extend(["Prototype table parser fallback used OCR text", *ocr_notes])

            page_result.regions.append(
                ParsedRegion(
                    page_number=rendered_page.page_number,
                    label=detection.label,
                    bbox=detection.bbox.clamp(rendered_page.width, rendered_page.height),
                    confidence=detection.confidence,
                    action=policy.action,
                    source=detection.source,
                    text=text,
                    table_data=table_data,
                    notes=notes,
                )
            )
        result.pages.append(page_result)

    write_outputs(result, output_dir)
    return result


def build_policies(overrides: dict[str, ExtractionAction]) -> dict[str, LabelPolicy]:
    policies = {label: policy.model_copy() for label, policy in DEFAULT_LABEL_POLICIES.items()}
    for label, action in overrides.items():
        policies[label] = LabelPolicy(label=label, action=action)
    return policies


def write_outputs(result: DocumentParseResult, output_dir: Path) -> None:
    result_json = output_dir / "result.json"
    result_md = output_dir / "result.md"

    result_json.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result_md.write_text(to_markdown(result), encoding="utf-8")


def to_markdown(result: DocumentParseResult) -> str:
    lines = [
        "# Parsed PDF Result",
        "",
        f"- Input: `{result.input_path}`",
        f"- Detector: `{result.detector}`",
    ]
    if result.model_path:
        lines.append(f"- Model: `{result.model_path}`")
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)

    for page in result.pages:
        lines.extend(["", f"## Page {page.page_number}", ""])
        for region in page.regions:
            bbox = region.bbox
            lines.append(
                f"### {region.label} "
                f"({region.action.value}, conf={region.confidence:.2f}, "
                f"bbox={bbox.x:.0f},{bbox.y:.0f},{bbox.width:.0f},{bbox.height:.0f})"
            )
            if region.notes:
                lines.extend(f"- {note}" for note in region.notes)
            if region.table_data:
                lines.append("")
                lines.extend("| " + " | ".join(row) + " |" for row in region.table_data)
            elif region.text:
                lines.extend(["", region.text])
            lines.append("")
    return "\n".join(lines).strip() + "\n"

