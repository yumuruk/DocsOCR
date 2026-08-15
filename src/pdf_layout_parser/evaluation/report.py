"""Report rendering.

The operational table comes first, before mAP. Selecting a trade-off is the point
of the exercise, and that decision is made on "how much OCR did we avoid, and how
much content did we lose", with full-page OCR and the oracle on the same axes for
reference. Per-class AP answers a different, narrower question and follows.
"""

from __future__ import annotations

import json
from pathlib import Path

from pdf_layout_parser.labels import BACKGROUND, MISSED
from pdf_layout_parser.models import EvaluationReport


def _percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def render_markdown(report: EvaluationReport) -> str:
    lines: list[str] = [f"# Evaluation — {report.run_name}", ""]

    lines.extend(
        [
            "| Field | Value |",
            "| --- | --- |",
            f"| Detector | `{report.detector}` |",
            f"| Model | `{report.model_path or 'n/a'}` |",
            f"| Label mapping | `{report.label_mapping}` |",
            f"| Label catalogue | `{report.catalogue_fingerprint}` ({len(report.label_catalogue)} classes) |",
            f"| Dataset fingerprint | `{report.dataset_fingerprint}` |",
            f"| Pages / GT boxes / predictions | {report.page_count} / {report.gt_box_count} / {report.pred_box_count} |",
            f"| Eval confidence | {report.eval_confidence} |",
            f"| Operating confidence | {report.operating_confidence} |",
            f"| Seed | {report.seed if report.seed is not None else 'n/a'} |",
        ]
    )
    if report.split:
        split = report.split
        lines.append(
            f"| Split ({split.strategy}, seed {split.seed}) | "
            f"train {split.train_pages}p / val {split.val_pages}p / test {split.test_pages}p |"
        )
        lines.append(
            f"| Test documents | {', '.join(split.test_documents) or 'n/a'} |"
        )
    if report.duration_seconds is not None:
        lines.append(f"| Scoring time | {report.duration_seconds:.2f}s |")
    lines.append("")

    lines.extend(_operational_section(report))
    lines.extend(_detection_section(report))
    lines.extend(_confusion_section(report))

    if report.notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _operational_section(report: EvaluationReport) -> list[str]:
    if not report.operational:
        return []
    lines = [
        "## Operational — selective OCR vs full-page OCR",
        "",
        "`reduction` is area saved against full-page OCR; `loss` is ground truth content",
        "discarded (treat as a hard constraint, not a tradeable average); `contamination`",
        "is the share of extracted area that is not real content.",
        "",
        "| Variant | Reduction | Loss | Contamination | Kept area / page area |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in report.operational:
        ratio = entry.pred_keep_area / entry.page_area if entry.page_area else 0.0
        lines.append(
            f"| `{entry.variant}` | {_percent(entry.reduction_rate)} | "
            f"{_percent(entry.loss_rate)} | {_percent(entry.contamination_rate)} | "
            f"{_percent(ratio)} |"
        )
    lines.append("")

    model = next((entry for entry in report.operational if entry.variant == "model"), None)
    oracle = next((entry for entry in report.operational if entry.variant == "oracle"), None)
    baseline = next((entry for entry in report.operational if entry.variant == "full_page_ocr"), None)
    if model and oracle and baseline:
        headroom = oracle.reduction_rate - model.reduction_rate
        lines.extend(
            [
                f"- Reduction ceiling (oracle): {_percent(oracle.reduction_rate)}; "
                f"model reaches {_percent(model.reduction_rate)}, "
                f"leaving {_percent(headroom)} of headroom.",
                f"- Full-page OCR contamination is {_percent(baseline.contamination_rate)}; "
                f"the model leaves {_percent(model.contamination_rate)}.",
                "",
            ]
        )
    return lines


def _detection_section(report: EvaluationReport) -> list[str]:
    detection = report.detection
    if detection is None:
        return []
    lines = [
        "## Detection — per-class AP",
        "",
        f"| Class | GT | Pred | AP@0.5 | AP@0.5:0.95 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in detection.per_class:
        lines.append(
            f"| `{entry.label}` | {entry.gt_count} | {entry.pred_count} | "
            f"{_score(entry.ap50)} | {_score(entry.ap5095)} |"
        )
    lines.extend(
        [
            f"| **mAP** | | | **{_score(detection.map50)}** | **{_score(detection.map5095)}** |",
            "",
            "mAP is the mean over classes that have ground truth; `n/a` classes are excluded.",
            "",
        ]
    )
    return lines


def _confusion_section(report: EvaluationReport) -> list[str]:
    if not report.confusion:
        return []
    gt_labels = sorted({cell.gt_label for cell in report.confusion if cell.gt_label != BACKGROUND})
    pred_labels = sorted({cell.pred_label for cell in report.confusion if cell.pred_label != MISSED})
    counts = {(cell.gt_label, cell.pred_label): cell.count for cell in report.confusion}

    header = "| GT ↓ / Pred → | " + " | ".join(f"`{label}`" for label in pred_labels) + " | `missed` |"
    divider = "| --- |" + " --- |" * (len(pred_labels) + 1)
    lines = [
        "## Confusion matrix",
        "",
        f"Class-agnostic IoU≥0.5 matching at the operating confidence ({report.operating_confidence}).",
        "",
        header,
        divider,
    ]
    for gt_label in gt_labels:
        row = [f"| `{gt_label}` "]
        for pred_label in pred_labels:
            row.append(f"| {counts.get((gt_label, pred_label), 0)} ")
        row.append(f"| {counts.get((gt_label, MISSED), 0)} |")
        lines.append("".join(row))

    background_row = [f"| `{BACKGROUND}` "]
    for pred_label in pred_labels:
        background_row.append(f"| {counts.get((BACKGROUND, pred_label), 0)} ")
    background_row.append("| — |")
    lines.append("".join(background_row))
    lines.append("")
    return lines


def write_report(report: EvaluationReport, output_dir: Path, basename: str = "evaluation") -> dict[str, str]:
    """Write ``<basename>.json`` and ``<basename>.md``, returning their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{basename}.json"
    md_path = output_dir / f"{basename}.md"
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def render_comparison(reports: list[EvaluationReport], title: str = "Experiment comparison") -> str:
    """One row per evaluated cell, for choosing between configurations.

    This is the table the experiment matrix is read from: mAP next to the
    operational numbers, so a configuration that wins on mAP but loses content is
    visible as such.
    """
    lines = [
        f"# {title}",
        "",
        "| Run | Pages | mAP@0.5 | mAP@0.5:0.95 | Reduction | Loss | Contamination | Dataset |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        model = next((entry for entry in report.operational if entry.variant == "model"), None)
        detection = report.detection
        lines.append(
            f"| `{report.run_name}` | {report.page_count} | "
            f"{_score(detection.map50 if detection else None)} | "
            f"{_score(detection.map5095 if detection else None)} | "
            f"{_percent(model.reduction_rate if model else None)} | "
            f"{_percent(model.loss_rate if model else None)} | "
            f"{_percent(model.contamination_rate if model else None)} | "
            f"`{report.dataset_fingerprint}` |"
        )
    lines.append("")
    return "\n".join(lines)
