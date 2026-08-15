"""Loading ground truth, generating predictions, and running an evaluation.

Prediction generation deliberately uses two confidence values:

* ``EVAL_CONFIDENCE`` (0.001) when asking the model for detections, so the recall
  tail of the PR curve survives. The detector's own default of 0.25 filters at
  predict time, which truncates the curve and understates AP -- an operating
  point must never be baked into an evaluation.
* ``OPERATING_CONFIDENCE`` (0.25) when computing operational metrics and the
  confusion matrix, because those describe deployed behaviour.

Every report records the label catalogue fingerprint, the label mapping used, a
dataset fingerprint, and the seed, so a number can be traced back to the data and
configuration that produced it.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from pdf_layout_parser.labels import (
    CANONICAL_LABELS,
    catalogue_fingerprint,
    is_canonical,
    map_label,
    resolve_mapping,
)
from pdf_layout_parser.models import (
    AnnotationDataset,
    AnnotationPage,
    BoundingBox,
    EvaluationReport,
    LabelPolicy,
    LabelStatus,
    SplitSummary,
)
from pdf_layout_parser.splits import document_id_for

from pdf_layout_parser.evaluation.detection import (
    GtBox,
    IOU_THRESHOLDS,
    PredBox,
    confusion_matrix,
    evaluate_detections,
)
from pdf_layout_parser.evaluation.geometry import clamp_to_page
from pdf_layout_parser.evaluation.operational import (
    EVAL_CONFIDENCE,
    OPERATING_CONFIDENCE,
    PageGeometry,
    evaluate_operational,
    full_page_baseline,
    oracle_baseline,
)


class GroundTruthError(ValueError):
    """Raised when ground truth cannot be loaded without losing information."""


def image_id_for(page: AnnotationPage) -> str:
    """Stable, path-independent page identity.

    Built from document id plus image file name rather than the stored
    ``image_path``, which is a server-absolute path pinned to one parse run
    (``runs/web/<uuid>/pages/...``). Reports keyed on that would die with the run
    directory; this survives the files moving.
    """
    return f"{document_id_for(page)}::{Path(page.image_path).name}"


@dataclass
class GroundTruth:
    pages: list[PageGeometry]
    boxes: list[GtBox]
    annotation_pages: list[AnnotationPage]
    fingerprint: str
    warnings: list[str] = field(default_factory=list)
    skipped_incomplete: int = 0


def dataset_fingerprint(pages: list[AnnotationPage]) -> str:
    """Content digest of the ground truth, for reproducibility bookkeeping.

    Covers page identity, size, and every box rounded to 0.01px. Two runs
    reporting the same fingerprint were scored on identical data; differing
    fingerprints explain differing numbers before anyone blames the model.
    """
    digest = hashlib.sha256()
    for page in sorted(pages, key=image_id_for):
        digest.update(f"{image_id_for(page)}|{page.width}x{page.height}|".encode("utf-8"))
        rows = sorted(
            f"{box.label}:{box.bbox.x:.2f},{box.bbox.y:.2f},{box.bbox.width:.2f},{box.bbox.height:.2f}:{box.source.value}"
            for box in page.annotations
        )
        digest.update(("&".join(rows) + "\n").encode("utf-8"))
    return digest.hexdigest()[:16]


def load_annotation_pages(annotation_path: Path) -> AnnotationDataset:
    data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    return AnnotationDataset.model_validate(data)


def load_ground_truth(
    annotation_path: Path | None = None,
    pages: list[AnnotationPage] | None = None,
    require_complete: bool = True,
    on_unknown_label: str = "error",
) -> GroundTruth:
    """Build ground truth from an annotation file or an explicit page list.

    ``on_unknown_label`` defaults to ``"error"``: a ground truth box with a label
    outside the catalogue used to be dropped without a word, which quietly lowers
    recall for a class nobody realises is missing.
    """
    if pages is None:
        if annotation_path is None:
            raise GroundTruthError("Provide either annotation_path or pages")
        pages = load_annotation_pages(annotation_path).pages

    warnings: list[str] = []
    usable: list[AnnotationPage] = []
    skipped = 0
    for page in pages:
        if require_complete and page.label_status != LabelStatus.COMPLETE:
            skipped += 1
            continue
        usable.append(page)
    if skipped:
        warnings.append(f"{skipped} page(s) excluded because label_status is not 'complete'")
    if not usable:
        raise GroundTruthError("No pages with label_status='complete' to evaluate")

    geometries: list[PageGeometry] = []
    boxes: list[GtBox] = []
    unknown: dict[str, int] = {}
    degenerate = 0

    for page in usable:
        image_id = image_id_for(page)
        geometries.append(PageGeometry(image_id=image_id, width=page.width, height=page.height))
        for annotation in page.annotations:
            if not is_canonical(annotation.label):
                unknown[annotation.label] = unknown.get(annotation.label, 0) + 1
                continue
            bbox = clamp_to_page(annotation.bbox, page.width, page.height)
            if bbox.width <= 0 or bbox.height <= 0:
                degenerate += 1
                continue
            boxes.append(GtBox(image_id=image_id, label=annotation.label, bbox=bbox))

    if unknown:
        detail = ", ".join(f"{label!r} x{count}" for label, count in sorted(unknown.items()))
        message = f"Ground truth contains non-canonical labels: {detail}"
        if on_unknown_label == "error":
            raise GroundTruthError(
                message
                + ". Add them to labels.CANONICAL_LABELS, or pass on_unknown_label='warn' "
                "to drop them knowingly."
            )
        warnings.append(message + " (dropped)")
    if degenerate:
        warnings.append(f"{degenerate} ground truth box(es) dropped for zero area after clamping")

    duplicates = len(geometries) - len({geometry.image_id for geometry in geometries})
    if duplicates:
        raise GroundTruthError(
            f"{duplicates} duplicate page id(s). Two annotated pages share document_id and "
            "file name, so their boxes would be pooled. Set distinct document_id values."
        )

    return GroundTruth(
        pages=geometries,
        boxes=boxes,
        annotation_pages=usable,
        fingerprint=dataset_fingerprint(usable),
        warnings=warnings,
        skipped_incomplete=skipped,
    )


def predictions_from_annotations(
    pages: list[AnnotationPage],
    score: float = 1.0,
) -> list[PredBox]:
    """Treat annotations as predictions. Used for the oracle and for tests."""
    return [
        PredBox(
            image_id=image_id_for(page),
            label=annotation.label,
            bbox=annotation.bbox,
            score=score,
        )
        for page in pages
        for annotation in page.annotations
        if is_canonical(annotation.label)
    ]


def generate_predictions(
    pages: list[AnnotationPage],
    detector_name: str = "doclayout-yolo",
    model_path: Path | None = None,
    device: str = "cpu",
    image_size: int = 1024,
    label_mapping: str | Path | None = None,
    confidence: float = EVAL_CONFIDENCE,
) -> tuple[list[PredBox], str, list[str]]:
    """Run a detector over annotated pages at evaluation confidence.

    Returns ``(predictions, mapping_name, warnings)``.
    """
    from pdf_layout_parser.detectors import build_detector

    mapping_name, mapping = resolve_mapping(label_mapping)
    detector = build_detector(
        detector_name=detector_name,
        model_path=model_path,
        device=device,
        image_size=image_size,
        confidence=confidence,
    )

    missing = [page.image_path for page in pages if not Path(page.image_path).exists()]
    if missing:
        preview = ", ".join(missing[:3])
        raise GroundTruthError(
            f"{len(missing)} annotated page image(s) are missing, e.g. {preview}. "
            "Annotation files store absolute paths into a parse run directory; "
            "re-render the pages or fix image_path."
        )

    warnings: list[str] = []
    unmapped: dict[str, int] = {}
    predictions: list[PredBox] = []
    for page in pages:
        image_id = image_id_for(page)
        detections = detector.detect(Path(page.image_path), page.page_number or 0)
        for detection in detections:
            canonical = map_label(detection.label, mapping)
            if canonical is None:
                unmapped[detection.label] = unmapped.get(detection.label, 0) + 1
                continue
            bbox = clamp_to_page(detection.bbox, page.width, page.height)
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            predictions.append(
                PredBox(
                    image_id=image_id,
                    label=canonical,
                    bbox=bbox,
                    score=detection.confidence,
                )
            )

    if unmapped:
        detail = ", ".join(f"{label!r} x{count}" for label, count in sorted(unmapped.items()))
        warnings.append(
            f"Detections dropped because mapping {mapping_name!r} has no target for: {detail}"
        )
    return predictions, mapping_name, warnings


def save_predictions(predictions: list[PredBox], path: Path) -> None:
    payload = [
        {
            "image_id": pred.image_id,
            "label": pred.label,
            "score": pred.score,
            "bbox": {
                "x": pred.bbox.x,
                "y": pred.bbox.y,
                "width": pred.bbox.width,
                "height": pred.bbox.height,
            },
        }
        for pred in predictions
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_predictions(path: Path) -> list[PredBox]:
    """Load cached predictions, so scoring does not require the model installed."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    predictions: list[PredBox] = []
    for entry in data:
        bbox = entry["bbox"]
        predictions.append(
            PredBox(
                image_id=str(entry["image_id"]),
                label=str(entry["label"]),
                bbox=BoundingBox(
                    x=float(bbox["x"]),
                    y=float(bbox["y"]),
                    width=float(bbox["width"]),
                    height=float(bbox["height"]),
                ),
                score=float(entry.get("score", 1.0)),
            )
        )
    return predictions


def evaluate(
    ground_truth: GroundTruth,
    predictions: list[PredBox],
    run_name: str,
    detector: str = "unknown",
    model_path: Path | None = None,
    label_mapping: str = "identity",
    policies: dict[str, LabelPolicy] | None = None,
    operating_confidence: float = OPERATING_CONFIDENCE,
    eval_confidence: float = EVAL_CONFIDENCE,
    seed: int | None = None,
    split: SplitSummary | None = None,
    annotation_path: Path | None = None,
    iou_thresholds: tuple[float, ...] = IOU_THRESHOLDS,
    notes: list[str] | None = None,
    include_per_page: bool = False,
    scored_split: str | None = None,
    tags: dict[str, str] | None = None,
) -> EvaluationReport:
    """Score predictions against ground truth and assemble a report."""
    started = time.perf_counter()

    # Drop predictions for pages the ground truth does not cover BEFORE scoring.
    # The operational metrics never saw them (they iterate GT pages), but the
    # detection matcher counted every one as a false positive, so a cached
    # prediction file covering more pages than the split silently deflated AP.
    gt_page_ids = {page.image_id for page in ground_truth.pages}
    off_split = [pred for pred in predictions if pred.image_id not in gt_page_ids]
    if off_split:
        predictions = [pred for pred in predictions if pred.image_id in gt_page_ids]

    detection = evaluate_detections(
        ground_truth.boxes,
        predictions,
        labels=CANONICAL_LABELS,
        iou_thresholds=iou_thresholds,
        score_threshold=eval_confidence,
    )
    model_operational, per_page = evaluate_operational(
        ground_truth.pages,
        ground_truth.boxes,
        predictions,
        policies=policies,
        score_threshold=operating_confidence,
        variant="model",
    )
    operational = [
        full_page_baseline(ground_truth.pages, ground_truth.boxes, policies=policies),
        model_operational,
        oracle_baseline(ground_truth.pages, ground_truth.boxes, policies=policies),
    ]
    confusion = confusion_matrix(
        ground_truth.boxes,
        predictions,
        iou_threshold=0.5,
        score_threshold=operating_confidence,
    )

    collected = [*ground_truth.warnings, *(notes or [])]
    if off_split:
        off_pages = sorted({pred.image_id for pred in off_split})
        collected.append(
            f"{len(off_split)} prediction(s) on {len(off_pages)} page(s) outside this split's "
            f"ground truth were excluded before scoring (e.g. {', '.join(off_pages[:3])}). "
            "They would otherwise count as false positives and deflate AP."
        )
    for entry in detection.per_class:
        if entry.gt_count == 0:
            collected.append(
                f"Class {entry.label!r} has no ground truth in this split; its AP is reported as "
                "null and excluded from mAP."
            )
        elif entry.pred_count == 0:
            collected.append(
                f"Class {entry.label!r} was never predicted. If the label mapping has no source "
                "for it (e.g. target_data under a zero-shot checkpoint), AP 0 is a label-space "
                "artefact rather than a model failure."
            )

    return EvaluationReport(
        run_name=run_name,
        detector=detector,
        model_path=str(model_path) if model_path else None,
        label_mapping=label_mapping,
        label_catalogue=list(CANONICAL_LABELS),
        catalogue_fingerprint=catalogue_fingerprint(),
        dataset_fingerprint=ground_truth.fingerprint,
        annotation_path=str(annotation_path) if annotation_path else None,
        page_count=len(ground_truth.pages),
        gt_box_count=len(ground_truth.boxes),
        pred_box_count=len(predictions),
        eval_confidence=eval_confidence,
        operating_confidence=operating_confidence,
        seed=seed,
        split=split,
        scored_split=scored_split,
        tags=dict(tags or {}),
        detection=detection,
        operational=operational,
        confusion=confusion,
        per_page_operational=per_page if include_per_page else [],
        notes=collected,
        duration_seconds=time.perf_counter() - started,
    )
