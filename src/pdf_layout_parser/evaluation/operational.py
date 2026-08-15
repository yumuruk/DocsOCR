"""Operational metrics: how much OCR was avoided, and what it cost.

mAP is the wrong instrument for the skip half of this project. Whether a figure
box is 3px off does not matter; whether a table was dropped entirely does. These
metrics measure the thing the pipeline actually decides -- which page area gets
sent to extraction -- in units of area:

* **reduction_rate** = 1 - area(pred_keep) / area(page)
  Cost saved versus full-page OCR. Higher is better.
* **loss_rate** = area(gt_keep \\ pred_keep) / area(gt_keep)
  Ground truth content that the pipeline threw away. This is the dangerous error
  and belongs in a hard constraint, not a weighted average.
* **contamination_rate** = area(pred_keep \\ gt_keep) / area(pred_keep)
  Share of what was sent to OCR that is not real content -- the "repeated headers
  keep showing up" complaint, quantified.

``pred_keep`` is the union of predicted regions whose label is a keep label under
the region policy, evaluated **at the operating confidence** (0.25), because this
describes deployed behaviour. Detection AP uses the full low-confidence
predictions instead; the two must not share a score threshold.

Two reference rows make the trade-off legible:

* ``full_page_ocr`` -- the incumbent method. reduction 0, loss 0, and its
  contamination is the noise the project exists to remove.
* ``oracle`` -- predictions replaced by ground truth. The reduction ceiling; a
  model cannot do better, so the gap to it is the remaining headroom.
"""

from __future__ import annotations

from pdf_layout_parser.labels import keep_labels
from pdf_layout_parser.models import (
    BoundingBox,
    LabelPolicy,
    OperationalMetrics,
    PageOperationalMetrics,
)

from pdf_layout_parser.evaluation.detection import GtBox, PredBox
from pdf_layout_parser.evaluation.geometry import difference_area, union_area

#: Confidence used to generate predictions for evaluation. Detections below the
#: operating point still carry PR-curve information, and discarding them at
#: predict time (the ``detectors.py`` default of 0.25) truncates the recall tail
#: and systematically understates AP.
EVAL_CONFIDENCE = 0.001

#: The deployed operating point, used for operational metrics and confusion.
OPERATING_CONFIDENCE = 0.25


class PageGeometry:
    """Minimal page descriptor: an id and the rendered pixel size."""

    __slots__ = ("image_id", "width", "height")

    def __init__(self, image_id: str, width: float, height: float) -> None:
        self.image_id = image_id
        self.width = float(width)
        self.height = float(height)

    @property
    def area(self) -> float:
        return self.width * self.height

    def full_page_box(self) -> BoundingBox:
        return BoundingBox(x=0.0, y=0.0, width=self.width, height=self.height)


def _keep_boxes(
    boxes: list[GtBox] | list[PredBox],
    keep: frozenset[str],
    image_id: str,
    score_threshold: float | None = None,
) -> list[BoundingBox]:
    selected: list[BoundingBox] = []
    for box in boxes:
        if box.image_id != image_id or box.label not in keep:
            continue
        if score_threshold is not None and getattr(box, "score", 1.0) < score_threshold:
            continue
        selected.append(box.bbox)
    return selected


def page_operational_metrics(
    page: PageGeometry,
    gt_keep_boxes: list[BoundingBox],
    pred_keep_boxes: list[BoundingBox],
) -> PageOperationalMetrics:
    gt_area = union_area(gt_keep_boxes)
    pred_area = union_area(pred_keep_boxes)
    return PageOperationalMetrics(
        image_id=page.image_id,
        page_area=page.area,
        gt_keep_area=gt_area,
        pred_keep_area=pred_area,
        loss_area=difference_area(gt_keep_boxes, pred_keep_boxes),
        contamination_area=difference_area(pred_keep_boxes, gt_keep_boxes),
    )


def _aggregate(name: str, pages: list[PageOperationalMetrics]) -> OperationalMetrics:
    """Area-weighted aggregation.

    Weighting by area rather than averaging per-page rates keeps a nearly empty
    page from carrying the same weight as a dense one, and keeps the numbers
    interpretable as "of all the area, this much".
    """
    page_area = sum(entry.page_area for entry in pages)
    gt_keep_area = sum(entry.gt_keep_area for entry in pages)
    pred_keep_area = sum(entry.pred_keep_area for entry in pages)
    loss_area = sum(entry.loss_area for entry in pages)
    contamination_area = sum(entry.contamination_area for entry in pages)
    return OperationalMetrics(
        variant=name,
        page_count=len(pages),
        page_area=page_area,
        gt_keep_area=gt_keep_area,
        pred_keep_area=pred_keep_area,
        loss_area=loss_area,
        contamination_area=contamination_area,
        reduction_rate=(1.0 - pred_keep_area / page_area) if page_area > 0 else 0.0,
        loss_rate=(loss_area / gt_keep_area) if gt_keep_area > 0 else None,
        contamination_rate=(contamination_area / pred_keep_area) if pred_keep_area > 0 else None,
    )


def evaluate_operational(
    pages: list[PageGeometry],
    gts: list[GtBox],
    preds: list[PredBox],
    policies: dict[str, LabelPolicy] | None = None,
    score_threshold: float = OPERATING_CONFIDENCE,
    variant: str = "model",
) -> tuple[OperationalMetrics, list[PageOperationalMetrics]]:
    keep = keep_labels(policies)
    per_page = [
        page_operational_metrics(
            page,
            _keep_boxes(gts, keep, page.image_id),
            _keep_boxes(preds, keep, page.image_id, score_threshold=score_threshold),
        )
        for page in pages
    ]
    return _aggregate(variant, per_page), per_page


def full_page_baseline(
    pages: list[PageGeometry],
    gts: list[GtBox],
    policies: dict[str, LabelPolicy] | None = None,
) -> OperationalMetrics:
    """The incumbent method: OCR the whole page, every page."""
    keep = keep_labels(policies)
    per_page = [
        page_operational_metrics(
            page,
            _keep_boxes(gts, keep, page.image_id),
            [page.full_page_box()],
        )
        for page in pages
    ]
    return _aggregate("full_page_ocr", per_page)


def oracle_baseline(
    pages: list[PageGeometry],
    gts: list[GtBox],
    policies: dict[str, LabelPolicy] | None = None,
) -> OperationalMetrics:
    """Perfect layout detection: the reduction ceiling."""
    keep = keep_labels(policies)
    per_page = [
        page_operational_metrics(
            page,
            _keep_boxes(gts, keep, page.image_id),
            _keep_boxes(gts, keep, page.image_id),
        )
        for page in pages
    ]
    return _aggregate("oracle", per_page)
