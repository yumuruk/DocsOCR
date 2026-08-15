"""Detection metrics: AP, mAP, and a label confusion matrix.

Implements the COCO convention so results are comparable with published numbers
and can be cross-checked against ``pycocotools`` (see ``tests/test_evaluation.py``):

* greedy one-to-one matching, detections consumed in descending score order,
  each matched to the highest-IoU unclaimed ground truth box of the same class
  on the same page;
* precision envelope made monotonically decreasing, then averaged over 101
  recall points 0.00, 0.01, ... 1.00;
* AP@0.5:0.95 is the mean over IoU thresholds 0.50, 0.55, ... 0.95;
* a class with zero ground truth boxes yields ``None`` (not 0.0) and is excluded
  from the mean, because "no examples" is not the same as "got them all wrong".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from pdf_layout_parser.labels import BACKGROUND, CANONICAL_LABELS, MISSED
from pdf_layout_parser.models import (
    BoundingBox,
    ClassDetectionMetrics,
    ConfusionCell,
    DetectionMetrics,
)

from pdf_layout_parser.evaluation.geometry import iou

#: COCO's threshold sweep: 0.50, 0.55, ... 0.95.
IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.50 + 0.05 * step, 2) for step in range(10))

#: Number of recall points in the 101-point interpolation.
RECALL_POINTS: tuple[float, ...] = tuple(index / 100.0 for index in range(101))


@dataclass(frozen=True)
class GtBox:
    image_id: str
    label: str
    bbox: BoundingBox


@dataclass(frozen=True)
class PredBox:
    image_id: str
    label: str
    bbox: BoundingBox
    score: float


def _group_by_image(boxes: list[GtBox]) -> dict[str, list[GtBox]]:
    grouped: dict[str, list[GtBox]] = defaultdict(list)
    for box in boxes:
        grouped[box.image_id].append(box)
    return grouped


def _sorted_predictions(preds: list[PredBox]) -> list[PredBox]:
    """Descending score, with a deterministic tie-break.

    Ties broken by image id then geometry so that two runs over the same data
    produce byte-identical metrics; an unstable order changes the PR curve.
    """
    return sorted(
        preds,
        key=lambda p: (-p.score, p.image_id, p.bbox.x, p.bbox.y, p.bbox.width, p.bbox.height),
    )


def _match_flags(
    gts_by_image: dict[str, list[GtBox]],
    preds: list[PredBox],
    iou_threshold: float,
) -> tuple[list[int], list[int]]:
    """Return per-prediction (tp, fp) indicator lists, in score order."""
    claimed: dict[str, set[int]] = defaultdict(set)
    tp: list[int] = []
    fp: list[int] = []
    for pred in preds:
        candidates = gts_by_image.get(pred.image_id, ())
        best_iou = 0.0
        best_index = -1
        taken = claimed[pred.image_id]
        for index, gt in enumerate(candidates):
            if index in taken:
                continue
            value = iou(pred.bbox, gt.bbox)
            if value > best_iou:
                best_iou = value
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            taken.add(best_index)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    return tp, fp


def average_precision_from_flags(tp: list[int], fp: list[int], gt_count: int) -> float | None:
    """101-point interpolated AP from score-ordered TP/FP indicators."""
    if gt_count <= 0:
        return None
    if not tp:
        return 0.0

    recalls: list[float] = []
    precisions: list[float] = []
    tp_cum = 0
    fp_cum = 0
    for tp_flag, fp_flag in zip(tp, fp):
        tp_cum += tp_flag
        fp_cum += fp_flag
        recalls.append(tp_cum / gt_count)
        denominator = tp_cum + fp_cum
        precisions.append(tp_cum / denominator if denominator else 0.0)

    # Precision envelope: replace each precision by the max of all later ones.
    for index in range(len(precisions) - 2, -1, -1):
        if precisions[index] < precisions[index + 1]:
            precisions[index] = precisions[index + 1]

    total = 0.0
    cursor = 0
    for recall_point in RECALL_POINTS:
        while cursor < len(recalls) and recalls[cursor] < recall_point:
            cursor += 1
        total += precisions[cursor] if cursor < len(precisions) else 0.0
    return total / len(RECALL_POINTS)


def average_precision(
    gts: list[GtBox],
    preds: list[PredBox],
    label: str,
    iou_threshold: float,
) -> float | None:
    """AP for a single class at a single IoU threshold."""
    class_gts = [gt for gt in gts if gt.label == label]
    class_preds = _sorted_predictions([pred for pred in preds if pred.label == label])
    tp, fp = _match_flags(_group_by_image(class_gts), class_preds, iou_threshold)
    return average_precision_from_flags(tp, fp, len(class_gts))


def evaluate_detections(
    gts: list[GtBox],
    preds: list[PredBox],
    labels: tuple[str, ...] = CANONICAL_LABELS,
    iou_thresholds: tuple[float, ...] = IOU_THRESHOLDS,
    score_threshold: float = 0.0,
) -> DetectionMetrics:
    """Per-class AP@0.5, AP@0.5:0.95, and their macro means."""
    per_class: list[ClassDetectionMetrics] = []
    for label in labels:
        class_gts = [gt for gt in gts if gt.label == label]
        class_preds = _sorted_predictions([pred for pred in preds if pred.label == label])
        gts_by_image = _group_by_image(class_gts)

        ap_by_threshold: list[float | None] = []
        for threshold in iou_thresholds:
            tp, fp = _match_flags(gts_by_image, class_preds, threshold)
            ap_by_threshold.append(average_precision_from_flags(tp, fp, len(class_gts)))

        ap50 = ap_by_threshold[0] if ap_by_threshold else None
        present = [value for value in ap_by_threshold if value is not None]
        ap5095 = sum(present) / len(present) if present else None
        per_class.append(
            ClassDetectionMetrics(
                label=label,
                gt_count=len(class_gts),
                pred_count=len(class_preds),
                ap50=ap50,
                ap5095=ap5095,
            )
        )

    scored50 = [entry.ap50 for entry in per_class if entry.ap50 is not None]
    scored5095 = [entry.ap5095 for entry in per_class if entry.ap5095 is not None]
    return DetectionMetrics(
        per_class=per_class,
        map50=sum(scored50) / len(scored50) if scored50 else None,
        map5095=sum(scored5095) / len(scored5095) if scored5095 else None,
        iou_thresholds=list(iou_thresholds),
        score_threshold=score_threshold,
        evaluated_labels=list(labels),
    )


def confusion_matrix(
    gts: list[GtBox],
    preds: list[PredBox],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
) -> list[ConfusionCell]:
    """Class-agnostic matching, then tally (gt label, predicted label).

    Matching ignores the class so that a table detected as a figure shows up as a
    confusion rather than as one miss plus one false positive. Unmatched ground
    truth is reported against :data:`MISSED`, unmatched detections against
    :data:`BACKGROUND`.

    Uses ``score_threshold`` -- the operating point, not the evaluation floor --
    because a confusion matrix describes deployed behaviour.
    """
    kept = [pred for pred in preds if pred.score >= score_threshold]
    gts_by_image = _group_by_image(gts)
    preds_by_image: dict[str, list[PredBox]] = defaultdict(list)
    for pred in kept:
        preds_by_image[pred.image_id].append(pred)

    tally: dict[tuple[str, str], int] = defaultdict(int)
    for image_id in set(gts_by_image) | set(preds_by_image):
        image_gts = gts_by_image.get(image_id, [])
        image_preds = _sorted_predictions(preds_by_image.get(image_id, []))

        pairs: list[tuple[float, int, int]] = []
        for gt_index, gt in enumerate(image_gts):
            for pred_index, pred in enumerate(image_preds):
                value = iou(pred.bbox, gt.bbox)
                if value >= iou_threshold:
                    pairs.append((value, gt_index, pred_index))
        # Highest IoU first; index tie-break keeps the result deterministic.
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

        used_gts: set[int] = set()
        used_preds: set[int] = set()
        for _, gt_index, pred_index in pairs:
            if gt_index in used_gts or pred_index in used_preds:
                continue
            used_gts.add(gt_index)
            used_preds.add(pred_index)
            tally[(image_gts[gt_index].label, image_preds[pred_index].label)] += 1

        for gt_index, gt in enumerate(image_gts):
            if gt_index not in used_gts:
                tally[(gt.label, MISSED)] += 1
        for pred_index, pred in enumerate(image_preds):
            if pred_index not in used_preds:
                tally[(BACKGROUND, pred.label)] += 1

    return [
        ConfusionCell(gt_label=gt_label, pred_label=pred_label, count=count)
        for (gt_label, pred_label), count in sorted(tally.items())
    ]
