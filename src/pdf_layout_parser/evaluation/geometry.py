"""Exact box geometry for evaluation.

Two things live here that deliberately do not reuse anything from ``models.py``:

1. :func:`iou` -- ``BoundingBox.intersects()`` computes
   ``intersection / min(area_a, area_b)``, not IoU. It exists as a text-attribution
   heuristic and returns True for a small box mostly inside a large one. Using it
   as an IoU@0.5 match would inflate mAP, so evaluation computes IoU from scratch.

2. Area of unions/differences of box *sets*, used by the operational metrics.
   These are computed by coordinate compression rather than by rasterising a
   mask: exact (no dpi-dependent rounding), dependency-free, and fast enough in
   pure Python for the page counts involved. A 1240x1754 page would be 2.2M
   pixels per mask; the compressed grid is a few thousand cells.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pdf_layout_parser.models import BoundingBox


def intersection_area(a: BoundingBox, b: BoundingBox) -> float:
    """Area of the intersection of two boxes."""
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection over union. This is the real thing, not ``intersects()``."""
    inter = intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = a.area() + b.area() - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _positive(boxes: Iterable[BoundingBox]) -> list[BoundingBox]:
    return [box for box in boxes if box.width > 0 and box.height > 0]


def _region_area(
    a_boxes: Sequence[BoundingBox],
    b_boxes: Sequence[BoundingBox],
    mode: str,
) -> float:
    """Area of a boolean combination of union(A) and union(B).

    ``mode`` is one of ``"or"``, ``"and"``, ``"diff"`` (A minus B).

    Coordinate compression: the plane is cut along every box edge, so each
    resulting cell is either wholly inside or wholly outside each box. Testing
    the cell centre is therefore exact, and overlapping boxes are counted once.
    """
    a_boxes = _positive(a_boxes)
    b_boxes = _positive(b_boxes)
    relevant = [*a_boxes, *b_boxes]
    if not relevant:
        return 0.0

    xs = sorted({value for box in relevant for value in (box.x, box.x2)})
    ys = sorted({value for box in relevant for value in (box.y, box.y2)})
    if len(xs) < 2 or len(ys) < 2:
        return 0.0

    total = 0.0
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        width = x1 - x0
        if width <= 0:
            continue
        cx = (x0 + x1) / 2.0
        # Restrict to boxes spanning this vertical strip before the inner loop.
        strip_a = [box for box in a_boxes if box.x <= cx <= box.x2]
        strip_b = [box for box in b_boxes if box.x <= cx <= box.x2]
        if mode == "and" and not (strip_a and strip_b):
            continue
        if mode == "diff" and not strip_a:
            continue
        if mode == "or" and not (strip_a or strip_b):
            continue
        for j in range(len(ys) - 1):
            y0, y1 = ys[j], ys[j + 1]
            height = y1 - y0
            if height <= 0:
                continue
            cy = (y0 + y1) / 2.0
            in_a = any(box.y <= cy <= box.y2 for box in strip_a)
            if mode == "diff":
                if not in_a:
                    continue
                if any(box.y <= cy <= box.y2 for box in strip_b):
                    continue
                total += width * height
                continue
            in_b = any(box.y <= cy <= box.y2 for box in strip_b)
            if mode == "and":
                if in_a and in_b:
                    total += width * height
            elif in_a or in_b:
                total += width * height
    return total


def union_area(boxes: Iterable[BoundingBox]) -> float:
    """Area covered by at least one box, counting overlaps once."""
    boxes = list(boxes)
    return _region_area(boxes, [], "or")


def overlap_area(a_boxes: Iterable[BoundingBox], b_boxes: Iterable[BoundingBox]) -> float:
    """Area of ``union(A) & union(B)``."""
    return _region_area(list(a_boxes), list(b_boxes), "and")


def difference_area(a_boxes: Iterable[BoundingBox], b_boxes: Iterable[BoundingBox]) -> float:
    """Area of ``union(A) \\ union(B)``."""
    return _region_area(list(a_boxes), list(b_boxes), "diff")


def clamp_to_page(bbox: BoundingBox, width: float, height: float) -> BoundingBox:
    """Clamp in x1y1x2y2 space.

    ``dataset.to_yolo_bbox`` clamped the *normalised centre and size independently*,
    which moves a box that hangs off the page edge instead of cropping it: a box at
    x=-10 width=30 has centre 5, but clamping centre and width separately keeps
    width 30 and so shifts the right edge. Clamping the corners first and only then
    converting is the geometrically correct order.
    """
    x1 = min(max(bbox.x, 0.0), width)
    y1 = min(max(bbox.y, 0.0), height)
    x2 = min(max(bbox.x2, 0.0), width)
    y2 = min(max(bbox.y2, 0.0), height)
    return BoundingBox.from_xyxy(x1, y1, x2, y2)
