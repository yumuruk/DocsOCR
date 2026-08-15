from __future__ import annotations

import json
import shutil
from pathlib import Path

from pdf_layout_parser.evaluation.geometry import clamp_to_page
from pdf_layout_parser.labels import CANONICAL_LABELS, canonical_id, is_canonical
from pdf_layout_parser.models import (
    AnnotationDataset,
    AnnotationPage,
    BoundingBox,
    LabelStatus,
    YoloExportResult,
)
from pdf_layout_parser.splits import DatasetSplits, SplitError, page_key


def load_annotation_dataset(path: Path) -> AnnotationDataset:
    data = json.loads(path.read_text(encoding="utf-8"))
    return AnnotationDataset.model_validate(data)


def export_yolo_dataset(
    annotation_path: Path,
    output_dir: Path,
    max_pages: int | None = None,
    train_ratio: float = 0.8,
    copy_images: bool = True,
    splits: DatasetSplits | None = None,
    strict: bool = False,
) -> YoloExportResult:
    """Convert annotations into a YOLO dataset.

    Pass ``splits`` (from :mod:`pdf_layout_parser.splits`) for anything whose
    numbers matter. Without it the legacy seedless prefix split is used, which
    leaks layout across train/val and gives a different val set for every
    ``max_pages`` -- fine for eyeballing a dataset, not for an experiment. Set
    ``strict=True`` to make the unsafe path an error instead of a warning.
    """
    dataset = load_annotation_dataset(annotation_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    warnings: list[str] = []
    unknown_labels: dict[str, int] = {}
    dropped_boxes = 0

    for label in dataset.labels:
        if not is_canonical(label):
            warnings.append(
                f"Annotation file declares label {label!r}, which is not in the canonical "
                "catalogue; its boxes will be dropped."
            )

    if splits is not None:
        if max_pages is not None:
            warnings.append("max_pages is ignored when explicit splits are provided")
        bucketed = {
            "train": list(splits.train),
            "val": list(splits.val),
            "test": list(splits.test),
        }
        warnings.extend(splits.warnings)
        split_strategy = f"{splits.strategy} (seed {splits.seed})"
    else:
        if strict:
            raise SplitError(
                "export_yolo_dataset(strict=True) requires explicit splits. Build them with "
                "splits.make_splits() so train/val/test are seeded and document-disjoint."
            )
        complete = [page for page in dataset.pages if page.label_status == LabelStatus.COMPLETE]
        skipped_incomplete = len(dataset.pages) - len(complete)
        pages = complete[:max_pages] if max_pages else complete
        if not pages:
            raise ValueError("Annotation dataset has no complete pages to export")
        train_pages, val_pages = split_pages(pages, train_ratio=train_ratio)
        bucketed = {"train": train_pages, "val": val_pages, "test": []}
        split_strategy = "legacy prefix (unseeded)"
        warnings.append(
            "Used the legacy prefix split: no seed, no held-out test set, and pages from one "
            "document land on both sides. Do not compare numbers produced from this split."
        )
        if skipped_incomplete:
            warnings.append(
                f"{skipped_incomplete} page(s) skipped because label_status is not 'complete'"
            )

    all_pages = [*bucketed["train"], *bucketed["val"], *bucketed["test"]]
    if not all_pages:
        raise ValueError("Annotation dataset has no complete pages to export")
    skipped_incomplete = sum(
        1 for page in dataset.pages if page.label_status != LabelStatus.COMPLETE
    )

    written: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    used_names: set[str] = set()

    for split_name in ("train", "val", "test"):
        for page_index, page in enumerate(sorted(bucketed[split_name], key=page_key), start=1):
            source_image = Path(page.image_path)
            if not source_image.exists():
                raise FileNotFoundError(f"Annotated page image does not exist: {source_image}")

            image_name = _unique_image_name(page, split_name, page_index, used_names)
            target_image = images_dir / image_name
            if copy_images and source_image.resolve() != target_image.resolve():
                shutil.copy2(source_image, target_image)

            label_lines, page_dropped, page_unknown = _label_lines_for_page(page)
            dropped_boxes += page_dropped
            for label, count in page_unknown.items():
                unknown_labels[label] = unknown_labels.get(label, 0) + count

            label_path = labels_dir / f"{target_image.stem}.txt"
            label_path.write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
            )
            written[split_name].append(target_image.relative_to(output_dir).as_posix())

    for split_name in ("train", "val", "test"):
        if split_name == "test" and not written["test"]:
            continue
        listing = output_dir / f"{split_name}.txt"
        listing.write_text("\n".join(written[split_name]) + "\n", encoding="utf-8")

    if unknown_labels:
        detail = ", ".join(f"{label!r} x{count}" for label, count in sorted(unknown_labels.items()))
        warnings.append(f"Dropped boxes with non-canonical labels: {detail}")
    if dropped_boxes:
        warnings.append(f"Dropped {dropped_boxes} box(es) with zero area after clamping to the page")

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(build_data_yaml(has_test=bool(written["test"])), encoding="utf-8")

    from pdf_layout_parser.evaluation.harness import dataset_fingerprint

    return YoloExportResult(
        output_dir=str(output_dir),
        images_dir=str(images_dir),
        labels_dir=str(labels_dir),
        train_txt=str(output_dir / "train.txt"),
        val_txt=str(output_dir / "val.txt"),
        test_txt=str(output_dir / "test.txt") if written["test"] else None,
        data_yaml=str(data_yaml),
        labels=list(CANONICAL_LABELS),
        train_count=len(written["train"]),
        val_count=len(written["val"]),
        test_count=len(written["test"]),
        page_count=len(all_pages),
        class_ids={label: canonical_id(label) for label in CANONICAL_LABELS},
        dataset_fingerprint=dataset_fingerprint(all_pages),
        skipped_incomplete_pages=skipped_incomplete,
        dropped_boxes=dropped_boxes,
        warnings=[f"split: {split_strategy}", *warnings],
    )


def _unique_image_name(
    page: AnnotationPage,
    split_name: str,
    page_index: int,
    used_names: set[str],
) -> str:
    """Flat, collision-free image name.

    Pages from different documents are routinely all called ``page_0001.png``, so
    the document id goes into the name; otherwise one document's image silently
    overwrites another's in the flat images directory.
    """
    from pdf_layout_parser.splits import document_id_for

    document = _slug(document_id_for(page))
    stem = Path(page.image_path).stem or f"page_{page_index:04d}"
    candidate = f"{split_name}_{document}_{_slug(stem)}.png"
    suffix = 1
    while candidate in used_names:
        suffix += 1
        candidate = f"{split_name}_{document}_{_slug(stem)}_{suffix}.png"
    used_names.add(candidate)
    return candidate


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)[:64] or "x"


def _label_lines_for_page(page: AnnotationPage) -> tuple[list[str], int, dict[str, int]]:
    lines: list[str] = []
    dropped = 0
    unknown: dict[str, int] = {}
    for annotation in page.annotations:
        if not is_canonical(annotation.label):
            unknown[annotation.label] = unknown.get(annotation.label, 0) + 1
            continue
        bbox = clamp_to_page(annotation.bbox, page.width, page.height)
        if bbox.width <= 0 or bbox.height <= 0:
            dropped += 1
            continue
        xc, yc, width, height = to_yolo_bbox(bbox, page.width, page.height)
        lines.append(f"{canonical_id(annotation.label)} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}")
    return lines, dropped, unknown


def split_pages(pages: list, train_ratio: float = 0.8) -> tuple[list, list]:
    """Legacy prefix split, kept for the non-experimental export path.

    The single-page case used to ``return pages, pages`` -- the same page as both
    train and val, a total leak. It now raises instead.
    """
    if not 0 < train_ratio <= 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if len(pages) < 2:
        raise SplitError(
            f"Cannot split {len(pages)} page(s) into train and val without putting the same "
            "page on both sides. Annotate at least 2 pages."
        )

    train_count = int(len(pages) * train_ratio)
    train_count = min(max(train_count, 1), len(pages) - 1)
    train_pages = pages[:train_count]
    val_pages = pages[train_count:]
    return train_pages, val_pages


def build_data_yaml(has_test: bool = False) -> str:
    """Emit the dataset descriptor with the canonical class list.

    ``nc`` and ``names`` are always the full catalogue regardless of which classes
    happen to appear in this dataset. Sizing them to the present classes is what
    made class ids dataset-dependent, so a checkpoint trained on one export could
    not be evaluated against another. There is deliberately no ``labels``
    parameter: an earlier version accepted one and silently ignored it, which is
    worse than either honouring or refusing it.
    """
    names = "\n".join(f"  {index}: {label}" for index, label in enumerate(CANONICAL_LABELS))
    test_line = "test: test.txt\n" if has_test else ""
    return (
        "path: .\n"
        "train: train.txt\n"
        "val: val.txt\n"
        f"{test_line}"
        f"nc: {len(CANONICAL_LABELS)}\n"
        "names:\n"
        f"{names}\n"
    )


def to_yolo_bbox(bbox: BoundingBox, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    """Pixel xywh -> normalised YOLO centre form.

    Clamping happens in corner space *before* the conversion. The previous version
    converted first and then clamped the normalised centre and size independently,
    which does not crop an out-of-bounds box -- it moves it. A box at x=-10,
    width=30 has its centre at 5; clamping centre and width separately keeps
    width 30 and pushes the right edge from 20 out to 35.
    """
    clamped = clamp_to_page(bbox, float(image_width), float(image_height))
    x_center = (clamped.x + clamped.width / 2) / image_width
    y_center = (clamped.y + clamped.height / 2) / image_height
    width = clamped.width / image_width
    height = clamped.height / image_height
    return (
        min(max(x_center, 0.0), 1.0),
        min(max(y_center, 0.0), 1.0),
        min(max(width, 0.0), 1.0),
        min(max(height, 0.0), 1.0),
    )
