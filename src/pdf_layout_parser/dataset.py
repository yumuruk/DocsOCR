from __future__ import annotations

import json
import shutil
from pathlib import Path

from pdf_layout_parser.models import AnnotationDataset, BoundingBox, YoloExportResult


def load_annotation_dataset(path: Path) -> AnnotationDataset:
    data = json.loads(path.read_text(encoding="utf-8"))
    return AnnotationDataset.model_validate(data)


def export_yolo_dataset(
    annotation_path: Path,
    output_dir: Path,
    max_pages: int | None = None,
    train_ratio: float = 0.8,
    copy_images: bool = True,
) -> YoloExportResult:
    dataset = load_annotation_dataset(annotation_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    pages = dataset.pages[:max_pages] if max_pages else dataset.pages
    if not pages:
        raise ValueError("Annotation dataset has no pages to export")

    label_to_id = {label: index for index, label in enumerate(dataset.labels)}
    train_pages, val_pages = split_pages(pages, train_ratio=train_ratio)
    train_page_ids = {id(page) for page in train_pages}
    val_page_ids = {id(page) for page in val_pages}
    train_lines: list[str] = []
    val_lines: list[str] = []

    for page_index, page in enumerate(pages, start=1):
        source_image = Path(page.image_path)
        if not source_image.exists():
            raise FileNotFoundError(f"Annotated page image does not exist: {source_image}")

        image_name = source_image.name if source_image.suffix else f"page_{page_index:04d}.png"
        image_name = f"page_{page_index:04d}_{image_name}"
        target_image = images_dir / image_name
        if copy_images and source_image.resolve() != target_image.resolve():
            shutil.copy2(source_image, target_image)

        label_path = labels_dir / f"{target_image.stem}.txt"
        label_lines = []
        for annotation in page.annotations:
            if annotation.label not in label_to_id:
                continue
            label_id = label_to_id[annotation.label]
            xc, yc, width, height = to_yolo_bbox(annotation.bbox, page.width, page.height)
            label_lines.append(f"{label_id} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}")
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        relative_image_path = target_image.relative_to(output_dir).as_posix()
        if id(page) in train_page_ids:
            train_lines.append(relative_image_path)
        if id(page) in val_page_ids:
            val_lines.append(relative_image_path)

    (output_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (output_dir / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(build_data_yaml(dataset.labels), encoding="utf-8")

    return YoloExportResult(
        output_dir=str(output_dir),
        images_dir=str(images_dir),
        labels_dir=str(labels_dir),
        train_txt=str(output_dir / "train.txt"),
        val_txt=str(output_dir / "val.txt"),
        data_yaml=str(data_yaml),
        labels=dataset.labels,
        train_count=len(train_lines),
        val_count=len(val_lines),
        page_count=len(pages),
    )


def split_pages(pages: list, train_ratio: float = 0.8) -> tuple[list, list]:
    if not 0 < train_ratio <= 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if len(pages) == 1:
        return pages, pages

    train_count = int(len(pages) * train_ratio)
    train_count = min(max(train_count, 1), len(pages) - 1)
    train_pages = pages[:train_count]
    val_pages = pages[train_count:]
    return train_pages, val_pages


def build_data_yaml(labels: list[str]) -> str:
    names = "\n".join(f"  {index}: {label}" for index, label in enumerate(labels))
    return (
        "path: .\n"
        "train: train.txt\n"
        "val: val.txt\n"
        f"nc: {len(labels)}\n"
        "names:\n"
        f"{names}\n"
    )


def to_yolo_bbox(bbox: BoundingBox, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x_center = (bbox.x + bbox.width / 2) / image_width
    y_center = (bbox.y + bbox.height / 2) / image_height
    width = bbox.width / image_width
    height = bbox.height / image_height
    return (
        min(max(x_center, 0.0), 1.0),
        min(max(y_center, 0.0), 1.0),
        min(max(width, 0.0), 1.0),
        min(max(height, 0.0), 1.0),
    )
