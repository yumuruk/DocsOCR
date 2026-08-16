from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

from pdf_layout_parser.models import BoundingBox, DetectedRegion


class LayoutDetector(ABC):
    name: str

    @abstractmethod
    def detect(self, image_path: Path, page_number: int) -> list[DetectedRegion]:
        raise NotImplementedError


class HeuristicLayoutDetector(LayoutDetector):
    name = "heuristic"

    def detect(self, image_path: Path, page_number: int) -> list[DetectedRegion]:
        with Image.open(image_path) as image:
            width, height = image.size

        header_h = max(40, height * 0.08)
        footer_h = max(40, height * 0.08)
        title_h = max(50, height * 0.10)
        body_y = header_h + title_h
        body_h = max(0, height - body_y - footer_h)

        return [
            DetectedRegion(
                label="header",
                bbox=BoundingBox(x=0, y=0, width=width, height=header_h),
                confidence=0.35,
                source=self.name,
            ),
            DetectedRegion(
                label="title",
                bbox=BoundingBox(x=width * 0.08, y=header_h, width=width * 0.84, height=title_h),
                confidence=0.4,
                source=self.name,
            ),
            DetectedRegion(
                label="body",
                bbox=BoundingBox(x=width * 0.08, y=body_y, width=width * 0.84, height=body_h),
                confidence=0.3,
                source=self.name,
            ),
            DetectedRegion(
                label="footer",
                bbox=BoundingBox(x=0, y=height - footer_h, width=width, height=footer_h),
                confidence=0.35,
                source=self.name,
            ),
        ]


class DocLayoutYoloDetector(LayoutDetector):
    name = "doclayout-yolo"

    def __init__(self, model_path: Path, device: str = "cpu", image_size: int = 1024, confidence: float = 0.25):
        self.model_path = model_path
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self._model: Any | None = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        configure_doclayout_runtime_cache()
        try:
            from doclayout_yolo import YOLOv10  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "doclayout-yolo is not installed. Install with `pip install -e .[doclayout]`."
            ) from exc
        from pdf_layout_parser.torchcompat import trusted_checkpoint_loading

        # PyTorch >= 2.6 rejects doclayout_yolo's pickled checkpoints under the
        # weights_only default; these are operator-supplied or self-trained files.
        with trusted_checkpoint_loading():
            self._model = YOLOv10(str(self.model_path))
        return self._model

    def detect(self, image_path: Path, page_number: int) -> list[DetectedRegion]:
        model = self._load_model()
        results = model.predict(
            str(image_path),
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
        )
        if not results:
            return []

        result = results[0]
        names = getattr(result, "names", {}) or getattr(model, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = _to_list(getattr(boxes, "xyxy", []))
        cls_values = _to_list(getattr(boxes, "cls", []))
        conf_values = _to_list(getattr(boxes, "conf", []))

        regions: list[DetectedRegion] = []
        for index, coords in enumerate(xyxy):
            if len(coords) < 4:
                continue
            label_id = int(_scalar(cls_values[index])) if index < len(cls_values) else None
            confidence = float(_scalar(conf_values[index])) if index < len(conf_values) else 0.0
            label = str(names.get(label_id, label_id if label_id is not None else "unknown"))
            regions.append(
                DetectedRegion(
                    label=label.lower(),
                    bbox=BoundingBox.from_xyxy(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])),
                    confidence=confidence,
                    source=self.name,
                    raw_label_id=label_id,
                )
            )
        return regions



def configure_doclayout_runtime_cache() -> None:
    cache_root = Path(os.environ.get("DOCSOCR_CACHE_DIR", Path.cwd() / ".cache" / "docsocr"))
    yolo_config_dir = cache_root / "ultralytics"
    matplotlib_config_dir = cache_root / "matplotlib"
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))


def build_detector(
    detector_name: str,
    model_path: Path | None = None,
    device: str = "cpu",
    image_size: int = 1024,
    confidence: float = 0.25,
) -> LayoutDetector:
    if detector_name == "heuristic":
        return HeuristicLayoutDetector()
    if detector_name == "doclayout-yolo":
        if model_path is None:
            raise ValueError("--model-path is required when detector is doclayout-yolo")
        return DocLayoutYoloDetector(model_path=model_path, device=device, image_size=image_size, confidence=confidence)
    raise ValueError(f"Unsupported detector: {detector_name}")


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _scalar(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else 0
    if hasattr(value, "item"):
        return value.item()
    return value

