from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field


class ExtractionAction(str, Enum):
    OCR = "ocr"
    TABLE = "table"
    SKIP = "skip"
    NATIVE_TEXT = "native_text"


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

    @computed_field
    @property
    def x2(self) -> float:
        return self.x + self.width

    @computed_field
    @property
    def y2(self) -> float:
        return self.y + self.height

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "BoundingBox":
        return cls(x=x1, y=y1, width=max(0.0, x2 - x1), height=max(0.0, y2 - y1))

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.x2, self.y2

    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def intersects(self, other: "BoundingBox", min_overlap_ratio: float = 0.1) -> bool:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if intersection <= 0:
            return False
        smaller = min(self.area(), other.area())
        if smaller <= 0:
            return False
        return intersection / smaller >= min_overlap_ratio

    def clamp(self, width: int, height: int) -> "BoundingBox":
        x1 = min(max(self.x, 0), width)
        y1 = min(max(self.y, 0), height)
        x2 = min(max(self.x2, 0), width)
        y2 = min(max(self.y2, 0), height)
        return BoundingBox.from_xyxy(x1, y1, x2, y2)


class LabelPolicy(BaseModel):
    label: str
    action: ExtractionAction
    description: str = ""


DEFAULT_LABEL_POLICIES: dict[str, LabelPolicy] = {
    "title": LabelPolicy(label="title", action=ExtractionAction.OCR, description="Heading/title region"),
    "body": LabelPolicy(label="body", action=ExtractionAction.OCR, description="Main body text"),
    "plain text": LabelPolicy(label="plain text", action=ExtractionAction.OCR, description="Generic text region"),
    "text": LabelPolicy(label="text", action=ExtractionAction.OCR, description="Generic text region"),
    "table": LabelPolicy(label="table", action=ExtractionAction.TABLE, description="Table region"),
    "figure": LabelPolicy(label="figure", action=ExtractionAction.SKIP, description="Figure/image region"),
    "image": LabelPolicy(label="image", action=ExtractionAction.SKIP, description="Image region"),
    "header": LabelPolicy(label="header", action=ExtractionAction.SKIP, description="Repeating header"),
    "footer": LabelPolicy(label="footer", action=ExtractionAction.SKIP, description="Repeating footer"),
    "ignore": LabelPolicy(label="ignore", action=ExtractionAction.SKIP, description="Explicitly ignored region"),
    "target_data": LabelPolicy(label="target_data", action=ExtractionAction.OCR, description="High priority target data"),
    # DocStructBench class names, emitted as-is by the pretrained DocLayout-YOLO
    # checkpoint. Without an entry here they fall through to the OCR default in
    # `pipeline.build_policies`, which means `abandon` -- running heads, page
    # numbers, the arXiv stamp -- gets extracted instead of discarded. Actions
    # follow the same judgement as `labels.DOCSTRUCTBENCH_TO_CANONICAL`.
    "abandon": LabelPolicy(label="abandon", action=ExtractionAction.SKIP, description="DocStructBench discardable furniture (running head, page number)"),
    "figure_caption": LabelPolicy(label="figure_caption", action=ExtractionAction.OCR, description="Figure caption prose"),
    "table_caption": LabelPolicy(label="table_caption", action=ExtractionAction.OCR, description="Table caption prose"),
    "table_footnote": LabelPolicy(label="table_footnote", action=ExtractionAction.OCR, description="Table footnote prose"),
    "isolate_formula": LabelPolicy(label="isolate_formula", action=ExtractionAction.SKIP, description="Display formula; OCR cannot read it"),
    "formula_caption": LabelPolicy(label="formula_caption", action=ExtractionAction.OCR, description="Formula caption/number"),
}


class DetectedRegion(BaseModel):
    label: str
    bbox: BoundingBox
    confidence: float = 0.0
    source: str = "unknown"
    raw_label_id: int | None = None


class ParsedRegion(BaseModel):
    region_id: str = Field(default_factory=lambda: str(uuid4()))
    page_number: int
    label: str
    bbox: BoundingBox
    confidence: float
    action: ExtractionAction
    source: str
    text: str = ""
    table_data: list[list[str]] | None = None
    notes: list[str] = Field(default_factory=list)


class PageParseResult(BaseModel):
    page_number: int
    image_path: str
    width: int
    height: int
    regions: list[ParsedRegion] = Field(default_factory=list)


class DocumentParseResult(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    input_path: str
    output_dir: str
    detector: str
    model_path: str | None = None
    pages: list[PageParseResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AnnotationSource(str, Enum):
    """Where a box came from.

    Needed to compare label provenance as an experiment axis: pure manual
    labelling, model predictions accepted as-is (self-training), and model
    predictions corrected by a human (HITL). Without this tag the three are
    indistinguishable once exported and cannot be compared.
    """

    MANUAL = "manual"
    PREDICTION = "prediction"
    CORRECTION = "correction"


class LabelStatus(str, Enum):
    """Whether a page's annotation is finished.

    Resolves the "empty page" ambiguity: previously a page with zero boxes
    exported as an empty YOLO label file, i.e. a pure-background negative sample.
    A half-labelled page was therefore indistinguishable from a genuinely blank
    one, and partial labelling silently corrupted both training and ground truth.
    Only COMPLETE pages are exported or evaluated; a COMPLETE page with zero
    boxes is a real negative and is kept deliberately.
    """

    UNLABELED = "unlabeled"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


class AnnotationBox(BaseModel):
    label: str
    bbox: BoundingBox
    source: AnnotationSource = AnnotationSource.MANUAL


class AnnotationPage(BaseModel):
    image_path: str
    width: int
    height: int
    annotations: list[AnnotationBox] = Field(default_factory=list)
    # Document-level identity, required for leak-free splits. Optional so existing
    # annotation files keep loading; `splits.document_id_for` falls back to the
    # image path when it is absent.
    document_id: str | None = None
    page_number: int | None = None
    label_status: LabelStatus = LabelStatus.COMPLETE


class AnnotationDataset(BaseModel):
    labels: list[str]
    pages: list[AnnotationPage]


class YoloExportResult(BaseModel):
    output_dir: str
    images_dir: str
    labels_dir: str
    train_txt: str
    val_txt: str
    data_yaml: str
    labels: list[str]
    train_count: int
    val_count: int
    page_count: int
    test_txt: str | None = None
    test_count: int = 0
    # Canonical class ids actually written, so a dataset can be checked against
    # the catalogue after the fact.
    class_ids: dict[str, int] = Field(default_factory=dict)
    dataset_fingerprint: str | None = None
    skipped_incomplete_pages: int = 0
    dropped_boxes: int = 0
    warnings: list[str] = Field(default_factory=list)


class SplitSummary(BaseModel):
    seed: int
    strategy: str
    train_pages: int
    val_pages: int
    test_pages: int
    train_documents: list[str] = Field(default_factory=list)
    val_documents: list[str] = Field(default_factory=list)
    test_documents: list[str] = Field(default_factory=list)


class ClassDetectionMetrics(BaseModel):
    label: str
    gt_count: int
    pred_count: int
    # None means "no ground truth for this class", which is not the same as 0.0.
    ap50: float | None = None
    ap5095: float | None = None


class DetectionMetrics(BaseModel):
    per_class: list[ClassDetectionMetrics] = Field(default_factory=list)
    map50: float | None = None
    map5095: float | None = None
    iou_thresholds: list[float] = Field(default_factory=list)
    score_threshold: float = 0.0
    evaluated_labels: list[str] = Field(default_factory=list)


class ConfusionCell(BaseModel):
    gt_label: str
    pred_label: str
    count: int


class PageOperationalMetrics(BaseModel):
    image_id: str
    page_area: float
    gt_keep_area: float
    pred_keep_area: float
    loss_area: float
    contamination_area: float


class OperationalMetrics(BaseModel):
    variant: str
    page_count: int
    page_area: float
    gt_keep_area: float
    pred_keep_area: float
    loss_area: float
    contamination_area: float
    reduction_rate: float
    loss_rate: float | None = None
    contamination_rate: float | None = None


class EvaluationReport(BaseModel):
    """Everything needed to reproduce and compare one evaluated cell."""

    run_name: str
    detector: str
    model_path: str | None = None
    label_mapping: str = "identity"
    label_catalogue: list[str] = Field(default_factory=list)
    catalogue_fingerprint: str | None = None
    dataset_fingerprint: str | None = None
    annotation_path: str | None = None
    page_count: int = 0
    gt_box_count: int = 0
    pred_box_count: int = 0
    eval_confidence: float = 0.001
    operating_confidence: float = 0.25
    seed: int | None = None
    split: SplitSummary | None = None
    # Which split was scored ("test"/"val"/...), and the experiment coordinates of
    # this cell (e.g. {"n": "20", "init": "docstructbench", "labels": "manual"}).
    # Recorded so a ledger row can be traced back to a position in the matrix.
    scored_split: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    detection: DetectionMetrics | None = None
    operational: list[OperationalMetrics] = Field(default_factory=list)
    confusion: list[ConfusionCell] = Field(default_factory=list)
    per_page_operational: list[PageOperationalMetrics] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    duration_seconds: float | None = None


class ParseRequestOptions(BaseModel):
    detector: Literal["heuristic", "doclayout-yolo"] = "heuristic"
    model_path: Path | None = None
    device: str = "cpu"
    image_size: int = 1024
    confidence: float = 0.25
    dpi: int = 180
    max_pages: int | None = None
    run_ocr: bool = True
    policy_overrides: dict[str, ExtractionAction] = Field(default_factory=dict)


JsonDict = dict[str, Any]


class TrainingConfig(BaseModel):
    data_yaml: Path
    output_dir: Path = Path("runs/training")
    run_name: str = "doclayout_finetune"
    model_size: str = "n"
    pretrain: Path | str | None = None
    epochs: int = 30
    batch_size: int = 4
    image_size: int = 1024
    device: str = "cpu"
    workers: int = 0
    learning_rate: float = 0.02
    optimizer: str = "auto"
    momentum: float = 0.9
    warmup_epochs: float = 3.0
    mosaic: float = 1.0
    run_validation: bool = True
    plots: bool = False
    save_period: int = 10
    patience: int = 100
    exist_ok: bool = True
    dry_run: bool = False


class TrainingResult(BaseModel):
    status: Literal["dry_run", "completed", "failed"]
    data_yaml: str
    output_dir: str
    run_name: str
    model_source: str
    best_model_path: str | None = None
    last_model_path: str | None = None
    command_preview: str
    metrics: JsonDict = Field(default_factory=dict)
    error: str | None = None


class ExperimentResult(BaseModel):
    sample_size: int
    dataset: YoloExportResult
    training: TrainingResult
