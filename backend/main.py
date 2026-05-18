from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pdf_layout_parser.dataset import export_yolo_dataset
from pdf_layout_parser.models import DocumentParseResult, ParseRequestOptions, TrainingConfig
from pdf_layout_parser.pipeline import parse_document
from pdf_layout_parser.training import run_doclayout_yolo_training

app = FastAPI(title="DocsOCR Backend", version="0.1.0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FALLBACK_WEB_DIR = PROJECT_ROOT / "src" / "pdf_layout_parser" / "web"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
RUNS_DIR = PROJECT_ROOT / "runs"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_DOCLAYOUT_MODEL = MODELS_DIR / "doclayout_yolo_docstructbench_imgsz1024.pt"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

ASSETS_DIR = FRONTEND_DIST / "assets" if (FRONTEND_DIST / "assets").exists() else FALLBACK_WEB_DIR

app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")


@app.get("/")
def index() -> FileResponse:
    built_index = FRONTEND_DIST / "index.html"
    if built_index.exists():
        return FileResponse(built_index)
    return FileResponse(FALLBACK_WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents/parse")
async def parse_pdf(
    file: UploadFile = File(...),
    detector: str = Form("heuristic"),
    model_path: str | None = Form(None),
    device: str = Form("cpu"),
    max_pages: int | None = Form(None),
    run_ocr: bool = Form(True),
) -> JSONResponse:
    run_id = str(uuid4())
    run_dir = RUNS_DIR / "web" / run_id
    upload_dir = run_dir / "uploads"
    output_dir = run_dir / "output"
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename or "document.pdf").name
    pdf_path = upload_dir / safe_name
    pdf_path.write_bytes(await file.read())

    options = ParseRequestOptions(
        detector=detector,  # type: ignore[arg-type]
        model_path=resolve_model_path(detector, model_path),
        device=device,
        max_pages=max_pages,
        run_ocr=run_ocr,
    )
    result = parse_document(pdf_path, output_dir, options)
    return JSONResponse(to_web_result(result))


@app.post("/annotations/export-yolo")
async def export_annotations(
    file: UploadFile = File(...),
    output_dir: str = Form("runs/api-yolo-dataset"),
    max_pages: int | None = Form(None),
    train_ratio: float = Form(0.8),
) -> JSONResponse:
    with tempfile.TemporaryDirectory(prefix="pdf-yolo-dataset-") as tmp:
        annotations_path = Path(tmp) / (file.filename or "annotations.json")
        annotations_path.write_bytes(await file.read())
        dataset_dir = Path(output_dir)
        paths = export_yolo_dataset(
            annotations_path,
            dataset_dir,
            max_pages=max_pages,
            train_ratio=train_ratio,
        )
        return JSONResponse(paths.model_dump(mode="json"))


@app.post("/training/fine-tune")
async def fine_tune_doclayout_yolo(
    file: UploadFile = File(...),
    output_dir: str = Form("runs/api-training"),
    max_pages: int | None = Form(None),
    train_ratio: float = Form(0.8),
    model_size: str = Form("n"),
    pretrain: str | None = Form(None),
    epochs: int = Form(30),
    batch_size: int = Form(4),
    image_size: int = Form(1024),
    device: str = Form("cpu"),
    dry_run: bool = Form(True),
) -> JSONResponse:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    annotations_path = output_path / (file.filename or "annotations.json")
    annotations_path.write_bytes(await file.read())

    dataset = export_yolo_dataset(
        annotations_path,
        output_path / "dataset",
        max_pages=max_pages,
        train_ratio=train_ratio,
    )
    training = run_doclayout_yolo_training(
        TrainingConfig(
            data_yaml=Path(dataset.data_yaml),
            output_dir=output_path / "training",
            model_size=model_size,
            pretrain=pretrain,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            device=device,
            dry_run=dry_run,
        )
    )
    return JSONResponse(
        {
            "dataset": dataset.model_dump(mode="json"),
            "training": training.model_dump(mode="json"),
        }
    )


def resolve_model_path(detector: str, model_path: str | None) -> Path | None:
    if model_path:
        path = Path(model_path)
        return path if path.is_absolute() else PROJECT_ROOT / path
    if detector == "doclayout-yolo" and DEFAULT_DOCLAYOUT_MODEL.exists():
        return DEFAULT_DOCLAYOUT_MODEL
    return None


def to_web_result(result: DocumentParseResult) -> dict:
    payload = result.model_dump(mode="json")
    output_dir = Path(result.output_dir)
    payload["result_json_url"] = path_to_runs_url(output_dir / "result.json")
    payload["result_markdown_url"] = path_to_runs_url(output_dir / "result.md")

    for page in payload["pages"]:
        page["image_url"] = path_to_runs_url(Path(page["image_path"]))
    return payload


def path_to_runs_url(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(RUNS_DIR)
    except ValueError:
        return ""
    return "/runs/" + relative.as_posix()

