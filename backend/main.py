from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.jobs import REGISTRY
from pdf_layout_parser.dataset import export_yolo_dataset
from pdf_layout_parser.models import (
    AnnotationDataset,
    DocumentParseResult,
    LabelStatus,
    ParseRequestOptions,
)
from pdf_layout_parser.pipeline import parse_document

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


class FinetuneRequest(BaseModel):
    """Web quick-loop request: the annotation payload plus a few knobs."""

    annotations: AnnotationDataset
    epochs: int = 20
    image_size: int = 1024
    device: str = "cpu"
    seed: int = 0
    pretrain: str | None = None
    test_ratio: float = 0.35
    val_ratio: float = 0.15


@app.post("/training/fine-tune")
def fine_tune_and_compare(request: FinetuneRequest) -> JSONResponse:
    """Fine-tune on the completed pages and score before/after on held-out pages.

    Returns a job id immediately; poll /training/jobs/{job_id}. The previous
    version of this endpoint ran the blocking training inside the async handler,
    freezing the whole event loop for the duration (FINDINGS M6), used the legacy
    leaky split, and had no comparison at all.
    """
    running = REGISTRY.running_job("finetune")
    if running:
        return JSONResponse(
            {"detail": f"A fine-tune job is already running ({running}); wait for it to finish."},
            status_code=409,
        )

    complete = [page for page in request.annotations.pages if page.label_status == LabelStatus.COMPLETE]
    boxes = sum(len(page.annotations) for page in complete)
    if len(complete) < 3:
        return JSONResponse(
            {
                "detail": f"Need at least 3 pages marked as Labeled to split train/val/test; "
                f"got {len(complete)}."
            },
            status_code=422,
        )
    if boxes == 0:
        return JSONResponse({"detail": "No annotation boxes on the Labeled pages."}, status_code=422)

    run_dir = RUNS_DIR / "web-training" / str(uuid4())
    run_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = run_dir / "annotations.json"
    annotation_path.write_text(
        json.dumps(request.annotations.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pretrain = resolve_model_path("doclayout-yolo", request.pretrain)

    def work(set_phase) -> dict:
        payload = execute_comparison_job(run_dir, annotation_path, request, pretrain, set_phase)
        # The web client needs a path it can paste back into "Model path";
        # keep it project-relative when possible.
        best = payload.get("best_model_path")
        if best:
            try:
                payload["best_model_relpath"] = Path(best).resolve().relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                payload["best_model_relpath"] = best
        return payload

    job_id = REGISTRY.submit("finetune", run_dir, work)
    return JSONResponse({"job_id": job_id, "output_dir": str(run_dir)})


def execute_comparison_job(
    run_dir: Path,
    annotation_path: Path,
    request: FinetuneRequest,
    pretrain: Path | None,
    set_phase,
) -> dict:
    """Run the comparison in a CHILD PROCESS and return its payload.

    Not a thread: torch inference on a non-main thread segfaulted the whole
    server on macOS (exit 139, during the post-train validation pass). In a
    subprocess a native crash is contained to the job. Phases come back through
    a file the child rewrites; the result through comparison.json.
    """
    phase_file = run_dir / "phase.txt"
    worker_log = run_dir / "worker.log"
    command = [
        sys.executable,
        "-m",
        "pdf_layout_parser.comparison",
        "--annotations", str(annotation_path),
        "--output-dir", str(run_dir),
        "--epochs", str(request.epochs),
        "--image-size", str(request.image_size),
        "--device", request.device,
        "--seed", str(request.seed),
        "--test-ratio", str(request.test_ratio),
        "--val-ratio", str(request.val_ratio),
        "--phase-file", str(phase_file),
    ]
    if pretrain is not None:
        command += ["--pretrain", str(pretrain)]

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    with worker_log.open("wb") as log:
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env
        )
        last_phase = ""
        while process.poll() is None:
            time.sleep(1.0)
            try:
                phase = phase_file.read_text(encoding="utf-8").strip()
            except OSError:
                phase = ""
            if phase and phase != last_phase:
                last_phase = phase
                set_phase(phase)

    if process.returncode != 0:
        tail = ""
        try:
            tail = "\n".join(
                worker_log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
            )
        except OSError:
            pass
        raise RuntimeError(f"Fine-tune worker exited with code {process.returncode}.\n{tail}")

    return json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))


@app.get("/training/jobs/{job_id}")
def training_job_status(job_id: str) -> JSONResponse:
    snapshot = REGISTRY.get(job_id)
    if snapshot is None:
        return JSONResponse({"detail": f"Unknown job {job_id}"}, status_code=404)
    return JSONResponse(snapshot)


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

