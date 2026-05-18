from __future__ import annotations

from pathlib import Path
from typing import Any

from pdf_layout_parser.dataset import export_yolo_dataset
from pdf_layout_parser.models import ExperimentResult, TrainingConfig, TrainingResult


def run_doclayout_yolo_training(config: TrainingConfig) -> TrainingResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_source = resolve_model_source(config)
    command_preview = build_training_preview(config, model_source)

    if config.dry_run:
        return TrainingResult(
            status="dry_run",
            data_yaml=str(config.data_yaml),
            output_dir=str(config.output_dir),
            run_name=config.run_name,
            model_source=model_source,
            command_preview=command_preview,
        )

    try:
        from doclayout_yolo import YOLOv10  # type: ignore
    except ImportError as exc:
        return TrainingResult(
            status="failed",
            data_yaml=str(config.data_yaml),
            output_dir=str(config.output_dir),
            run_name=config.run_name,
            model_source=model_source,
            command_preview=command_preview,
            error="doclayout-yolo is not installed. Install with `pip install -e .[doclayout]`.",
        )

    try:
        model = YOLOv10(model_source)
        train_output = model.train(
            data=str(config.data_yaml),
            epochs=config.epochs,
            warmup_epochs=config.warmup_epochs,
            lr0=config.learning_rate,
            optimizer=config.optimizer,
            momentum=config.momentum,
            imgsz=config.image_size,
            mosaic=config.mosaic,
            batch=config.batch_size,
            device=config.device,
            workers=config.workers,
            plots=config.plots,
            exist_ok=config.exist_ok,
            val=config.run_validation,
            save_period=config.save_period,
            patience=config.patience,
            project=str(config.output_dir),
            name=config.run_name,
        )
        run_dir = config.output_dir / config.run_name
        best_model_path = find_weight(run_dir, "best.pt")
        last_model_path = find_weight(run_dir, "last.pt")
        return TrainingResult(
            status="completed",
            data_yaml=str(config.data_yaml),
            output_dir=str(config.output_dir),
            run_name=config.run_name,
            model_source=model_source,
            best_model_path=str(best_model_path) if best_model_path else None,
            last_model_path=str(last_model_path) if last_model_path else None,
            command_preview=command_preview,
            metrics=extract_metrics(train_output),
        )
    except Exception as exc:
        return TrainingResult(
            status="failed",
            data_yaml=str(config.data_yaml),
            output_dir=str(config.output_dir),
            run_name=config.run_name,
            model_source=model_source,
            command_preview=command_preview,
            error=str(exc),
        )


def resolve_model_source(config: TrainingConfig) -> str:
    if config.pretrain is None:
        return f"yolov10{config.model_size}.yaml"
    pretrain = config.pretrain.as_posix() if isinstance(config.pretrain, Path) else str(config.pretrain)
    if pretrain.lower() == "coco":
        return f"yolov10{config.model_size}.pt"
    return pretrain


def build_training_preview(config: TrainingConfig, model_source: str | None = None) -> str:
    source = model_source or resolve_model_source(config)
    return (
        "YOLOv10("
        f"{source!r}"
        ").train("
        f"data={str(config.data_yaml)!r}, "
        f"epochs={config.epochs}, "
        f"imgsz={config.image_size}, "
        f"batch={config.batch_size}, "
        f"device={config.device!r}, "
        f"project={str(config.output_dir)!r}, "
        f"name={config.run_name!r}"
        ")"
    )


def find_weight(run_dir: Path, filename: str) -> Path | None:
    direct = run_dir / "weights" / filename
    if direct.exists():
        return direct
    matches = list(run_dir.glob(f"**/{filename}"))
    return matches[0] if matches else None


def extract_metrics(train_output: Any) -> dict[str, Any]:
    if train_output is None:
        return {}
    if isinstance(train_output, dict):
        return train_output
    results_dict = getattr(train_output, "results_dict", None)
    if isinstance(results_dict, dict):
        return results_dict
    metrics = getattr(train_output, "metrics", None)
    if isinstance(metrics, dict):
        return metrics
    return {"repr": repr(train_output)}


def run_page_count_experiments(
    annotation_path: Path,
    output_dir: Path,
    sample_sizes: list[int],
    training_config: TrainingConfig,
    train_ratio: float = 0.8,
    run_training: bool = False,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_size in sample_sizes:
        dataset_dir = output_dir / f"pages_{sample_size}" / "dataset"
        dataset = export_yolo_dataset(
            annotation_path=annotation_path,
            output_dir=dataset_dir,
            max_pages=sample_size,
            train_ratio=train_ratio,
        )
        config = training_config.model_copy(
            update={
                "data_yaml": Path(dataset.data_yaml),
                "output_dir": output_dir / f"pages_{sample_size}" / "training",
                "run_name": f"doclayout_pages_{sample_size}",
                "dry_run": training_config.dry_run or not run_training,
            }
        )
        training = run_doclayout_yolo_training(config)
        results.append(ExperimentResult(sample_size=sample_size, dataset=dataset, training=training))

    return results
