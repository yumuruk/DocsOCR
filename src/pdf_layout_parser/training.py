from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pdf_layout_parser.dataset import export_yolo_dataset, load_annotation_dataset
from pdf_layout_parser.models import ExperimentResult, JsonDict, TrainingConfig, TrainingResult
from pdf_layout_parser.splits import make_splits, with_train_sample


def training_kwargs(config: TrainingConfig) -> JsonDict:
    """The exact keyword arguments passed to ``YOLOv10.train()``.

    Single source of truth: the live call, the recorded ``train_kwargs``, and the
    human-readable preview are all rendered from this one dict, so the record can
    never drift from what actually ran. Previously the preview was assembled by
    hand from a subset of the config and omitted seed, freeze, lr, and most of the
    rest -- a reproducibility record that under-describes the run is a trap.
    """
    kwargs: JsonDict = {
        "data": str(config.data_yaml),
        "epochs": config.epochs,
        "warmup_epochs": config.warmup_epochs,
        "lr0": config.learning_rate,
        "optimizer": config.optimizer,
        "momentum": config.momentum,
        "imgsz": config.image_size,
        "mosaic": config.mosaic,
        "batch": config.batch_size,
        "device": config.device,
        "workers": config.workers,
        "seed": config.seed,
        "plots": config.plots,
        "exist_ok": config.exist_ok,
        "val": config.run_validation,
        "save_period": config.save_period,
        "patience": config.patience,
        "project": str(config.output_dir),
        "name": config.run_name,
    }
    if config.freeze is not None:
        kwargs["freeze"] = config.freeze
    return kwargs


def run_doclayout_yolo_training(config: TrainingConfig) -> TrainingResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_source = resolve_model_source(config)
    kwargs = training_kwargs(config)
    command_preview = build_training_preview(config, model_source)

    warnings: list[str] = []
    if config.pretrain is None:
        # M3: the default silently trained from random init under a run name that
        # said "finetune". Allowed, but never again without saying so.
        warnings.append(
            f"pretrain is not set: training {model_source} FROM SCRATCH (random init). "
            "Pass --pretrain with the DocStructBench checkpoint to fine-tune."
        )

    def finalize(result: TrainingResult) -> TrainingResult:
        return _write_training_result(result, config)

    if config.dry_run:
        return finalize(
            TrainingResult(
                status="dry_run",
                data_yaml=str(config.data_yaml),
                output_dir=str(config.output_dir),
                run_name=config.run_name,
                model_source=model_source,
                command_preview=command_preview,
                seed=config.seed,
                freeze=config.freeze,
                train_kwargs=kwargs,
                warnings=warnings,
            )
        )

    try:
        from doclayout_yolo import YOLOv10  # type: ignore
    except ImportError:
        return finalize(
            TrainingResult(
                status="failed",
                data_yaml=str(config.data_yaml),
                output_dir=str(config.output_dir),
                run_name=config.run_name,
                model_source=model_source,
                command_preview=command_preview,
                seed=config.seed,
                freeze=config.freeze,
                train_kwargs=kwargs,
                warnings=warnings,
                error="doclayout-yolo is not installed. Install with `pip install -e .[doclayout]`.",
            )
        )

    started = time.perf_counter()
    try:
        model = YOLOv10(model_source)
        train_output = model.train(**kwargs)
        duration = time.perf_counter() - started
        run_dir = config.output_dir / config.run_name
        best_model_path = find_weight(run_dir, "best.pt")
        last_model_path = find_weight(run_dir, "last.pt")
        return finalize(
            TrainingResult(
                status="completed",
                data_yaml=str(config.data_yaml),
                output_dir=str(config.output_dir),
                run_name=config.run_name,
                model_source=model_source,
                best_model_path=str(best_model_path) if best_model_path else None,
                last_model_path=str(last_model_path) if last_model_path else None,
                command_preview=command_preview,
                metrics=extract_metrics(train_output),
                seed=config.seed,
                freeze=config.freeze,
                train_kwargs=kwargs,
                duration_seconds=duration,
                warnings=warnings,
            )
        )
    except Exception as exc:
        return finalize(
            TrainingResult(
                status="failed",
                data_yaml=str(config.data_yaml),
                output_dir=str(config.output_dir),
                run_name=config.run_name,
                model_source=model_source,
                command_preview=command_preview,
                seed=config.seed,
                freeze=config.freeze,
                train_kwargs=kwargs,
                duration_seconds=time.perf_counter() - started,
                warnings=warnings,
                error=str(exc),
            )
        )


def _write_training_result(result: TrainingResult, config: TrainingConfig) -> TrainingResult:
    """Persist the result next to the run and record where.

    Previously the result object was returned, printed to stdout, and lost: a
    finished (possibly hours-long) run left no machine-readable record on disk.
    """
    path = config.output_dir / f"{config.run_name}_result.json"
    result.result_path = str(path)
    path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def resolve_model_source(config: TrainingConfig) -> str:
    if config.pretrain is None:
        return f"yolov10{config.model_size}.yaml"
    pretrain = config.pretrain.as_posix() if isinstance(config.pretrain, Path) else str(config.pretrain)
    if pretrain.lower() == "coco":
        return f"yolov10{config.model_size}.pt"
    return pretrain


def build_training_preview(config: TrainingConfig, model_source: str | None = None) -> str:
    source = model_source or resolve_model_source(config)
    kwargs = training_kwargs(config)
    rendered = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
    return f"YOLOv10({source!r}).train({rendered})"


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
    split_seed: int = 0,
    sample_seed: int = 0,
    test_ratio: float = 0.35,
    val_ratio: float = 0.15,
    split_strategy: str = "document",
    run_training: bool = False,
) -> list[ExperimentResult]:
    """Label-efficiency experiment: one cell per sample size.

    The splits are made ONCE and every cell shares the same val and test sets;
    only the train set changes, by nested sampling (N=10 pages are a subset of
    N=20's). The previous version re-split a "first N pages" prefix per cell, so
    each N was validated on different pages and the resulting curve compared
    nothing with nothing (FINDINGS C2). Datasets are exported with strict=True:
    if the corpus cannot be split safely, the experiment refuses to start rather
    than producing incomparable cells.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = load_annotation_dataset(annotation_path).pages
    base_splits = make_splits(
        pages,
        seed=split_seed,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        strategy=split_strategy,
    )

    results: list[ExperimentResult] = []
    for sample_size in sorted(sample_sizes):
        cell_splits = with_train_sample(base_splits, sample_size, seed=sample_seed)
        dataset_dir = output_dir / f"pages_{sample_size}" / "dataset"
        dataset = export_yolo_dataset(
            annotation_path=annotation_path,
            output_dir=dataset_dir,
            splits=cell_splits,
            strict=True,
        )
        config = training_config.model_copy(
            update={
                "data_yaml": Path(dataset.data_yaml),
                "output_dir": output_dir / f"pages_{sample_size}" / "training",
                "run_name": f"doclayout_pages_{sample_size}_seed{training_config.seed}",
                "dry_run": training_config.dry_run or not run_training,
            }
        )
        training = run_doclayout_yolo_training(config)
        results.append(
            ExperimentResult(
                sample_size=sample_size,
                dataset=dataset,
                training=training,
                split=cell_splits.summary(),
                sample_seed=sample_seed,
            )
        )

    manifest = output_dir / "experiments.json"
    manifest.write_text(
        json.dumps([result.model_dump(mode="json") for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results
