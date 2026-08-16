"""One-shot fine-tune-and-compare: the smallest honest before/after loop.

Splits the annotated pages, fine-tunes on the train portion, then scores the
baseline model and the tuned checkpoint on the SAME held-out test pages with the
existing evaluation harness. This is the web UI's "did my corrections buy
anything" answer, and it deliberately reuses the measurement code path the CLI
experiments use -- a second, simpler scoring路 would drift.

Honesty constraints baked in rather than left to the caller:

* Only pages marked ``label_status: complete`` participate (C5).
* Both models are scored on test pages only, never on pages that trained.
* The web corpus is typically ONE document, so the default split strategy is
  ``page`` -- held-out pages of the same document. That matches the MVP claim
  ("annotate 10-20 pages, work on unseen pages of the same family") but it is
  NOT comparable with document-level CLI experiments, and the report's split
  summary says which strategy produced it.
* The baseline is the pretrain checkpoint scored through the DocStructBench
  label mapping; with no pretrain the baseline falls back to the heuristic
  detector and training starts from scratch (with the usual loud warning).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from pdf_layout_parser.dataset import export_yolo_dataset
from pdf_layout_parser.evaluation.harness import (
    evaluate,
    generate_predictions,
    load_annotation_pages,
    load_ground_truth,
)
from pdf_layout_parser.models import (
    EvaluationReport,
    FinetuneComparison,
    FinetuneVariantSummary,
    TrainingConfig,
)
from pdf_layout_parser.splits import make_splits

ProgressFn = Callable[[str], None]


def _summarize(variant: str, report: EvaluationReport) -> FinetuneVariantSummary:
    model_row = next((row for row in report.operational if row.variant == "model"), None)
    return FinetuneVariantSummary(
        variant=variant,
        detector=report.detector,
        model_path=report.model_path,
        label_mapping=report.label_mapping,
        pages=report.page_count,
        map50=report.detection.map50 if report.detection else None,
        map5095=report.detection.map5095 if report.detection else None,
        reduction_rate=model_row.reduction_rate if model_row else None,
        loss_rate=model_row.loss_rate if model_row else None,
        contamination_rate=model_row.contamination_rate if model_row else None,
    )


def run_finetune_comparison(
    annotation_path: Path,
    output_dir: Path,
    pretrain: Path | None = None,
    epochs: int = 20,
    image_size: int = 1024,
    device: str = "cpu",
    seed: int = 0,
    split_strategy: str = "page",
    test_ratio: float = 0.35,
    val_ratio: float = 0.15,
    progress: ProgressFn | None = None,
) -> FinetuneComparison:
    def report_phase(phase: str) -> None:
        if progress:
            progress(phase)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_phase("splitting")
    pages = load_annotation_pages(annotation_path).pages
    splits = make_splits(
        pages,
        seed=seed,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        strategy=split_strategy,
    )

    report_phase("exporting dataset")
    dataset = export_yolo_dataset(
        annotation_path=annotation_path,
        output_dir=output_dir / "dataset",
        splits=splits,
        strict=True,
    )

    report_phase(f"training ({epochs} epochs)")
    config = TrainingConfig(
        data_yaml=Path(dataset.data_yaml),
        output_dir=output_dir / "training",
        run_name=f"web_finetune_seed{seed}",
        pretrain=pretrain,
        epochs=epochs,
        image_size=image_size,
        device=device,
        seed=seed,
    )
    training = run_training(config)
    if training.status != "completed" or not training.best_model_path:
        raise RuntimeError(
            f"Training did not produce a checkpoint (status={training.status}): "
            f"{training.error or 'no best.pt found'}"
        )

    ground_truth = load_ground_truth(pages=splits.test)

    # Baseline: what you had before annotating anything. With a pretrain
    # checkpoint that is zero-shot DocStructBench (through the label mapping);
    # without one the incumbent is the heuristic detector.
    report_phase("evaluating baseline")
    if pretrain is not None:
        baseline_preds, baseline_mapping, baseline_warnings = generate_predictions(
            ground_truth.annotation_pages,
            detector_name="doclayout-yolo",
            model_path=pretrain,
            device=device,
            image_size=image_size,
            label_mapping="docstructbench",
        )
        baseline_detector, baseline_model = "doclayout-yolo", pretrain
    else:
        baseline_preds, baseline_mapping, baseline_warnings = generate_predictions(
            ground_truth.annotation_pages,
            detector_name="heuristic",
            label_mapping="identity",
        )
        baseline_detector, baseline_model = "heuristic", None
    baseline_report = evaluate(
        ground_truth,
        baseline_preds,
        run_name="baseline",
        detector=baseline_detector,
        model_path=baseline_model,
        label_mapping=baseline_mapping,
        seed=seed,
        split=splits.summary(),
        annotation_path=annotation_path,
        scored_split="test",
        notes=[*splits.warnings, *baseline_warnings],
    )

    report_phase("evaluating tuned model")
    tuned_path = Path(training.best_model_path)
    tuned_preds, tuned_mapping, tuned_warnings = generate_predictions(
        ground_truth.annotation_pages,
        detector_name="doclayout-yolo",
        model_path=tuned_path,
        device=device,
        image_size=image_size,
        label_mapping="identity",
    )
    tuned_report = evaluate(
        ground_truth,
        tuned_preds,
        run_name="tuned",
        detector="doclayout-yolo",
        model_path=tuned_path,
        label_mapping=tuned_mapping,
        seed=seed,
        split=splits.summary(),
        annotation_path=annotation_path,
        scored_split="test",
        notes=[*splits.warnings, *tuned_warnings],
    )

    comparison = FinetuneComparison(
        output_dir=str(output_dir),
        split=splits.summary(),
        split_warnings=list(splits.warnings),
        dataset=dataset,
        training=training,
        best_model_path=training.best_model_path,
        baseline=baseline_report,
        tuned=tuned_report,
        summary=[
            _summarize("baseline", baseline_report),
            _summarize("tuned", tuned_report),
        ],
    )
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return comparison


def run_training(config: TrainingConfig):
    """Seam for tests: patched to fake a checkpoint without torch."""
    from pdf_layout_parser.training import run_doclayout_yolo_training

    return run_doclayout_yolo_training(config)
