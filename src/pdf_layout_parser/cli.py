from __future__ import annotations

import argparse
import json
from pathlib import Path

from pdf_layout_parser.dataset import export_yolo_dataset
from pdf_layout_parser.evaluation import (
    EVAL_CONFIDENCE,
    OPERATING_CONFIDENCE,
    evaluate,
    generate_predictions,
    load_ground_truth,
    load_predictions,
    save_predictions,
    write_report,
)
from pdf_layout_parser.evaluation import render_comparison
from pdf_layout_parser.evaluation.harness import load_annotation_pages
from pdf_layout_parser.experiment_log import LEDGER_PATH, append_run, load_reports, parse_tags
from pdf_layout_parser.models import ParseRequestOptions, TrainingConfig
from pdf_layout_parser.pipeline import parse_document
from pdf_layout_parser.splits import make_splits
from pdf_layout_parser.training import run_doclayout_yolo_training, run_page_count_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotation-driven PDF layout parser prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Parse a PDF into structured JSON/Markdown")
    parse_parser.add_argument("pdf", type=Path)
    parse_parser.add_argument("--output-dir", type=Path, default=Path("runs/latest"))
    parse_parser.add_argument("--detector", choices=["heuristic", "doclayout-yolo"], default="heuristic")
    parse_parser.add_argument("--model-path", type=Path)
    parse_parser.add_argument("--device", default="cpu")
    parse_parser.add_argument("--image-size", type=int, default=1024)
    parse_parser.add_argument("--confidence", type=float, default=0.25)
    parse_parser.add_argument("--dpi", type=int, default=180)
    parse_parser.add_argument("--max-pages", type=int)
    parse_parser.add_argument("--no-ocr", action="store_true")

    export_parser = subparsers.add_parser("export-yolo", help="Convert annotation JSON to YOLO dataset")
    export_parser.add_argument("annotations", type=Path)
    export_parser.add_argument("output_dir", type=Path)
    export_parser.add_argument("--max-pages", type=int)
    export_parser.add_argument("--train-ratio", type=float, default=0.8)

    train_parser = subparsers.add_parser("train", help="Export annotations and fine-tune DocLayout-YOLO")
    train_parser.add_argument("annotations", type=Path)
    train_parser.add_argument("output_dir", type=Path)
    train_parser.add_argument("--max-pages", type=int)
    train_parser.add_argument("--train-ratio", type=float, default=0.8)
    add_training_args(train_parser)

    experiment_parser = subparsers.add_parser("experiment", help="Run 10/20/etc page-count fine-tuning experiments")
    experiment_parser.add_argument("annotations", type=Path)
    experiment_parser.add_argument("output_dir", type=Path)
    experiment_parser.add_argument("--sample-sizes", default="10,20,50")
    experiment_parser.add_argument("--train-ratio", type=float, default=0.8)
    experiment_parser.add_argument("--train", action="store_true", help="Actually run training; default only prepares datasets")
    add_training_args(experiment_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Score a detector against annotated ground truth on a held-out split",
    )
    evaluate_parser.add_argument("annotations", type=Path)
    evaluate_parser.add_argument("output_dir", type=Path)
    evaluate_parser.add_argument("--run-name", default="eval")
    evaluate_parser.add_argument("--detector", choices=["heuristic", "doclayout-yolo"], default="doclayout-yolo")
    evaluate_parser.add_argument("--model-path", type=Path)
    evaluate_parser.add_argument("--device", default="cpu")
    evaluate_parser.add_argument("--image-size", type=int, default=1024)
    evaluate_parser.add_argument(
        "--label-mapping",
        default="identity",
        help="'identity' for canonical-label checkpoints, 'docstructbench' for the pretrained "
        "weights, or a path to a JSON mapping",
    )
    evaluate_parser.add_argument(
        "--predictions",
        type=Path,
        help="Score cached predictions JSON instead of running the model",
    )
    evaluate_parser.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    evaluate_parser.add_argument("--split-strategy", choices=["document", "page"], default="document")
    evaluate_parser.add_argument("--split-seed", type=int, default=0)
    evaluate_parser.add_argument("--test-ratio", type=float, default=0.35)
    evaluate_parser.add_argument("--val-ratio", type=float, default=0.15)
    evaluate_parser.add_argument(
        "--eval-confidence",
        type=float,
        default=EVAL_CONFIDENCE,
        help="Confidence used to request detections; keep this low so the PR curve keeps its tail",
    )
    evaluate_parser.add_argument(
        "--operating-confidence",
        type=float,
        default=OPERATING_CONFIDENCE,
        help="Deployed operating point, used for operational metrics and the confusion matrix",
    )
    evaluate_parser.add_argument("--per-page", action="store_true", help="Include per-page operational rows")
    evaluate_parser.add_argument(
        "--tag",
        action="append",
        metavar="KEY=VALUE",
        help="Experiment coordinates for the ledger, repeatable: --tag n=20 --tag init=docstructbench",
    )
    evaluate_parser.add_argument(
        "--ledger",
        type=Path,
        default=LEDGER_PATH,
        help="Append-only markdown ledger to record this run in",
    )
    evaluate_parser.add_argument(
        "--no-ledger", action="store_true", help="Score without recording a ledger row"
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Build a comparison table from evaluation reports in a directory",
    )
    compare_parser.add_argument("reports_dir", type=Path)
    compare_parser.add_argument("--output", type=Path, help="Markdown file to write (default: stdout)")
    compare_parser.add_argument("--title", default="Experiment comparison")

    args = parser.parse_args()

    if args.command == "parse":
        options = ParseRequestOptions(
            detector=args.detector,
            model_path=args.model_path,
            device=args.device,
            image_size=args.image_size,
            confidence=args.confidence,
            dpi=args.dpi,
            max_pages=args.max_pages,
            run_ocr=not args.no_ocr,
        )
        result = parse_document(args.pdf, args.output_dir, options)
        print(json.dumps({"result_json": str(args.output_dir / "result.json"), "pages": len(result.pages)}, ensure_ascii=False))
    elif args.command == "export-yolo":
        export_result = export_yolo_dataset(
            args.annotations,
            args.output_dir,
            max_pages=args.max_pages,
            train_ratio=args.train_ratio,
        )
        print(json.dumps(export_result.model_dump(mode="json"), ensure_ascii=False))
    elif args.command == "train":
        dataset = export_yolo_dataset(
            args.annotations,
            args.output_dir / "dataset",
            max_pages=args.max_pages,
            train_ratio=args.train_ratio,
        )
        config = training_config_from_args(args, data_yaml=Path(dataset.data_yaml), output_dir=args.output_dir / "training")
        training = run_doclayout_yolo_training(config)
        print(
            json.dumps(
                {
                    "dataset": dataset.model_dump(mode="json"),
                    "training": training.model_dump(mode="json"),
                },
                ensure_ascii=False,
            )
        )
    elif args.command == "experiment":
        sample_sizes = parse_sample_sizes(args.sample_sizes)
        config = training_config_from_args(args, data_yaml=Path("data.yaml"), output_dir=args.output_dir / "training")
        results = run_page_count_experiments(
            annotation_path=args.annotations,
            output_dir=args.output_dir,
            sample_sizes=sample_sizes,
            training_config=config,
            train_ratio=args.train_ratio,
            run_training=args.train,
        )
        print(json.dumps([result.model_dump(mode="json") for result in results], ensure_ascii=False))
    elif args.command == "evaluate":
        print(json.dumps(run_evaluate(args), ensure_ascii=False))
    elif args.command == "compare":
        reports = load_reports(args.reports_dir)
        if not reports:
            raise SystemExit(f"No evaluation reports found in {args.reports_dir}")
        table = render_comparison(reports, title=args.title)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(table, encoding="utf-8")
            print(json.dumps({"output": str(args.output), "runs": len(reports)}, ensure_ascii=False))
        else:
            print(table)


def run_evaluate(args: argparse.Namespace) -> dict:
    dataset = load_annotation_pages(args.annotations)
    splits = make_splits(
        dataset.pages,
        seed=args.split_seed,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        strategy=args.split_strategy,
    )
    selected = {
        "test": splits.test,
        "val": splits.val,
        "train": splits.train,
        "all": [*splits.train, *splits.val, *splits.test],
    }[args.split]

    ground_truth = load_ground_truth(pages=selected)

    if args.predictions:
        predictions = load_predictions(args.predictions)
        mapping_name = args.label_mapping
        prediction_warnings: list[str] = [f"Scored cached predictions from {args.predictions}"]
    else:
        predictions, mapping_name, prediction_warnings = generate_predictions(
            ground_truth.annotation_pages,
            detector_name=args.detector,
            model_path=args.model_path,
            device=args.device,
            image_size=args.image_size,
            label_mapping=args.label_mapping,
            confidence=args.eval_confidence,
        )
        save_predictions(predictions, args.output_dir / f"{args.run_name}_predictions.json")

    report = evaluate(
        ground_truth,
        predictions,
        run_name=args.run_name,
        detector=args.detector,
        model_path=args.model_path,
        label_mapping=mapping_name,
        eval_confidence=args.eval_confidence,
        operating_confidence=args.operating_confidence,
        seed=args.split_seed,
        split=splits.summary(),
        annotation_path=args.annotations,
        notes=[*splits.warnings, *prediction_warnings, f"Scored the {args.split!r} split"],
        include_per_page=args.per_page,
        scored_split=args.split,
        tags=parse_tags(args.tag),
    )
    paths = write_report(report, args.output_dir, basename=args.run_name)

    ledger_row = None
    if not args.no_ledger:
        ledger_row = append_run(
            report,
            ledger_path=args.ledger,
            report_link=_relative_link(Path(paths["markdown"]), Path(args.ledger)),
        )

    model_row = next((entry for entry in report.operational if entry.variant == "model"), None)
    return {
        "report": paths,
        "ledger": str(args.ledger) if ledger_row else None,
        "ledger_row": ledger_row,
        "pages": report.page_count,
        "map50": report.detection.map50 if report.detection else None,
        "map5095": report.detection.map5095 if report.detection else None,
        "reduction_rate": model_row.reduction_rate if model_row else None,
        "loss_rate": model_row.loss_rate if model_row else None,
        "contamination_rate": model_row.contamination_rate if model_row else None,
        "dataset_fingerprint": report.dataset_fingerprint,
    }


def _relative_link(target: Path, ledger_path: Path) -> str:
    """Link from the ledger to a report, relative when they share a tree."""
    try:
        return target.resolve().relative_to(ledger_path.resolve().parent).as_posix()
    except ValueError:
        return target.as_posix()


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-size", default="n", help="YOLOv10 size suffix, e.g. n/s/m/b/l/x")
    parser.add_argument("--pretrain", help="Path to .pt checkpoint, or 'coco'")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--run-name", default="doclayout_finetune")
    parser.add_argument("--dry-run", action="store_true")


def training_config_from_args(args: argparse.Namespace, data_yaml: Path, output_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        data_yaml=data_yaml,
        output_dir=output_dir,
        run_name=args.run_name,
        model_size=args.model_size,
        pretrain=args.pretrain,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=args.device,
        workers=args.workers,
        learning_rate=args.learning_rate,
        dry_run=args.dry_run,
    )


def parse_sample_sizes(value: str) -> list[int]:
    sizes = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        sizes.append(int(raw))
    if not sizes:
        raise ValueError("At least one sample size is required")
    return sizes


if __name__ == "__main__":
    main()
