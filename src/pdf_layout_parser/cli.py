from __future__ import annotations

import argparse
import json
from pathlib import Path

from pdf_layout_parser.dataset import export_yolo_dataset
from pdf_layout_parser.models import ParseRequestOptions, TrainingConfig
from pdf_layout_parser.pipeline import parse_document
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
