from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from pdf_layout_parser.dataset import export_yolo_dataset, to_yolo_bbox
from pdf_layout_parser.detectors import HeuristicLayoutDetector
from pdf_layout_parser.models import BoundingBox, DocumentParseResult, ExtractionAction, PageParseResult, ParsedRegion, TrainingConfig
from pdf_layout_parser.pipeline import to_markdown
from pdf_layout_parser.training import run_doclayout_yolo_training


class CorePrototypeTests(unittest.TestCase):
    def test_yolo_bbox_conversion(self):
        bbox = BoundingBox(x=10, y=20, width=30, height=40)
        self.assertEqual(to_yolo_bbox(bbox, image_width=100, image_height=200), (0.25, 0.2, 0.3, 0.2))

    def test_heuristic_detector_returns_expected_regions(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            Image.new("RGB", (1000, 1400), "white").save(image_path)
            regions = HeuristicLayoutDetector().detect(image_path, page_number=1)
        self.assertEqual([region.label for region in regions], ["header", "title", "body", "footer"])

    def test_markdown_contains_region_text(self):
        result = DocumentParseResult(input_path="sample.pdf", output_dir="runs/sample", detector="heuristic")
        result.pages.append(
            PageParseResult(
                page_number=1,
                image_path="page.png",
                width=100,
                height=100,
                regions=[
                    ParsedRegion(
                        page_number=1,
                        label="body",
                        bbox=BoundingBox(x=0, y=0, width=100, height=50),
                        confidence=0.9,
                        action=ExtractionAction.OCR,
                        source="test",
                        text="hello document",
                    )
                ],
            )
        )
        markdown = to_markdown(result)
        self.assertIn("hello document", markdown)
        self.assertIn("Page 1", markdown)

    def test_export_yolo_dataset_writes_train_val_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_paths = []
            for index in range(3):
                image_path = root / f"page_{index + 1}.png"
                Image.new("RGB", (100, 200), "white").save(image_path)
                image_paths.append(image_path)
            annotation_path = root / "annotations.json"
            annotation_path.write_text(
                """
{
  "labels": ["title", "body"],
  "pages": [
    {"image_path": "%s", "width": 100, "height": 200, "annotations": [{"label": "title", "bbox": {"x": 0, "y": 0, "width": 50, "height": 50}}]},
    {"image_path": "%s", "width": 100, "height": 200, "annotations": [{"label": "body", "bbox": {"x": 0, "y": 50, "width": 100, "height": 100}}]},
    {"image_path": "%s", "width": 100, "height": 200, "annotations": [{"label": "body", "bbox": {"x": 0, "y": 100, "width": 100, "height": 100}}]}
  ]
}
                """
                % tuple(str(path).replace("\\", "\\\\") for path in image_paths),
                encoding="utf-8",
            )
            export = export_yolo_dataset(annotation_path, root / "dataset", train_ratio=0.67)

            self.assertEqual(export.train_count, 2)
            self.assertEqual(export.val_count, 1)
            self.assertTrue(Path(export.data_yaml).exists())
            self.assertIn("names:", Path(export.data_yaml).read_text(encoding="utf-8"))

    def test_training_dry_run_returns_preview(self):
        result = run_doclayout_yolo_training(
            TrainingConfig(
                data_yaml=Path("dataset/data.yaml"),
                output_dir=Path("runs/training"),
                pretrain=Path("models/base.pt"),
                dry_run=True,
            )
        )

        self.assertEqual(result.status, "dry_run")
        self.assertIn("YOLOv10", result.command_preview)
        self.assertEqual(result.model_source, "models/base.pt")


if __name__ == "__main__":
    unittest.main()
