"""Tests for the evaluation harness.

The point of these is numeric trust. Every AP assertion is either a hand-computed
value derived from the 101-point rule or a comparison against pycocotools, because
a metric implementation that is merely plausible is worse than none -- it produces
confident numbers nobody can check.
"""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from pdf_layout_parser.dataset import build_data_yaml, export_yolo_dataset, to_yolo_bbox
from pdf_layout_parser.evaluation import (
    EVAL_CONFIDENCE,
    OPERATING_CONFIDENCE,
    GtBox,
    PageGeometry,
    PredBox,
    average_precision,
    confusion_matrix,
    difference_area,
    evaluate,
    evaluate_detections,
    evaluate_operational,
    full_page_baseline,
    iou,
    load_ground_truth,
    oracle_baseline,
    render_markdown,
    union_area,
)
from pdf_layout_parser.evaluation.detection import average_precision_from_flags
from pdf_layout_parser.labels import (
    DOCSTRUCTBENCH_TO_CANONICAL,
    UnknownLabelError,
    canonical_id,
    keep_labels,
    map_label,
    resolve_mapping,
    skip_labels,
)
from pdf_layout_parser.models import (
    AnnotationBox,
    AnnotationPage,
    BoundingBox,
    LabelStatus,
)
from pdf_layout_parser.splits import make_splits


def box(x, y, width, height) -> BoundingBox:
    return BoundingBox(x=x, y=y, width=width, height=height)


class GeometryTests(unittest.TestCase):
    def test_iou_identical_boxes(self):
        self.assertAlmostEqual(iou(box(0, 0, 10, 10), box(0, 0, 10, 10)), 1.0)

    def test_iou_disjoint_boxes(self):
        self.assertEqual(iou(box(0, 0, 10, 10), box(20, 20, 10, 10)), 0.0)

    def test_iou_known_value(self):
        # Overlap 5x10=50; union 100+100-50=150.
        self.assertAlmostEqual(iou(box(0, 0, 10, 10), box(5, 0, 10, 10)), 50 / 150)

    def test_iou_is_not_the_intersects_heuristic(self):
        """Regression guard for the metric that FINDINGS flagged as inflating mAP.

        `BoundingBox.intersects` returns True for a small box mostly inside a big
        one; real IoU is tiny. If evaluation ever reuses `intersects`, this fails.
        """
        small = box(10, 10, 10, 10)
        large = box(0, 0, 100, 100)
        self.assertTrue(small.intersects(large))
        self.assertLess(iou(small, large), 0.02)

    def test_union_area_counts_overlap_once(self):
        # Two 10x10 boxes overlapping in a 5x10 strip: 100 + 100 - 50 = 150.
        self.assertAlmostEqual(union_area([box(0, 0, 10, 10), box(5, 0, 10, 10)]), 150.0)

    def test_union_area_of_nested_boxes(self):
        self.assertAlmostEqual(union_area([box(0, 0, 100, 100), box(10, 10, 10, 10)]), 10000.0)

    def test_difference_area(self):
        # 10x10 minus its left 5x10 half.
        self.assertAlmostEqual(difference_area([box(0, 0, 10, 10)], [box(0, 0, 5, 10)]), 50.0)

    def test_difference_area_with_no_overlap(self):
        self.assertAlmostEqual(difference_area([box(0, 0, 10, 10)], [box(50, 50, 10, 10)]), 100.0)


class AveragePrecisionTests(unittest.TestCase):
    def test_single_perfect_detection(self):
        gts = [GtBox("p1", "body", box(0, 0, 10, 10))]
        preds = [PredBox("p1", "body", box(0, 0, 10, 10), 0.9)]
        self.assertAlmostEqual(average_precision(gts, preds, "body", 0.5), 1.0)

    def test_hand_computed_half_recall(self):
        """Two GT, one TP and one FP.

        recalls  = [0.5, 0.5]
        precision= [1.0, 0.5] -> envelope [1.0, 0.5]
        The 101 recall points 0.00..1.00 include 51 points <= 0.50, each scoring
        precision 1.0; the remaining 50 points have no recall reaching them and
        score 0. AP = 51 / 101.
        """
        gts = [
            GtBox("p1", "body", box(0, 0, 10, 10)),
            GtBox("p1", "body", box(50, 50, 10, 10)),
        ]
        preds = [
            PredBox("p1", "body", box(0, 0, 10, 10), 0.9),
            PredBox("p1", "body", box(200, 200, 10, 10), 0.8),
        ]
        self.assertAlmostEqual(average_precision(gts, preds, "body", 0.5), 51 / 101)

    def test_flags_helper_matches_hand_calculation(self):
        self.assertAlmostEqual(average_precision_from_flags([1, 0], [0, 1], gt_count=2), 51 / 101)

    def test_class_without_ground_truth_is_none_not_zero(self):
        gts = [GtBox("p1", "body", box(0, 0, 10, 10))]
        preds = [PredBox("p1", "table", box(0, 0, 10, 10), 0.9)]
        self.assertIsNone(average_precision(gts, preds, "table", 0.5))

    def test_duplicate_detection_is_a_false_positive(self):
        """One GT, two overlapping detections: the second must not also match."""
        gts = [GtBox("p1", "body", box(0, 0, 10, 10))]
        preds = [
            PredBox("p1", "body", box(0, 0, 10, 10), 0.9),
            PredBox("p1", "body", box(0, 0, 10, 10), 0.8),
        ]
        # recalls [1.0, 1.0]; precisions [1.0, 0.5] -> envelope [1.0, 1.0]; AP 1.0.
        self.assertAlmostEqual(average_precision(gts, preds, "body", 0.5), 1.0)

    def test_iou_threshold_is_respected(self):
        gts = [GtBox("p1", "body", box(0, 0, 10, 10))]
        preds = [PredBox("p1", "body", box(6, 0, 10, 10), 0.9)]  # IoU = 40/160 = 0.25
        self.assertAlmostEqual(average_precision(gts, preds, "body", 0.2), 1.0)
        self.assertAlmostEqual(average_precision(gts, preds, "body", 0.5), 0.0)

    def test_map_excludes_classes_without_ground_truth(self):
        gts = [GtBox("p1", "body", box(0, 0, 10, 10))]
        preds = [PredBox("p1", "body", box(0, 0, 10, 10), 0.9)]
        metrics = evaluate_detections(gts, preds)
        by_label = {entry.label: entry for entry in metrics.per_class}
        self.assertAlmostEqual(by_label["body"].ap50, 1.0)
        self.assertIsNone(by_label["table"].ap50)
        # mAP is 1.0, not 1/8, because the seven empty classes are excluded.
        self.assertAlmostEqual(metrics.map50, 1.0)

    def test_low_confidence_detections_still_count_towards_ap(self):
        """The C4 regression: a correct detection below the operating point.

        Predicting at 0.25 would discard this box entirely and report AP 0. The
        harness asks the model at 0.001, so recall is preserved.
        """
        gts = [GtBox("p1", "body", box(0, 0, 10, 10))]
        preds = [PredBox("p1", "body", box(0, 0, 10, 10), 0.01)]
        self.assertAlmostEqual(average_precision(gts, preds, "body", 0.5), 1.0)
        self.assertLess(preds[0].score, OPERATING_CONFIDENCE)
        self.assertGreaterEqual(preds[0].score, EVAL_CONFIDENCE)


class PycocotoolsParityTests(unittest.TestCase):
    """Cross-check against the reference implementation when it is installed."""

    def _pycocotools(self):
        try:
            import numpy  # noqa: F401
            from pycocotools.coco import COCO  # noqa: F401
            from pycocotools.cocoeval import COCOeval  # noqa: F401
        except ImportError:
            self.skipTest("pycocotools/numpy not installed; parity check skipped")
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        return COCO, COCOeval

    def _reference_ap(self, gts, preds, label, thresholds):
        """Per-threshold AP for one class, straight out of pycocotools."""
        COCO, COCOeval = self._pycocotools()

        image_ids = {name: index + 1 for index, name in enumerate(sorted({gt.image_id for gt in gts} | {p.image_id for p in preds}))}
        labels = sorted({gt.label for gt in gts} | {p.label for p in preds})
        category_ids = {name: index + 1 for index, name in enumerate(labels)}

        coco_gt = COCO()
        coco_gt.dataset = {
            "images": [{"id": value} for value in image_ids.values()],
            "categories": [{"id": value, "name": name} for name, value in category_ids.items()],
            "annotations": [
                {
                    "id": index + 1,
                    "image_id": image_ids[gt.image_id],
                    "category_id": category_ids[gt.label],
                    "bbox": [gt.bbox.x, gt.bbox.y, gt.bbox.width, gt.bbox.height],
                    "area": gt.bbox.area(),
                    "iscrowd": 0,
                }
                for index, gt in enumerate(gts)
            ],
        }
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(
            [
                {
                    "image_id": image_ids[pred.image_id],
                    "category_id": category_ids[pred.label],
                    "bbox": [pred.bbox.x, pred.bbox.y, pred.bbox.width, pred.bbox.height],
                    "score": pred.score,
                }
                for pred in preds
            ]
        )
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.iouThrs = list(thresholds)
        evaluator.params.areaRng = [[0, 1e10]]
        evaluator.params.areaRngLbl = ["all"]
        evaluator.params.maxDets = [100]
        evaluator.evaluate()
        evaluator.accumulate()

        category_index = labels.index(label)
        return [
            float(evaluator.eval["precision"][threshold_index, :, category_index, 0, 0].mean())
            for threshold_index in range(len(thresholds))
        ]

    def test_ap50_matches_pycocotools(self):
        gts = [
            GtBox("1", "body", box(10, 10, 50, 50)),
            GtBox("1", "body", box(100, 100, 40, 30)),
            GtBox("2", "body", box(0, 0, 20, 20)),
        ]
        preds = [
            PredBox("1", "body", box(12, 12, 50, 50), 0.9),
            PredBox("1", "body", box(300, 300, 40, 30), 0.7),
            PredBox("2", "body", box(0, 0, 22, 21), 0.6),
        ]
        reference = self._reference_ap(gts, preds, "body", [0.5])[0]
        self.assertAlmostEqual(average_precision(gts, preds, "body", 0.5), reference, places=6)

    def test_threshold_sweep_matches_pycocotools(self):
        """Harder case: several classes, duplicate detections, near-miss boxes.

        Duplicate and marginal detections are where matching order and the
        precision envelope actually bite, and the whole 0.50:0.95 sweep is
        compared rather than a single threshold.
        """
        gts = [
            GtBox("1", "body", box(10, 10, 100, 40)),
            GtBox("1", "body", box(10, 60, 100, 40)),
            GtBox("1", "table", box(10, 120, 100, 80)),
            GtBox("2", "body", box(0, 0, 50, 50)),
            GtBox("2", "table", box(60, 0, 50, 50)),
            GtBox("3", "body", box(5, 5, 200, 30)),
        ]
        preds = [
            # Tight, loose, and duplicate detections on page 1.
            PredBox("1", "body", box(10, 10, 100, 40), 0.95),
            PredBox("1", "body", box(12, 62, 100, 40), 0.80),
            PredBox("1", "body", box(11, 11, 98, 39), 0.60),  # duplicate of the first
            PredBox("1", "table", box(20, 130, 90, 70), 0.55),
            # Page 2: one good, one class-swapped, one spurious.
            PredBox("2", "body", box(0, 0, 55, 52), 0.90),
            PredBox("2", "body", box(60, 0, 50, 50), 0.50),  # actually a table
            PredBox("2", "table", box(300, 300, 20, 20), 0.40),
            # Page 3: a marginal box that passes some thresholds and not others.
            PredBox("3", "body", box(5, 5, 160, 30), 0.70),
        ]
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        for label in ("body", "table"):
            reference = self._reference_ap(gts, preds, label, thresholds)
            for threshold, expected in zip(thresholds, reference):
                with self.subTest(label=label, iou=threshold):
                    self.assertAlmostEqual(
                        average_precision(gts, preds, label, threshold), expected, places=6
                    )

    def test_map5095_matches_pycocotools(self):
        gts = [
            GtBox("1", "body", box(10, 10, 100, 40)),
            GtBox("1", "table", box(10, 120, 100, 80)),
            GtBox("2", "body", box(0, 0, 50, 50)),
        ]
        preds = [
            PredBox("1", "body", box(11, 12, 100, 40), 0.95),
            PredBox("1", "table", box(20, 130, 90, 70), 0.55),
            PredBox("2", "body", box(0, 0, 55, 52), 0.90),
        ]
        thresholds = [round(0.5 + 0.05 * step, 2) for step in range(10)]
        metrics = evaluate_detections(gts, preds, iou_thresholds=tuple(thresholds))
        by_label = {entry.label: entry for entry in metrics.per_class}
        for label in ("body", "table"):
            reference = self._reference_ap(gts, preds, label, thresholds)
            with self.subTest(label=label):
                self.assertAlmostEqual(
                    by_label[label].ap5095, sum(reference) / len(reference), places=6
                )


class OperationalMetricsTests(unittest.TestCase):
    """A 100x100 page with a single 50x50 body region as the only keep content."""

    def setUp(self):
        self.pages = [PageGeometry("p1", 100, 100)]
        self.gts = [
            GtBox("p1", "body", box(0, 0, 50, 50)),
            # A skip-label region: must not count towards keep area.
            GtBox("p1", "footer", box(0, 90, 100, 10)),
        ]

    def test_keep_and_skip_partition(self):
        self.assertEqual(keep_labels(), frozenset({"title", "body", "table", "target_data"}))
        self.assertEqual(skip_labels(), frozenset({"figure", "header", "footer", "ignore"}))

    def test_full_page_baseline(self):
        metrics = full_page_baseline(self.pages, self.gts)
        self.assertAlmostEqual(metrics.reduction_rate, 0.0)
        self.assertAlmostEqual(metrics.loss_rate, 0.0)
        # 10000 extracted, only 2500 is real content.
        self.assertAlmostEqual(metrics.contamination_rate, 0.75)

    def test_oracle_baseline_is_the_reduction_ceiling(self):
        metrics = oracle_baseline(self.pages, self.gts)
        self.assertAlmostEqual(metrics.reduction_rate, 0.75)
        self.assertAlmostEqual(metrics.loss_rate, 0.0)
        self.assertAlmostEqual(metrics.contamination_rate, 0.0)

    def test_model_losing_half_the_content(self):
        preds = [PredBox("p1", "body", box(0, 0, 25, 50), 0.9)]
        metrics, per_page = evaluate_operational(self.pages, self.gts, preds)
        self.assertAlmostEqual(metrics.reduction_rate, 1.0 - 1250 / 10000)
        self.assertAlmostEqual(metrics.loss_rate, 0.5)
        self.assertAlmostEqual(metrics.contamination_rate, 0.0)
        self.assertEqual(len(per_page), 1)

    def test_predicting_a_skip_label_does_not_add_extraction_area(self):
        preds = [
            PredBox("p1", "body", box(0, 0, 50, 50), 0.9),
            PredBox("p1", "figure", box(50, 50, 50, 50), 0.9),
        ]
        metrics, _ = evaluate_operational(self.pages, self.gts, preds)
        self.assertAlmostEqual(metrics.reduction_rate, 0.75)
        self.assertAlmostEqual(metrics.loss_rate, 0.0)

    def test_operational_metrics_filter_at_the_operating_point(self):
        """Detections below the operating point are not extracted in production."""
        preds = [PredBox("p1", "body", box(0, 0, 50, 50), 0.01)]
        metrics, _ = evaluate_operational(self.pages, self.gts, preds)
        self.assertAlmostEqual(metrics.reduction_rate, 1.0)
        self.assertAlmostEqual(metrics.loss_rate, 1.0)

    def test_contamination_from_an_over_wide_prediction(self):
        preds = [PredBox("p1", "body", box(0, 0, 100, 50), 0.9)]
        metrics, _ = evaluate_operational(self.pages, self.gts, preds)
        # 5000 extracted, 2500 of it real.
        self.assertAlmostEqual(metrics.contamination_rate, 0.5)
        self.assertAlmostEqual(metrics.loss_rate, 0.0)

    def test_oracle_reduction_equals_full_page_contamination(self):
        """An identity worth asserting: both are 1 - gt_keep / page_area.

        Full-page OCR extracts everything, so its contamination is the share of
        the page that is not content; the oracle extracts exactly the content, so
        its saving is the same quantity. If these ever diverge, one of the two
        area computations is wrong.
        """
        baseline = full_page_baseline(self.pages, self.gts)
        oracle = oracle_baseline(self.pages, self.gts)
        self.assertAlmostEqual(oracle.reduction_rate, baseline.contamination_rate)

    def test_reduction_is_bounded_by_the_oracle_when_nothing_is_lost(self):
        preds = [PredBox("p1", "body", box(0, 0, 50, 50), 0.9)]
        model, _ = evaluate_operational(self.pages, self.gts, preds)
        oracle = oracle_baseline(self.pages, self.gts)
        self.assertAlmostEqual(model.loss_rate, 0.0)
        self.assertLessEqual(model.reduction_rate, oracle.reduction_rate + 1e-9)

    def test_overlapping_predictions_are_not_double_charged(self):
        """Two overlapping keep boxes cost the area of their union, not the sum."""
        preds = [
            PredBox("p1", "body", box(0, 0, 50, 50), 0.9),
            PredBox("p1", "title", box(25, 0, 50, 50), 0.9),
        ]
        metrics, _ = evaluate_operational(self.pages, self.gts, preds)
        # Union is 75x50 = 3750, not 2500 + 2500.
        self.assertAlmostEqual(metrics.pred_keep_area, 3750.0)


class ConfusionMatrixTests(unittest.TestCase):
    def test_table_detected_as_figure_is_a_confusion_not_a_miss(self):
        gts = [GtBox("p1", "table", box(0, 0, 10, 10))]
        preds = [PredBox("p1", "figure", box(0, 0, 10, 10), 0.9)]
        cells = {(cell.gt_label, cell.pred_label): cell.count for cell in confusion_matrix(gts, preds)}
        self.assertEqual(cells.get(("table", "figure")), 1)
        self.assertNotIn(("table", "__missed__"), cells)

    def test_unmatched_ground_truth_is_missed(self):
        gts = [GtBox("p1", "table", box(0, 0, 10, 10))]
        cells = {(cell.gt_label, cell.pred_label): cell.count for cell in confusion_matrix(gts, [])}
        self.assertEqual(cells.get(("table", "__missed__")), 1)

    def test_unmatched_prediction_is_background(self):
        preds = [PredBox("p1", "table", box(0, 0, 10, 10), 0.9)]
        cells = {(cell.gt_label, cell.pred_label): cell.count for cell in confusion_matrix([], preds)}
        self.assertEqual(cells.get(("__background__", "table")), 1)

    def test_below_operating_confidence_predictions_are_excluded(self):
        gts = [GtBox("p1", "table", box(0, 0, 10, 10))]
        preds = [PredBox("p1", "table", box(0, 0, 10, 10), 0.01)]
        cells = {(cell.gt_label, cell.pred_label): cell.count for cell in confusion_matrix(gts, preds)}
        self.assertEqual(cells.get(("table", "__missed__")), 1)


class LabelCatalogueTests(unittest.TestCase):
    def test_class_ids_are_fixed(self):
        self.assertEqual(canonical_id("title"), 0)
        self.assertEqual(canonical_id("target_data"), 7)

    def test_unknown_label_raises(self):
        with self.assertRaises(UnknownLabelError):
            canonical_id("plain text")

    def test_docstructbench_mapping(self):
        mapping = DOCSTRUCTBENCH_TO_CANONICAL
        self.assertEqual(map_label("plain text", mapping), "body")
        self.assertEqual(map_label("abandon", mapping), "ignore")
        self.assertEqual(map_label("table_caption", mapping), "body")
        self.assertEqual(map_label("isolate_formula", mapping), "figure")
        # target_data has no zero-shot source.
        self.assertNotIn("target_data", mapping.values())

    def test_mapping_is_case_insensitive_on_the_raw_side(self):
        self.assertEqual(map_label("Plain Text", DOCSTRUCTBENCH_TO_CANONICAL), "body")

    def test_resolve_mapping_reports_its_name(self):
        name, mapping = resolve_mapping("docstructbench")
        self.assertEqual(name, "docstructbench")
        self.assertEqual(mapping["plain text"], "body")
        self.assertEqual(resolve_mapping(None)[0], "identity")

    def test_resolve_mapping_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            resolve_mapping("not-a-mapping")


class DatasetExportTests(unittest.TestCase):
    def _annotation_pages(self, root: Path, document_count: int, pages_per_document: int):
        pages = []
        for document_index in range(document_count):
            for page_index in range(pages_per_document):
                image_path = root / f"doc{document_index}_page{page_index}.png"
                Image.new("RGB", (100, 200), "white").save(image_path)
                pages.append(
                    AnnotationPage(
                        image_path=str(image_path),
                        width=100,
                        height=200,
                        document_id=f"doc{document_index}",
                        page_number=page_index + 1,
                        annotations=[
                            AnnotationBox(label="body", bbox=box(0, 0, 50, 50)),
                        ],
                    )
                )
        return pages

    def test_out_of_bounds_box_is_cropped_not_shifted(self):
        """The dataset.py:115 defect.

        A box from x=-10 to x=20 must become 0..20: centre 10, width 20. The old
        component-wise clamp kept width 30 and reported centre 5, pushing the
        right edge from 20 out to 35.
        """
        result = to_yolo_bbox(box(-10, 0, 30, 10), image_width=100, image_height=200)
        self.assertAlmostEqual(result[0], 0.10)
        self.assertAlmostEqual(result[2], 0.20)

    def test_in_bounds_conversion_is_unchanged(self):
        self.assertEqual(to_yolo_bbox(box(10, 20, 30, 40), 100, 200), (0.25, 0.2, 0.3, 0.2))

    def test_data_yaml_always_declares_the_full_catalogue(self):
        """C3: nc and names must not depend on which classes a dataset contains."""
        yaml_text = build_data_yaml()
        self.assertIn("nc: 8", yaml_text)
        self.assertIn("0: title", yaml_text)
        self.assertIn("7: target_data", yaml_text)
        self.assertNotIn("test:", yaml_text)
        self.assertIn("test: test.txt", build_data_yaml(has_test=True))

    def test_export_with_explicit_splits_writes_a_test_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pages = self._annotation_pages(root, document_count=4, pages_per_document=3)
            annotation_path = root / "annotations.json"
            annotation_path.write_text(
                '{"labels": ["body"], "pages": %s}'
                % "[%s]" % ",".join(page.model_dump_json() for page in pages),
                encoding="utf-8",
            )
            splits = make_splits(pages, seed=0)
            export = export_yolo_dataset(annotation_path, root / "dataset", splits=splits, strict=True)

            self.assertGreater(export.test_count, 0)
            self.assertEqual(export.train_count + export.val_count + export.test_count, 12)
            self.assertEqual(export.class_ids["body"], 1)
            self.assertTrue(Path(export.test_txt).exists())
            self.assertIn("test: test.txt", Path(export.data_yaml).read_text(encoding="utf-8"))
            # Labels are written with canonical ids, not the annotation file's order.
            label_file = next(Path(export.labels_dir).glob("*.txt"))
            self.assertTrue(label_file.read_text(encoding="utf-8").startswith("1 "))

    def test_strict_export_refuses_the_legacy_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pages = self._annotation_pages(root, document_count=4, pages_per_document=3)
            annotation_path = root / "annotations.json"
            annotation_path.write_text(
                '{"labels": ["body"], "pages": [%s]}'
                % ",".join(page.model_dump_json() for page in pages),
                encoding="utf-8",
            )
            with self.assertRaises(Exception):
                export_yolo_dataset(annotation_path, root / "dataset", strict=True)

    def test_legacy_export_warns_that_it_is_not_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pages = self._annotation_pages(root, document_count=2, pages_per_document=2)
            annotation_path = root / "annotations.json"
            annotation_path.write_text(
                '{"labels": ["body"], "pages": [%s]}'
                % ",".join(page.model_dump_json() for page in pages),
                encoding="utf-8",
            )
            export = export_yolo_dataset(annotation_path, root / "dataset")
            self.assertTrue(any("legacy prefix split" in warning for warning in export.warnings))

    def test_incomplete_pages_are_not_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pages = self._annotation_pages(root, document_count=2, pages_per_document=2)
            pages[0].label_status = LabelStatus.IN_PROGRESS
            annotation_path = root / "annotations.json"
            annotation_path.write_text(
                '{"labels": ["body"], "pages": [%s]}'
                % ",".join(page.model_dump_json() for page in pages),
                encoding="utf-8",
            )
            export = export_yolo_dataset(annotation_path, root / "dataset")
            self.assertEqual(export.page_count, 3)
            self.assertEqual(export.skipped_incomplete_pages, 1)


class GroundTruthLoadingTests(unittest.TestCase):
    def test_non_canonical_ground_truth_label_raises_by_default(self):
        page = AnnotationPage(
            image_path="doc/pages/page_0001.png",
            width=100,
            height=100,
            document_id="doc",
            annotations=[AnnotationBox(label="plain text", bbox=box(0, 0, 10, 10))],
        )
        with self.assertRaises(Exception):
            load_ground_truth(pages=[page])

    def test_incomplete_pages_are_excluded_from_ground_truth(self):
        complete = AnnotationPage(
            image_path="doc/pages/page_0001.png",
            width=100,
            height=100,
            document_id="doc",
            annotations=[AnnotationBox(label="body", bbox=box(0, 0, 10, 10))],
        )
        partial = AnnotationPage(
            image_path="doc/pages/page_0002.png",
            width=100,
            height=100,
            document_id="doc",
            label_status=LabelStatus.IN_PROGRESS,
            annotations=[AnnotationBox(label="body", bbox=box(0, 0, 10, 10))],
        )
        ground_truth = load_ground_truth(pages=[complete, partial])
        self.assertEqual(len(ground_truth.pages), 1)
        self.assertEqual(ground_truth.skipped_incomplete, 1)

    def test_fingerprint_is_stable_and_content_sensitive(self):
        def make(width):
            return AnnotationPage(
                image_path="doc/pages/page_0001.png",
                width=100,
                height=100,
                document_id="doc",
                annotations=[AnnotationBox(label="body", bbox=box(0, 0, width, 10))],
            )

        first = load_ground_truth(pages=[make(10)]).fingerprint
        again = load_ground_truth(pages=[make(10)]).fingerprint
        different = load_ground_truth(pages=[make(11)]).fingerprint
        self.assertEqual(first, again)
        self.assertNotEqual(first, different)

    def test_duplicate_page_ids_are_rejected(self):
        page = AnnotationPage(
            image_path="doc/pages/page_0001.png",
            width=100,
            height=100,
            document_id="doc",
            annotations=[],
        )
        with self.assertRaises(Exception):
            load_ground_truth(pages=[page, page.model_copy()])


class ReportTests(unittest.TestCase):
    def test_report_renders_the_trade_off_table_and_flags_absent_classes(self):
        pages = [
            AnnotationPage(
                image_path="doc/pages/page_0001.png",
                width=100,
                height=100,
                document_id="doc",
                page_number=1,
                annotations=[AnnotationBox(label="body", bbox=box(0, 0, 50, 50))],
            )
        ]
        ground_truth = load_ground_truth(pages=pages)
        predictions = [PredBox(ground_truth.pages[0].image_id, "body", box(0, 0, 50, 50), 0.9)]
        report = evaluate(ground_truth, predictions, run_name="unit-test", detector="stub")

        variants = [entry.variant for entry in report.operational]
        self.assertEqual(variants, ["full_page_ocr", "model", "oracle"])

        markdown = render_markdown(report)
        self.assertIn("Operational — selective OCR vs full-page OCR", markdown)
        self.assertIn("`full_page_ocr`", markdown)
        self.assertIn("Reduction ceiling (oracle)", markdown)
        self.assertIn("Detection — per-class AP", markdown)
        # A perfect prediction on the only labelled class.
        self.assertAlmostEqual(report.detection.map50, 1.0)
        # And the report says out loud why the other classes are null.
        self.assertTrue(any("no ground truth" in note for note in report.notes))

    def test_report_records_reproducibility_fields(self):
        pages = [
            AnnotationPage(
                image_path="doc/pages/page_0001.png",
                width=100,
                height=100,
                document_id="doc",
                annotations=[AnnotationBox(label="body", bbox=box(0, 0, 50, 50))],
            )
        ]
        ground_truth = load_ground_truth(pages=pages)
        report = evaluate(ground_truth, [], run_name="unit-test", seed=7)
        self.assertEqual(report.seed, 7)
        self.assertIsNotNone(report.catalogue_fingerprint)
        self.assertIsNotNone(report.dataset_fingerprint)
        self.assertEqual(report.eval_confidence, EVAL_CONFIDENCE)
        self.assertEqual(report.operating_confidence, OPERATING_CONFIDENCE)


if __name__ == "__main__":
    unittest.main()
