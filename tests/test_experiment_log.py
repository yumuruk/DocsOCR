"""Tests for the append-only experiment ledger.

The ledger's whole value is that it is trustworthy as a record, so the properties
worth testing are: rows accumulate in order, nothing already written is altered,
and each row carries enough provenance to tell comparable runs from replicates.
"""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdf_layout_parser.evaluation import PredBox, evaluate, load_ground_truth
from pdf_layout_parser.experiment_log import (
    TABLE_MARKER,
    append_note,
    append_run,
    count_rows,
    ensure_ledger,
    load_reports,
    parse_tags,
)
from pdf_layout_parser.evaluation.report import write_report
from pdf_layout_parser.models import AnnotationBox, AnnotationPage, BoundingBox


def make_report(run_name: str, seed: int = 0, width: float = 50.0, tags=None):
    pages = [
        AnnotationPage(
            image_path="doc/pages/page_0001.png",
            width=100,
            height=100,
            document_id="doc",
            page_number=1,
            annotations=[
                AnnotationBox(label="body", bbox=BoundingBox(x=0, y=0, width=width, height=50))
            ],
        )
    ]
    ground_truth = load_ground_truth(pages=pages)
    predictions = [
        PredBox(ground_truth.pages[0].image_id, "body", BoundingBox(x=0, y=0, width=50, height=50), 0.9)
    ]
    return evaluate(
        ground_truth,
        predictions,
        run_name=run_name,
        detector="stub",
        seed=seed,
        scored_split="test",
        tags=tags or {},
    )


class LedgerTests(unittest.TestCase):
    def test_ledger_is_created_with_a_row_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = ensure_ledger(Path(tmp) / "docs" / "EXPERIMENTS.md")
            self.assertTrue(path.exists())
            self.assertIn(TABLE_MARKER, path.read_text(encoding="utf-8"))
            self.assertEqual(count_rows(path), 0)

    def test_rows_accumulate_oldest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "EXPERIMENTS.md"
            append_run(make_report("first"), ledger_path=path, timestamp="2026-01-01 00:00")
            append_run(make_report("second"), ledger_path=path, timestamp="2026-01-02 00:00")

            body = path.read_text(encoding="utf-8")
            _, _, rows_text = body.partition(TABLE_MARKER)
            rows = [line for line in rows_text.splitlines() if line.startswith("| ")]
            self.assertEqual(len(rows), 2)
            self.assertIn("`first`", rows[0])
            self.assertIn("`second`", rows[1])
            self.assertTrue(rows[0].startswith("| 1 |"))
            self.assertTrue(rows[1].startswith("| 2 |"))

    def test_appending_does_not_alter_existing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "EXPERIMENTS.md"
            first = append_run(make_report("first"), ledger_path=path, timestamp="2026-01-01 00:00")
            append_run(make_report("second"), ledger_path=path, timestamp="2026-01-02 00:00")
            self.assertIn(first, path.read_text(encoding="utf-8"))

    def test_row_carries_experiment_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "EXPERIMENTS.md"
            row = append_run(
                make_report("cell", tags={"n": "20", "init": "docstructbench", "labels": "manual"}),
                ledger_path=path,
                timestamp="2026-01-01 00:00",
            )
            # Sorted for a stable column, regardless of tag insertion order.
            self.assertIn("init=docstructbench labels=manual n=20", row)
            self.assertIn("document/test" if "document" in row else "test", row)

    def test_row_records_seed_and_dataset_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "EXPERIMENTS.md"
            row = append_run(make_report("run", seed=7), ledger_path=path, timestamp="2026-01-01 00:00")
            report = make_report("run", seed=7)
            self.assertIn(f"`{report.dataset_fingerprint}`", row)
            self.assertIn("| 7 |", row)

    def test_different_ground_truth_gets_a_different_fingerprint(self):
        """The column has to be able to say 'these two rows are not comparable'."""
        self.assertNotEqual(
            make_report("a", width=50.0).dataset_fingerprint,
            make_report("b", width=60.0).dataset_fingerprint,
        )

    def test_ledger_without_a_marker_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "EXPERIMENTS.md"
            path.write_text("# Hand-written notes\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                ensure_ledger(path)

    def test_report_link_is_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "EXPERIMENTS.md"
            row = append_run(
                make_report("linked"),
                ledger_path=path,
                report_link="runs/linked.md",
                timestamp="2026-01-01 00:00",
            )
            self.assertIn("[md](runs/linked.md)", row)


class ResearchLogTests(unittest.TestCase):
    def test_notes_append_with_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "RESEARCH_LOG.md"
            append_note("First finding", "Body one.", log_path=path, timestamp="2026-01-01")
            append_note("Second finding", "Body two.", log_path=path, timestamp="2026-01-02")
            text = path.read_text(encoding="utf-8")
            self.assertIn("## 2026-01-01 — First finding", text)
            self.assertIn("## 2026-01-02 — Second finding", text)
            self.assertLess(text.index("First finding"), text.index("Second finding"))


class ReportLoadingTests(unittest.TestCase):
    def test_load_reports_skips_prediction_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_report(make_report("alpha"), directory, basename="alpha")
            write_report(make_report("beta"), directory, basename="beta")
            (directory / "alpha_predictions.json").write_text("[]", encoding="utf-8")
            (directory / "unrelated.json").write_text('{"nope": 1}', encoding="utf-8")

            reports = load_reports(directory)
            self.assertEqual(sorted(report.run_name for report in reports), ["alpha", "beta"])


class TagParsingTests(unittest.TestCase):
    def test_parses_key_value_pairs(self):
        self.assertEqual(parse_tags(["n=20", "init=coco"]), {"n": "20", "init": "coco"})

    def test_empty_input(self):
        self.assertEqual(parse_tags(None), {})

    def test_rejects_malformed_tag(self):
        with self.assertRaises(ValueError):
            parse_tags(["justakey"])

    def test_rejects_empty_key(self):
        with self.assertRaises(ValueError):
            parse_tags(["=20"])


if __name__ == "__main__":
    unittest.main()
