"""Tests for the fine-tune-and-compare orchestration and the web job registry.

Real training is exercised in the browser E2E, not here; these tests pin the
orchestration contract with the training and prediction seams patched out:
held-out scoring, phase reporting, guard rails, and the job lifecycle.
"""

import json
import time
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from pdf_layout_parser.evaluation import PredBox, image_id_for
from pdf_layout_parser.models import (
    AnnotationBox,
    AnnotationPage,
    BoundingBox,
    TrainingResult,
)


def build_corpus(root: Path, page_count: int = 6) -> Path:
    pages = []
    for index in range(page_count):
        image_path = root / f"page_{index}.png"
        Image.new("RGB", (100, 200), "white").save(image_path)
        pages.append(
            AnnotationPage(
                image_path=str(image_path),
                width=100,
                height=200,
                document_id="doc",
                page_number=index + 1,
                annotations=[
                    AnnotationBox(label="body", bbox=BoundingBox(x=10, y=10, width=60, height=80))
                ],
            )
        )
    annotation_path = root / "annotations.json"
    annotation_path.write_text(
        '{"labels": ["body"], "pages": [%s]}'
        % ",".join(page.model_dump_json() for page in pages),
        encoding="utf-8",
    )
    return annotation_path


def fake_training(best_path: Path):
    def run(config):
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_bytes(b"fake-weights")
        return TrainingResult(
            status="completed",
            data_yaml=str(config.data_yaml),
            output_dir=str(config.output_dir),
            run_name=config.run_name,
            model_source="fake",
            best_model_path=str(best_path),
            command_preview="fake",
            seed=config.seed,
        )

    return run


def fake_predictions(pages, **kwargs):
    """Perfect predictions for the tuned pass, sloppy ones for the baseline.

    Distinguished by the label mapping the caller asks for, mirroring how the
    real code differs between the two passes.
    """
    mapping = kwargs.get("label_mapping", "identity")
    perfect = mapping == "identity"
    preds = [
        PredBox(
            image_id_for(page),
            "body",
            # Baseline boxes cover a third of the width: IoU 1/3, below the 0.5
            # matching threshold, and two thirds of the content area is lost.
            BoundingBox(x=10, y=10, width=60 if perfect else 20, height=80),
            0.9,
        )
        for page in pages
    ]
    return preds, str(mapping), []


class FinetuneComparisonTests(unittest.TestCase):
    def _run(self, tmp: Path):
        from pdf_layout_parser import comparison

        annotation_path = build_corpus(tmp)
        best = tmp / "out" / "training" / "web_finetune_seed0" / "weights" / "best.pt"
        phases: list[str] = []
        with patch.object(comparison, "run_training", fake_training(best)), patch.object(
            comparison, "generate_predictions", fake_predictions
        ):
            return (
                comparison.run_finetune_comparison(
                    annotation_path=annotation_path,
                    output_dir=tmp / "out",
                    pretrain=tmp / "pretrain.pt",
                    seed=0,
                    progress=phases.append,
                ),
                phases,
            )

    def test_scores_both_models_on_the_same_held_out_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, phases = self._run(Path(tmp))
        self.assertEqual(result.baseline.page_count, result.tuned.page_count)
        self.assertEqual(result.baseline.dataset_fingerprint, result.tuned.dataset_fingerprint)
        self.assertEqual(result.baseline.scored_split, "test")
        # Test pages never overlap training pages.
        self.assertEqual(
            result.split.train_pages + result.split.val_pages + result.split.test_pages, 6
        )
        self.assertGreater(result.split.test_pages, 0)

    def test_tuned_summary_beats_the_sloppy_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = self._run(Path(tmp))
        rows = {row.variant: row for row in result.summary}
        self.assertAlmostEqual(rows["tuned"].map50, 1.0)
        self.assertLess(rows["baseline"].map50, 1.0)
        # Baseline half-width boxes lose half the content area.
        self.assertGreater(rows["baseline"].loss_rate, 0.4)
        self.assertAlmostEqual(rows["tuned"].loss_rate, 0.0)

    def test_reports_phases_and_persists_the_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, phases = self._run(Path(tmp))
            on_disk = json.loads((Path(tmp) / "out" / "comparison.json").read_text(encoding="utf-8"))
        self.assertIn("splitting", phases)
        self.assertTrue(any(phase.startswith("training") for phase in phases))
        self.assertIn("evaluating tuned model", phases)
        self.assertEqual(on_disk["summary"][0]["variant"], "baseline")
        self.assertEqual(on_disk["best_model_path"], result.best_model_path)

    def test_failed_training_raises_instead_of_scoring_nothing(self):
        from pdf_layout_parser import comparison

        def failing(config):
            return TrainingResult(
                status="failed",
                data_yaml=str(config.data_yaml),
                output_dir=str(config.output_dir),
                run_name=config.run_name,
                model_source="fake",
                command_preview="fake",
                error="boom",
            )

        with tempfile.TemporaryDirectory() as tmp:
            annotation_path = build_corpus(Path(tmp))
            with patch.object(comparison, "run_training", failing):
                with self.assertRaises(RuntimeError) as caught:
                    comparison.run_finetune_comparison(
                        annotation_path=annotation_path,
                        output_dir=Path(tmp) / "out",
                    )
        self.assertIn("boom", str(caught.exception))


class WebTrainingEndpointTests(unittest.TestCase):
    def _client(self):
        from fastapi.testclient import TestClient

        from backend.main import app

        return TestClient(app)

    def _payload(self, tmp: Path, complete_pages: int = 4, total_pages: int = 4):
        pages = []
        for index in range(total_pages):
            image_path = tmp / f"page_{index}.png"
            Image.new("RGB", (100, 200), "white").save(image_path)
            pages.append(
                {
                    "image_path": str(image_path),
                    "width": 100,
                    "height": 200,
                    "page_number": index + 1,
                    "label_status": "complete" if index < complete_pages else "in_progress",
                    "annotations": [
                        {"label": "body", "bbox": {"x": 10, "y": 10, "width": 60, "height": 80}}
                    ],
                }
            )
        return {"annotations": {"labels": ["body"], "pages": pages}}

    def test_rejects_fewer_than_three_labeled_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            response = self._client().post(
                "/training/fine-tune", json=self._payload(Path(tmp), complete_pages=2)
            )
        self.assertEqual(response.status_code, 422)
        self.assertIn("at least 3", response.json()["detail"])

    def test_unknown_job_is_404(self):
        response = self._client().get("/training/jobs/nope")
        self.assertEqual(response.status_code, 404)

    def test_job_lifecycle_with_patched_worker(self):
        """Patches the subprocess seam: the endpoint's contract is job creation,
        phase relay, result exposure, and on-disk persistence -- not training."""
        import backend.main as main

        def fake_worker(run_dir, annotation_path, request, pretrain, set_phase):
            set_phase("training (fake)")
            return {"best_model_path": str(main.PROJECT_ROOT / "runs" / "fake" / "best.pt")}

        client = self._client()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(main, "execute_comparison_job", fake_worker):
                response = client.post("/training/fine-tune", json=self._payload(Path(tmp)))
                self.assertEqual(response.status_code, 200, response.text)
                job_id = response.json()["job_id"]

                snapshot = None
                for _ in range(50):
                    snapshot = client.get(f"/training/jobs/{job_id}").json()
                    if snapshot["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.1)

        self.assertEqual(snapshot["status"], "completed", snapshot)
        self.assertEqual(snapshot["result"]["best_model_relpath"], "runs/fake/best.pt")
        # The job snapshot also survives on disk for post-restart inspection.
        self.assertTrue((Path(snapshot["output_dir"]) / "job.json").exists())

    def test_concurrent_job_is_refused(self):
        import backend.main as main

        def slow_worker(run_dir, annotation_path, request, pretrain, set_phase):
            time.sleep(1.0)
            return {"best_model_path": None}

        client = self._client()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(main, "execute_comparison_job", slow_worker):
                first = client.post("/training/fine-tune", json=self._payload(Path(tmp)))
                self.assertEqual(first.status_code, 200)
                second = client.post("/training/fine-tune", json=self._payload(Path(tmp)))
                self.assertEqual(second.status_code, 409)
                # Let the first job settle so it does not leak into other tests.
                job_id = first.json()["job_id"]
                for _ in range(30):
                    if client.get(f"/training/jobs/{job_id}").json()["status"] == "completed":
                        break
                    time.sleep(0.1)


if __name__ == "__main__":
    unittest.main()
