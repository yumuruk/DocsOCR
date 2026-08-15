"""Tests for the training wrapper's reproducibility surface.

The M3 findings were all of the form "the run happened but the record does not
say what ran": seed uncontrollable, freeze unavailable, results never written to
disk, and a hand-assembled preview that omitted most of the real arguments.
These tests pin the repaired contract: the recorded kwargs ARE the executed
kwargs, and every run (dry or real) leaves a file behind.
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from pdf_layout_parser.models import AnnotationBox, AnnotationPage, BoundingBox, TrainingConfig
from pdf_layout_parser.training import (
    build_training_preview,
    run_doclayout_yolo_training,
    run_page_count_experiments,
    training_kwargs,
)


def make_config(**overrides) -> TrainingConfig:
    defaults = dict(
        data_yaml=Path("dataset/data.yaml"),
        output_dir=Path("runs/training"),
        pretrain=Path("models/base.pt"),
        dry_run=True,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


class TrainingKwargsTests(unittest.TestCase):
    def test_seed_and_freeze_are_passed_to_train(self):
        kwargs = training_kwargs(make_config(seed=7, freeze=10))
        self.assertEqual(kwargs["seed"], 7)
        self.assertEqual(kwargs["freeze"], 10)

    def test_freeze_is_omitted_when_unset(self):
        """None must not reach ultralytics -- absence is how 'train everything'
        is expressed, not freeze=None."""
        self.assertNotIn("freeze", training_kwargs(make_config()))

    def test_preview_is_rendered_from_the_same_kwargs(self):
        """The preview and the live call share one dict, so every recorded
        argument (seed included) appears verbatim in the preview."""
        config = make_config(seed=3, freeze=11, learning_rate=0.005)
        preview = build_training_preview(config)
        self.assertIn("YOLOv10", preview)
        for key, value in training_kwargs(config).items():
            self.assertIn(f"{key}={value!r}", preview)


class DryRunRecordTests(unittest.TestCase):
    def test_dry_run_records_seed_freeze_and_kwargs(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_doclayout_yolo_training(
                make_config(output_dir=Path(tmp), seed=5, freeze=10)
            )
        self.assertEqual(result.status, "dry_run")
        self.assertEqual(result.seed, 5)
        self.assertEqual(result.freeze, 10)
        self.assertEqual(result.train_kwargs["seed"], 5)
        self.assertEqual(result.model_source, "models/base.pt")

    def test_result_is_written_to_disk(self):
        """A training run must leave a machine-readable record; previously the
        result object died with the process."""
        with tempfile.TemporaryDirectory() as tmp:
            result = run_doclayout_yolo_training(make_config(output_dir=Path(tmp)))
            self.assertIsNotNone(result.result_path)
            on_disk = json.loads(Path(result.result_path).read_text(encoding="utf-8"))
        self.assertEqual(on_disk["status"], "dry_run")
        self.assertEqual(on_disk["train_kwargs"]["data"], "dataset/data.yaml")

    def test_from_scratch_training_warns(self):
        """M3: pretrain=None silently trained from random init under a run name
        that said 'finetune'. Still allowed -- never silent."""
        with tempfile.TemporaryDirectory() as tmp:
            bare = run_doclayout_yolo_training(make_config(output_dir=Path(tmp), pretrain=None))
            tuned = run_doclayout_yolo_training(make_config(output_dir=Path(tmp)))
        self.assertTrue(any("FROM SCRATCH" in warning for warning in bare.warnings))
        self.assertFalse(tuned.warnings)


if __name__ == "__main__":
    unittest.main()
