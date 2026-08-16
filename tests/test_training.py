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
from unittest.mock import patch

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


class FinalisationSalvageTests(unittest.TestCase):
    """doclayout_yolo's post-train finalisation (reloading best.pt for a last
    validation pass) crashed three unrelated ways on this setup while the
    checkpoint was already on disk. A finished training must not be reported as
    failed because of a crash in that redundant step."""

    def _run_with_fake_trainer(self, write_best: bool):
        import contextlib
        import types

        fake = types.ModuleType("doclayout_yolo")

        class FakeYOLOv10:
            def __init__(self, source):
                self.source = source

            def train(self, **kwargs):
                if write_best:
                    weights = Path(kwargs["project"]) / kwargs["name"] / "weights"
                    weights.mkdir(parents=True, exist_ok=True)
                    (weights / "best.pt").write_bytes(b"w")
                raise RuntimeError("'Conv' object has no attribute 'bn'")

        fake.YOLOv10 = FakeYOLOv10
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(output_dir=Path(tmp), dry_run=False)
            # Bypass the torch-load shim: importing torch inside the unittest
            # process (which also hosts FastAPI test threads) segfaulted on
            # macOS, and these tests exercise the salvage logic, not torch.
            with patch.dict(sys.modules, {"doclayout_yolo": fake}), patch(
                "pdf_layout_parser.torchcompat.trusted_checkpoint_loading",
                contextlib.nullcontext,
            ):
                return run_doclayout_yolo_training(config)

    def test_existing_checkpoint_is_salvaged_with_a_warning(self):
        result = self._run_with_fake_trainer(write_best=True)
        self.assertEqual(result.status, "completed")
        self.assertTrue(result.best_model_path.endswith("best.pt"))
        self.assertTrue(any("finalisation" in warning for warning in result.warnings))
        self.assertIsNone(result.error)

    def test_crash_without_a_checkpoint_still_fails(self):
        result = self._run_with_fake_trainer(write_best=False)
        self.assertEqual(result.status, "failed")
        self.assertIn("Conv", result.error)


class PageCountExperimentTests(unittest.TestCase):
    """Dry-run the experiment loop with real images: dataset export is not mocked."""

    def _corpus(self, root: Path) -> Path:
        pages = []
        for doc_index in range(6):
            for page_index in range(4):
                image_path = root / f"doc{doc_index}_p{page_index}.png"
                Image.new("RGB", (100, 200), "white").save(image_path)
                pages.append(
                    AnnotationPage(
                        image_path=str(image_path),
                        width=100,
                        height=200,
                        document_id=f"doc{doc_index}",
                        page_number=page_index + 1,
                        annotations=[
                            AnnotationBox(label="body", bbox=BoundingBox(x=0, y=0, width=50, height=50))
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

    def test_cells_share_val_and_test_and_nest_train(self):
        """The C2 regression: every N must be scored against the same held-out
        pages, and N=2's train pages must be a subset of N=4's."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = run_page_count_experiments(
                annotation_path=self._corpus(root),
                output_dir=root / "exp",
                sample_sizes=[4, 2],
                training_config=make_config(output_dir=root / "unused"),
                split_seed=0,
                sample_seed=0,
            )

            by_size = {result.sample_size: result for result in results}
            val_2 = Path(by_size[2].dataset.val_txt).read_text(encoding="utf-8")
            val_4 = Path(by_size[4].dataset.val_txt).read_text(encoding="utf-8")
            test_2 = Path(by_size[2].dataset.test_txt).read_text(encoding="utf-8")
            test_4 = Path(by_size[4].dataset.test_txt).read_text(encoding="utf-8")
            self.assertEqual(val_2, val_4)
            self.assertEqual(test_2, test_4)

            train_2 = set(Path(by_size[2].dataset.train_txt).read_text(encoding="utf-8").split())
            train_4 = set(Path(by_size[4].dataset.train_txt).read_text(encoding="utf-8").split())
            self.assertEqual(len(train_2), 2)
            self.assertEqual(len(train_4), 4)
            # Listings are copies named per dataset; compare source page identity.
            names_2 = {name.split("/")[-1] for name in train_2}
            names_4 = {name.split("/")[-1] for name in train_4}
            self.assertTrue(names_2.issubset(names_4))

    def test_cells_record_split_and_seed_in_run_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = run_page_count_experiments(
                annotation_path=self._corpus(root),
                output_dir=root / "exp",
                sample_sizes=[2],
                training_config=make_config(output_dir=root / "unused", seed=3),
            )
        result = results[0]
        self.assertEqual(result.training.run_name, "doclayout_pages_2_seed3")
        self.assertIsNotNone(result.split)
        self.assertEqual(result.split.train_pages, 2)
        self.assertEqual(result.sample_seed, 0)

    def test_manifest_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_page_count_experiments(
                annotation_path=self._corpus(root),
                output_dir=root / "exp",
                sample_sizes=[2],
                training_config=make_config(output_dir=root / "unused"),
            )
            manifest = json.loads((root / "exp" / "experiments.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["sample_size"], 2)
        self.assertEqual(manifest[0]["training"]["status"], "dry_run")

    def test_oversized_sample_refuses_up_front(self):
        """Asking for more train pages than the pool holds must fail before any
        cell trains, not silently shrink the cell."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(Exception):
                run_page_count_experiments(
                    annotation_path=self._corpus(root),
                    output_dir=root / "exp",
                    sample_sizes=[500],
                    training_config=make_config(output_dir=root / "unused"),
                )


if __name__ == "__main__":
    unittest.main()
