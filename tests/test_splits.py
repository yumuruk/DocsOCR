"""Tests for leak-free splitting.

C1 and C2 in FINDINGS.md are both split defects, and neither was caught by the
existing suite. These tests exist so a regression fails loudly rather than
producing a plausible-looking curve.
"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdf_layout_parser.models import AnnotationBox, AnnotationPage, BoundingBox, LabelStatus
from pdf_layout_parser.splits import (
    SplitError,
    assert_no_leak,
    document_id_for,
    make_splits,
    sample_train_pages,
)


def make_corpus(spec: dict[str, int]) -> list[AnnotationPage]:
    """Pages for documents of explicitly uneven sizes, e.g. {"manual": 60, "a": 8}."""
    pages = []
    for document_id, page_count in spec.items():
        for page_index in range(page_count):
            pages.append(
                AnnotationPage(
                    image_path=f"runs/{document_id}/pages/page_{page_index + 1:04d}.png",
                    width=1240,
                    height=1754,
                    document_id=document_id,
                    page_number=page_index + 1,
                    annotations=[
                        AnnotationBox(label="body", bbox=BoundingBox(x=0, y=0, width=100, height=100))
                    ],
                )
            )
    return pages


def make_pages(document_count: int, pages_per_document: int) -> list[AnnotationPage]:
    pages = []
    for document_index in range(document_count):
        for page_index in range(pages_per_document):
            pages.append(
                AnnotationPage(
                    image_path=f"runs/doc{document_index}/pages/page_{page_index + 1:04d}.png",
                    width=1240,
                    height=1754,
                    document_id=f"doc{document_index}",
                    page_number=page_index + 1,
                    annotations=[
                        AnnotationBox(label="body", bbox=BoundingBox(x=0, y=0, width=100, height=100))
                    ],
                )
            )
    return pages


class DocumentIdentityTests(unittest.TestCase):
    def test_explicit_document_id_wins(self):
        page = AnnotationPage(image_path="a/b/c.png", width=1, height=1, document_id="explicit")
        self.assertEqual(document_id_for(page), "explicit")

    def test_document_id_inferred_from_run_directory(self):
        page = AnnotationPage(
            image_path="runs/web/8f2c-uuid/pages/page_0001.png", width=1, height=1
        )
        self.assertEqual(document_id_for(page), "8f2c-uuid")


class SplitTests(unittest.TestCase):
    def test_no_page_appears_in_two_splits(self):
        splits = make_splits(make_pages(5, 4), seed=0)
        assert_no_leak(splits)
        paths = [page.image_path for page in (*splits.train, *splits.val, *splits.test)]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(len(paths), 20)

    def test_documents_do_not_straddle_splits(self):
        """C1: pages of one document are near-duplicates, so they must stay together."""
        splits = make_splits(make_pages(6, 3), seed=1)
        train_documents = {document_id_for(page) for page in splits.train}
        val_documents = {document_id_for(page) for page in splits.val}
        test_documents = {document_id_for(page) for page in splits.test}
        self.assertFalse(train_documents & test_documents)
        self.assertFalse(train_documents & val_documents)
        self.assertFalse(val_documents & test_documents)

    def test_split_is_deterministic_for_a_seed(self):
        first = make_splits(make_pages(5, 4), seed=42)
        second = make_splits(make_pages(5, 4), seed=42)
        self.assertEqual(
            [page.image_path for page in first.test],
            [page.image_path for page in second.test],
        )

    def test_split_is_independent_of_input_order(self):
        """A prefix split made val depend on page order; seeded shuffling must not."""
        pages = make_pages(5, 4)
        forward = make_splits(pages, seed=3)
        backward = make_splits(list(reversed(pages)), seed=3)
        self.assertEqual(
            [page.image_path for page in forward.test],
            [page.image_path for page in backward.test],
        )

    def test_different_seeds_generally_give_different_test_sets(self):
        by_seed = {
            seed: tuple(page.image_path for page in make_splits(make_pages(8, 2), seed=seed).test)
            for seed in range(4)
        }
        self.assertGreater(len(set(by_seed.values())), 1)

    def test_single_page_no_longer_lands_in_both_train_and_val(self):
        """The exact defect: `if len(pages) == 1: return pages, pages`."""
        with self.assertRaises(SplitError):
            make_splits(make_pages(1, 1), seed=0)

    def test_document_strategy_requires_several_documents(self):
        with self.assertRaises(SplitError) as caught:
            make_splits(make_pages(2, 10), seed=0)
        self.assertIn("at least 3 documents", str(caught.exception))

    def test_page_strategy_is_allowed_but_says_it_leaks(self):
        splits = make_splits(make_pages(1, 12), seed=0, strategy="page")
        self.assertTrue(any("near-duplicates" in warning for warning in splits.warnings))
        assert_no_leak(splits)

    def test_incomplete_pages_are_excluded(self):
        pages = make_pages(4, 3)
        pages[0].label_status = LabelStatus.IN_PROGRESS
        pages[1].label_status = LabelStatus.UNLABELED
        splits = make_splits(pages, seed=0)
        total = len(splits.train) + len(splits.val) + len(splits.test)
        self.assertEqual(total, 10)
        self.assertTrue(any("label_status" in warning for warning in splits.warnings))

    def test_missing_document_id_is_flagged(self):
        pages = [
            AnnotationPage(image_path=f"runs/doc{index}/pages/page_0001.png", width=10, height=10)
            for index in range(4)
        ]
        splits = make_splits(pages, seed=0)
        self.assertTrue(any("document_id" in warning for warning in splits.warnings))

    def test_summary_reports_the_split(self):
        splits = make_splits(make_pages(5, 4), seed=9)
        summary = splits.summary()
        self.assertEqual(summary.seed, 9)
        self.assertEqual(summary.strategy, "document")
        self.assertEqual(
            summary.train_pages + summary.val_pages + summary.test_pages, 20
        )
        self.assertTrue(summary.test_documents)


class UnevenCorpusTests(unittest.TestCase):
    """The realistic corpus shape: one long manual plus a few short documents.

    The previous allocator filled test, then val, and gave train the leftovers,
    so this shape either emptied train entirely (SplitError on half the seeds)
    or sent the manual to test. These tests pin the repaired behaviour: every
    seed must produce a usable, non-empty, leak-free split with train holding
    the bulk of the pages.
    """

    CORPUS = {"manual": 60, "a": 8, "b": 6, "c": 5, "d": 4, "e": 3}

    def test_every_seed_splits_without_error(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                splits = make_splits(make_corpus(self.CORPUS), seed=seed)
                assert_no_leak(splits)
                self.assertTrue(splits.train and splits.val and splits.test)

    def test_the_large_document_lands_in_train(self):
        """Size-descending placement: the manual must seed train, not test."""
        for seed in range(10):
            splits = make_splits(make_corpus(self.CORPUS), seed=seed)
            train_documents = {document_id_for(page) for page in splits.train}
            self.assertIn("manual", train_documents, f"seed={seed}")

    def test_train_holds_enough_pages_for_the_label_efficiency_curve(self):
        """The n=50 cell needs 50 train pages; the old allocator left 3-12."""
        splits = make_splits(make_corpus(self.CORPUS), seed=0)
        self.assertGreaterEqual(len(splits.train_pool), 50)

    def test_extreme_imbalance_warns_instead_of_failing(self):
        corpus = {"big": 100, **{f"s{index}": 1 for index in range(6)}}
        splits = make_splits(make_corpus(corpus), seed=0)
        self.assertTrue(splits.train and splits.val and splits.test)
        self.assertTrue(
            any("imbalance" in warning or "uneven" in warning for warning in splits.warnings),
            splits.warnings,
        )

    def test_zero_ratio_is_rejected(self):
        with self.assertRaises(SplitError):
            make_splits(make_pages(4, 3), seed=0, val_ratio=0.0)


class NestedSamplingTests(unittest.TestCase):
    """C2: the N-page curve is only interpretable if the samples are nested."""

    def test_smaller_sample_is_a_subset_of_the_larger(self):
        pool = make_pages(10, 5)
        small = {page.image_path for page in sample_train_pages(pool, 10, seed=0)}
        large = {page.image_path for page in sample_train_pages(pool, 20, seed=0)}
        self.assertTrue(small.issubset(large))
        self.assertEqual(len(small), 10)
        self.assertEqual(len(large), 20)

    def test_sampling_is_deterministic(self):
        pool = make_pages(6, 5)
        first = [page.image_path for page in sample_train_pages(pool, 12, seed=5)]
        second = [page.image_path for page in sample_train_pages(pool, 12, seed=5)]
        self.assertEqual(first, second)

    def test_requesting_more_pages_than_available_is_an_error(self):
        with self.assertRaises(SplitError) as caught:
            sample_train_pages(make_pages(2, 3), 50, seed=0)
        self.assertIn("pool holds", str(caught.exception))

    def test_train_pool_excludes_test(self):
        splits = make_splits(make_pages(6, 4), seed=2)
        pool_paths = {page.image_path for page in splits.train_pool}
        test_paths = {page.image_path for page in splits.test}
        self.assertFalse(pool_paths & test_paths)

    def test_train_pool_excludes_val(self):
        """train_pool once returned train+val, so pages sampled for training could
        sit in the very split that early stopping selects the model on."""
        splits = make_splits(make_pages(6, 4), seed=2)
        pool_paths = {page.image_path for page in splits.train_pool}
        val_paths = {page.image_path for page in splits.val}
        self.assertTrue(val_paths)
        self.assertFalse(pool_paths & val_paths)


if __name__ == "__main__":
    unittest.main()
