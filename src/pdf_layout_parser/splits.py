"""Leak-free dataset splits.

Replaces the previous split, which was a seedless prefix slice: ``pages[:k]`` for
train and ``pages[k:]`` for val, with a special case that returned
``(pages, pages)`` for a single page -- putting the same page in both sides.

Three separate defects are fixed here:

* **Train/val leakage.** Pages from one document are near-duplicates of each
  other, so a page-level split leaks layout. Splitting by document is the only
  honest option, which is why ``AnnotationPage`` gained ``document_id``.
* **No held-out test set.** With per-N val sets, N=10 and N=20 were scored on
  different pages, so points on a label-efficiency curve could not be compared.
  One test split is carved out first, fixed by seed, and never trained on.
* **Order bias.** A prefix slice always put the back of the document in val.
  Documents are shuffled with an explicit seed after being sorted into a
  canonical order, so the result depends on the seed and not on input order.

Sampling for the N-page curve is *nested*: the pages used at N=10 are a subset of
those used at N=20. Otherwise a larger N could score worse purely because it drew
harder pages, and the curve would measure luck.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from pdf_layout_parser.models import AnnotationPage, LabelStatus, SplitSummary

UNKNOWN_DOCUMENT = "__unknown__"


class SplitError(ValueError):
    """Raised when a requested split cannot be produced without leaking."""


def document_id_for(page: AnnotationPage) -> str:
    """Document identity for a page, falling back to its image path.

    Explicit ``document_id`` wins. Otherwise the path is used, because the parse
    pipeline writes one directory per run: ``runs/web/<uuid>/pages/page_0001.png``
    means the document is ``<uuid>``. This is a fallback, not a substitute -- it
    breaks as soon as two documents share an output directory, so the exporter
    warns when it has to guess.
    """
    if page.document_id:
        return page.document_id
    path = Path(page.image_path)
    parent = path.parent
    if parent.name in {"pages", "images"} and parent.parent.name:
        return parent.parent.name
    if parent.name:
        return parent.name
    return UNKNOWN_DOCUMENT


def page_key(page: AnnotationPage) -> tuple[str, int, str]:
    """Stable sort key, so shuffling is reproducible regardless of input order."""
    return (
        document_id_for(page),
        page.page_number if page.page_number is not None else 0,
        page.image_path,
    )


@dataclass(frozen=True)
class DatasetSplits:
    train: list[AnnotationPage]
    val: list[AnnotationPage]
    test: list[AnnotationPage]
    seed: int
    strategy: str
    documents: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def train_pool(self) -> list[AnnotationPage]:
        """Pages available for training, i.e. everything outside test."""
        return [*self.train, *self.val]

    def summary(self) -> SplitSummary:
        return SplitSummary(
            seed=self.seed,
            strategy=self.strategy,
            train_pages=len(self.train),
            val_pages=len(self.val),
            test_pages=len(self.test),
            train_documents=self.documents.get("train", []),
            val_documents=self.documents.get("val", []),
            test_documents=self.documents.get("test", []),
        )


def complete_pages(pages: list[AnnotationPage]) -> list[AnnotationPage]:
    """Only pages whose annotation is finished.

    An in-progress page exports as a page with a few boxes on it, which trains and
    scores as though the unlabelled rest of the page is background.
    """
    return [page for page in pages if page.label_status == LabelStatus.COMPLETE]


def group_by_document(pages: list[AnnotationPage]) -> dict[str, list[AnnotationPage]]:
    grouped: dict[str, list[AnnotationPage]] = {}
    for page in sorted(pages, key=page_key):
        grouped.setdefault(document_id_for(page), []).append(page)
    return grouped


def make_splits(
    pages: list[AnnotationPage],
    seed: int = 0,
    test_ratio: float = 0.35,
    val_ratio: float = 0.15,
    strategy: str = "document",
    require_complete: bool = True,
) -> DatasetSplits:
    """Partition pages into train / val / test.

    ``strategy="document"`` keeps every page of a document inside one split.
    ``strategy="page"`` splits individual pages and is only appropriate when the
    corpus genuinely is a single document; it leaks layout similarity and says so
    in ``warnings``.
    """
    if not 0.0 <= test_ratio < 1.0 or not 0.0 <= val_ratio < 1.0:
        raise SplitError("test_ratio and val_ratio must each be in [0, 1)")
    if test_ratio + val_ratio >= 1.0:
        raise SplitError("test_ratio + val_ratio must leave pages for training")

    warnings: list[str] = []
    usable = complete_pages(pages) if require_complete else list(pages)
    skipped = len(pages) - len(usable)
    if skipped:
        warnings.append(f"{skipped} page(s) skipped because label_status is not 'complete'")
    if not usable:
        raise SplitError("No pages with label_status='complete' to split")

    if not any(page.document_id for page in usable):
        warnings.append(
            "No page carries an explicit document_id; document identity was inferred "
            "from image paths. Set document_id to make splits reliable."
        )

    if strategy == "page":
        return _page_level_splits(usable, seed, test_ratio, val_ratio, warnings)
    if strategy != "document":
        raise SplitError(f"Unknown split strategy {strategy!r}")

    grouped = group_by_document(usable)
    document_ids = sorted(grouped)
    if len(document_ids) < 3:
        raise SplitError(
            f"Document-level splitting needs at least 3 documents, found {len(document_ids)} "
            f"({', '.join(document_ids)}). Label pages from more documents, or pass "
            "strategy='page' and accept that train/test share a document."
        )

    shuffled = list(document_ids)
    random.Random(seed).shuffle(shuffled)

    total_pages = len(usable)
    test_target = test_ratio * total_pages
    val_target = val_ratio * total_pages

    assignment: dict[str, str] = {}
    test_count = 0
    val_count = 0
    for document_id in shuffled:
        size = len(grouped[document_id])
        if test_count < test_target:
            assignment[document_id] = "test"
            test_count += size
        elif val_count < val_target:
            assignment[document_id] = "val"
            val_count += size
        else:
            assignment[document_id] = "train"

    buckets: dict[str, list[AnnotationPage]] = {"train": [], "val": [], "test": []}
    documents: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for document_id in shuffled:
        target = assignment[document_id]
        buckets[target].extend(grouped[document_id])
        documents[target].append(document_id)

    for name in ("train", "val", "test"):
        documents[name].sort()
        buckets[name].sort(key=page_key)
        if not buckets[name]:
            raise SplitError(
                f"Split {name!r} came out empty with seed={seed}, test_ratio={test_ratio}, "
                f"val_ratio={val_ratio} over {len(document_ids)} documents. "
                "Adjust the ratios or label more documents."
            )

    splits = DatasetSplits(
        train=buckets["train"],
        val=buckets["val"],
        test=buckets["test"],
        seed=seed,
        strategy="document",
        documents=documents,
        warnings=warnings,
    )
    assert_no_leak(splits)
    return splits


def _page_level_splits(
    pages: list[AnnotationPage],
    seed: int,
    test_ratio: float,
    val_ratio: float,
    warnings: list[str],
) -> DatasetSplits:
    if len(pages) < 3:
        raise SplitError(
            f"Need at least 3 complete pages to split, found {len(pages)}. "
            "The previous implementation returned the same page as both train and val."
        )
    warnings = [
        *warnings,
        "strategy='page' splits pages from the same document across train/val/test. "
        "Pages of one document are near-duplicates, so scores from this split are "
        "optimistic and must not be compared with document-level results.",
    ]
    ordered = sorted(pages, key=page_key)
    random.Random(seed).shuffle(ordered)

    test_count = max(1, round(len(ordered) * test_ratio))
    val_count = max(1, round(len(ordered) * val_ratio))
    if test_count + val_count >= len(ordered):
        raise SplitError(
            f"Ratios leave no training pages: {len(ordered)} pages, "
            f"test={test_count}, val={val_count}"
        )

    test = ordered[:test_count]
    val = ordered[test_count : test_count + val_count]
    train = ordered[test_count + val_count :]

    splits = DatasetSplits(
        train=sorted(train, key=page_key),
        val=sorted(val, key=page_key),
        test=sorted(test, key=page_key),
        seed=seed,
        strategy="page",
        documents={
            "train": sorted({document_id_for(page) for page in train}),
            "val": sorted({document_id_for(page) for page in val}),
            "test": sorted({document_id_for(page) for page in test}),
        },
        warnings=warnings,
    )
    assert_no_leak(splits)
    return splits


def assert_no_leak(splits: DatasetSplits) -> None:
    """Fail loudly if any page appears in more than one split."""
    seen: dict[str, str] = {}
    for name, pages in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        for page in pages:
            previous = seen.get(page.image_path)
            if previous is not None:
                raise SplitError(
                    f"Page {page.image_path!r} appears in both {previous!r} and {name!r}"
                )
            seen[page.image_path] = name


def sample_train_pages(
    pool: list[AnnotationPage],
    sample_size: int,
    seed: int = 0,
) -> list[AnnotationPage]:
    """Draw ``sample_size`` pages from ``pool``, nested across sample sizes.

    The pool is shuffled once with ``seed`` and sampling takes a prefix, so
    ``sample_train_pages(pool, 10, seed)`` is a subset of
    ``sample_train_pages(pool, 20, seed)``. A label-efficiency curve built this
    way measures added data, not which pages happened to be drawn.
    """
    if sample_size <= 0:
        raise SplitError("sample_size must be positive")
    ordered = sorted(pool, key=page_key)
    random.Random(seed).shuffle(ordered)
    if sample_size > len(ordered):
        raise SplitError(
            f"Requested {sample_size} training pages but the pool holds {len(ordered)}. "
            "Label more pages, or lower the sample size."
        )
    return sorted(ordered[:sample_size], key=page_key)
