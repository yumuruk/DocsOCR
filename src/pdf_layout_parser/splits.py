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
        """Pages available for training: the train split only.

        val is deliberately excluded. The val split is what training-time
        validation (early stopping, best.pt selection) scores against, so a page
        sampled from val into the training set would let the model be selected on
        data it trained on -- the same leak this module exists to prevent. An
        earlier version returned train+val and had exactly that problem.
        """
        return list(self.train)

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
    if not 0.0 < test_ratio < 1.0 or not 0.0 < val_ratio < 1.0:
        raise SplitError("test_ratio and val_ratio must each be in (0, 1)")
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

    # Placement order is size-descending so the largest documents are placed
    # while every split still has room -- a big document lands in the split with
    # the biggest deficit (usually train) instead of blowing past whichever
    # target a shuffled order happened to be filling. The previous version
    # filled test, then val, then gave train the leftovers, so a corpus like one
    # 60-page manual plus a few short documents either emptied train entirely or
    # sent the manual to test.
    #
    # The seed breaks ties among equal-size documents. For documents of distinct
    # sizes the assignment is therefore stable across seeds -- deliberately: the
    # protocol wants ONE frozen test set, and the seed's job is tie-breaking and
    # downstream sampling, not reshuffling what "held-out" means every run.
    shuffled = list(document_ids)
    random.Random(seed).shuffle(shuffled)
    shuffle_rank = {document_id: index for index, document_id in enumerate(shuffled)}
    placement_order = sorted(
        document_ids, key=lambda d: (-len(grouped[d]), shuffle_rank[d])
    )

    total_pages = len(usable)
    targets = {
        "train": (1.0 - test_ratio - val_ratio) * total_pages,
        "test": test_ratio * total_pages,
        "val": val_ratio * total_pages,
    }
    counts = {"train": 0, "test": 0, "val": 0}
    documents: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    # Each document goes to the relatively most under-filled split. Ties resolve
    # train -> test -> val, matching target size, so the first (largest) document
    # tends to seed train.
    for document_id in placement_order:
        chosen = min(("train", "test", "val"), key=lambda n: counts[n] / targets[n])
        documents[chosen].append(document_id)
        counts[chosen] += len(grouped[document_id])

    # Repair: with >= 3 documents every split can and must be non-empty. Move the
    # smallest document out of the most populated split until that holds.
    repaired = False
    for name in ("train", "test", "val"):
        if documents[name]:
            continue
        donor = max(
            (other for other in ("train", "test", "val") if len(documents[other]) >= 2),
            key=lambda n: len(documents[n]),
        )
        moved = min(documents[donor], key=lambda d: (len(grouped[d]), d))
        documents[donor].remove(moved)
        documents[name].append(moved)
        counts[donor] -= len(grouped[moved])
        counts[name] += len(grouped[moved])
        repaired = True
    if repaired:
        warnings.append(
            "Document sizes are too uneven to honour the requested ratios; at least one "
            "split had to be seeded with a single document. Check the split summary "
            "before trusting comparisons."
        )

    # Loud, not fatal: a heavily skewed corpus still splits, but the caller is
    # told the achieved ratios are far from the requested ones.
    for name in ("train", "test", "val"):
        deviation = abs(counts[name] - targets[name])
        if targets[name] > 0 and deviation / total_pages > 0.15:
            warnings.append(
                f"Split {name!r} holds {counts[name]} page(s) against a target of "
                f"{targets[name]:.1f} -- document size imbalance prevents a closer fit."
            )

    buckets: dict[str, list[AnnotationPage]] = {"train": [], "val": [], "test": []}
    for name in ("train", "val", "test"):
        for document_id in documents[name]:
            buckets[name].extend(grouped[document_id])
        documents[name].sort()
        buckets[name].sort(key=page_key)

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
