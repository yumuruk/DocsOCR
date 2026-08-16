"""Build the Phase A labelling bundles: one combined PDF per genre.

Phase A measures how well the zero-shot model separates content from furniture
across genres, so the labelling corpus follows the docs-over-pages rule: many
documents, few pages each (a genre verdict from 2 documents cannot be told apart
from 2 peculiar documents). Uploading 30 documents one by one is needless
friction, so the selected pages are merged into one PDF per genre -- six uploads
total -- and a manifest records which combined page came from which document so
the merge script can re-inject document identity after labelling.

Page selection is stratified, not random: the first page (title/cover furniture
lives there) plus the highest table/figure-density pages, because tables are the
scarcest class that matters and uniform sampling would under-sample them.

Deterministic: document choice is seeded, page scoring has no randomness.
Re-running overwrites runs/demo/phase-a/ with identical output.

Usage:  uv run python scripts/prepare_phase_a.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "runs" / "demo" / "phase-a"
SEED = 0

# genre -> (source dir, documents to take, pages per document)
GENRES: dict[str, tuple[str, int, int]] = {
    # Receipts are single-page documents: the docs-over-pages principle at its
    # purest, so take more documents instead of more pages.
    "receipt": ("runs/demo/genres/receipt", 10, 1),
    "ti-datasheet": ("runs/demo/family/ti-datasheet", 5, 3),
    "form": ("runs/demo/genres/form", 5, 2),
    "paper": ("runs/demo/genres/paper", 5, 2),
    "minutes": ("runs/demo/genres/minutes", 5, 2),
    "manual": ("runs/demo/genres/manual", 5, 3),
}

# Never score more than this many pages of one document; huge manuals (5000+
# pages) would otherwise dominate runtime for no selection benefit.
MAX_SCAN_PAGES = 90


def page_score(page: fitz.Page) -> float:
    """Content-density score used to pick the non-first pages.

    Tables dominate the score because table is the scarcest class the
    experiment cares about; images stand in for figures; text length breaks
    ties so empty/divider pages sink.
    """
    try:
        tables = len(page.find_tables().tables)
    except Exception:
        tables = 0
    images = len(page.get_images(full=False))
    text_length = len(page.get_text())
    return tables * 10.0 + min(images, 3) * 2.0 + min(text_length, 5000) / 5000.0


def select_pages(pdf_path: Path, pages_wanted: int) -> list[int]:
    """0-based source pages: page 0, then the densest of the scanned window."""
    with fitz.open(pdf_path) as doc:
        total = len(doc)
        if total <= pages_wanted:
            return list(range(total))
        stride = max(1, total // MAX_SCAN_PAGES)
        candidates = list(range(1, total, stride))[:MAX_SCAN_PAGES]
        scored = sorted(
            ((page_score(doc[index]), index) for index in candidates),
            key=lambda item: (-item[0], item[1]),
        )
        picked = [0] + [index for _, index in scored[: pages_wanted - 1]]
        return sorted(set(picked))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, list[dict]] = {}
    summary_rows: list[str] = []

    for genre, (source_dir, doc_count, pages_per_doc) in GENRES.items():
        pdfs = sorted((PROJECT_ROOT / source_dir).glob("*.pdf"))
        if len(pdfs) < doc_count:
            raise SystemExit(f"{genre}: need {doc_count} documents, found {len(pdfs)}")
        chosen = sorted(random.Random(SEED).sample(pdfs, doc_count), key=lambda p: p.stem)

        combined = fitz.open()
        rows: list[dict] = []
        for pdf_path in chosen:
            for source_page in select_pages(pdf_path, pages_per_doc):
                with fitz.open(pdf_path) as src:
                    combined.insert_pdf(src, from_page=source_page, to_page=source_page)
                rows.append(
                    {
                        "page": len(combined),  # 1-based page in the combined PDF
                        "doc_id": pdf_path.stem,
                        "source_file": str(pdf_path.relative_to(PROJECT_ROOT)),
                        "source_page": source_page + 1,  # 1-based for humans
                    }
                )

        out_pdf = OUTPUT_DIR / f"{genre}.pdf"
        combined.save(out_pdf)
        combined.close()
        manifest[genre] = rows
        documents = len({row["doc_id"] for row in rows})
        summary_rows.append(f"{genre:14s} {documents}개 문서, {len(rows)}페이지 -> {out_pdf.name}")

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps({"seed": SEED, "genres": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_pages = sum(len(rows) for rows in manifest.values())
    total_docs = sum(len({row["doc_id"] for row in rows}) for rows in manifest.values())
    print("\n".join(summary_rows))
    print(f"합계: {total_docs}개 문서, {total_pages}페이지")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
