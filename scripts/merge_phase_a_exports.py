"""Re-attach document identity to Phase A labelling exports.

The labelling bundles are one combined PDF per genre, so the web export knows
pages only as 1..N of that upload. This script maps each page back to its source
document via the manifest that prepare_phase_a.py wrote, injects document_id
(and the original source page), and emits one evaluation-ready annotation file
per genre. Without this step every genre would look like a single document and
document-level splitting/attribution would be meaningless.

Usage:
  uv run python scripts/merge_phase_a_exports.py \
      --manifest runs/demo/phase-a/manifest.json \
      --export receipt=~/Downloads/annotations.json \
      --export form=~/Downloads/annotations(1).json \
      --output runs/demo/phase-a/gt

Then evaluate zero-shot per genre:
  uv run pdf-layout-parser evaluate runs/demo/phase-a/gt/receipt.json runs/demo/phase-a/eval \
      --run-name zeroshot-receipt --detector doclayout-yolo \
      --model-path models/doclayout_yolo_docstructbench_imgsz1024.pt \
      --label-mapping docstructbench --split all --tag genre=receipt --tag init=zeroshot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_genre(genre: str, export_path: Path, manifest_rows: list[dict]) -> tuple[dict, list[str]]:
    data = json.loads(export_path.expanduser().read_text(encoding="utf-8"))
    pages = data.get("pages", [])
    notes: list[str] = []

    if len(pages) != len(manifest_rows):
        raise SystemExit(
            f"{genre}: export has {len(pages)} pages but the manifest expects "
            f"{len(manifest_rows)} -- is this the right file for this genre?"
        )

    by_page = {row["page"]: row for row in manifest_rows}
    complete = 0
    boxes = 0
    empty_complete = []
    for index, page in enumerate(pages, start=1):
        page_number = page.get("page_number", index)
        row = by_page.get(page_number)
        if row is None:
            raise SystemExit(f"{genre}: export page {page_number} not in the manifest")
        page["document_id"] = row["doc_id"]
        # Keep the combined page number for image lookup, record provenance
        # separately so nothing downstream has to guess.
        page["source_file"] = row["source_file"]
        page["source_page"] = row["source_page"]
        status = page.get("label_status", "complete")
        if status == "complete":
            complete += 1
            if not page.get("annotations"):
                empty_complete.append(page_number)
        boxes += len(page.get("annotations", []))

    documents = len({row["doc_id"] for row in manifest_rows})
    notes.append(
        f"{genre:14s} 문서 {documents}개 · 페이지 {complete}/{len(pages)} complete · 박스 {boxes}개"
    )
    if complete < len(pages):
        skipped = len(pages) - complete
        notes.append(
            f"  ⚠ {genre}: {skipped}페이지가 Labeled 미체크 상태 -- 평가에서 제외됩니다"
        )
    if empty_complete:
        notes.append(
            f"  ⚠ {genre}: complete인데 박스 0개인 페이지 {empty_complete} -- "
            "정말 빈 페이지가 맞는지 확인"
        )
    return data, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--export",
        action="append",
        required=True,
        metavar="GENRE=PATH",
        help="repeatable: which exported annotations.json belongs to which genre",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["genres"]
    args.output.mkdir(parents=True, exist_ok=True)

    all_notes: list[str] = []
    for item in args.export:
        genre, _, raw_path = item.partition("=")
        if not raw_path:
            raise SystemExit(f"--export must be GENRE=PATH, got {item!r}")
        if genre not in manifest:
            raise SystemExit(f"Unknown genre {genre!r}; manifest has {sorted(manifest)}")
        merged, notes = merge_genre(genre, Path(raw_path), manifest[genre])
        out = args.output / f"{genre}.json"
        out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        all_notes.extend(notes)
        all_notes.append(f"  -> {out}")

    print("\n".join(all_notes))


if __name__ == "__main__":
    main()
