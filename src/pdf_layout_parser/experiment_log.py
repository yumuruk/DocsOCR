"""Append-only markdown ledger for evaluation runs.

Three files, three jobs, deliberately kept apart:

* ``docs/EVALUATION.md`` -- the protocol. What is measured, how, and why. Written
  by hand, changes rarely.
* ``docs/EXPERIMENTS.md`` -- the ledger. One row per scored run, appended by this
  module. Never rewritten: a result that looked wrong later is still what the run
  produced, and silently editing it is how a record stops being evidence.
* ``docs/RESEARCH_LOG.md`` -- the narrative. Dated entries on what was decided and
  why. Written by hand; :func:`append_note` exists for machine-generated entries.

Every ledger row carries the dataset fingerprint and the seed, so two rows that
disagree can be told apart -- different data, different split, or genuinely
different model.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pdf_layout_parser.models import EvaluationReport

LEDGER_PATH = Path("docs/EXPERIMENTS.md")
RESEARCH_LOG_PATH = Path("docs/RESEARCH_LOG.md")

TABLE_MARKER = "<!-- ledger:rows -->"

LEDGER_HEADER = f"""# Experiment ledger

Append-only record of every scored run. Rows are added by
`pdf_layout_parser.experiment_log.append_run` (the `evaluate` CLI does this by
default). **Do not edit or delete rows.** If a run turns out to have been
misconfigured, add a new row and say so in `RESEARCH_LOG.md` -- the old row is
still what happened.

Column meanings are defined in [EVALUATION.md](EVALUATION.md). Short version:
`reduction` = OCR area saved vs full-page, `loss` = ground truth content
discarded (hard constraint), `contam` = share of extracted area that is not
content.

`data` is a content hash of **the pages that were scored**, not of the whole
corpus. Two rows sharing it were scored on byte-identical ground truth and can be
compared directly. Two rows differing in it were not -- usually because the split
seed differs, which makes them replicates rather than comparable cells. Check this
column before reading any difference between rows as a model difference.

| # | Date | Run | Cell | Split | Pages | mAP@.5 | mAP@.5:.95 | Reduction | Loss | Contam | data | seed | Report |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{TABLE_MARKER}
"""

RESEARCH_LOG_HEADER = """# Research log

Dated, narrative entries: what question was being asked, what the run showed, and
what changed as a result. The ledger in [EXPERIMENTS.md](EXPERIMENTS.md) records
numbers; this file records reasoning. Newest entries at the bottom.
"""


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _format_cell(tags: dict[str, str]) -> str:
    """Render the experiment coordinates, e.g. ``n=20 init=docstructbench``."""
    if not tags:
        return "—"
    return " ".join(f"{key}={value}" for key, value in sorted(tags.items()))


def ensure_ledger(path: Path = LEDGER_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(LEDGER_HEADER, encoding="utf-8")
    elif TABLE_MARKER not in path.read_text(encoding="utf-8"):
        raise ValueError(
            f"{path} exists but has no {TABLE_MARKER!r} marker; refusing to guess where rows go."
        )
    return path


def count_rows(path: Path = LEDGER_PATH) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8")
    _, _, after = text.partition(TABLE_MARKER)
    return sum(1 for line in after.splitlines() if line.startswith("| "))


def append_run(
    report: EvaluationReport,
    ledger_path: Path = LEDGER_PATH,
    report_link: str | None = None,
    tags: dict[str, str] | None = None,
    timestamp: str | None = None,
) -> str:
    """Append one row for ``report`` and return the rendered row.

    ``timestamp`` can be supplied for deterministic tests; otherwise now().
    """
    path = ensure_ledger(ledger_path)
    text = path.read_text(encoding="utf-8")

    model = next((entry for entry in report.operational if entry.variant == "model"), None)
    detection = report.detection
    effective_tags = dict(report.tags or {})
    effective_tags.update(tags or {})
    stamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M")
    index = count_rows(path) + 1
    split_label = (
        f"{report.split.strategy}/{report.scored_split}" if report.split else (report.scored_split or "—")
    )
    link = f"[md]({report_link})" if report_link else "—"

    row = (
        f"| {index} | {stamp} | `{report.run_name}` | {_format_cell(effective_tags)} | "
        f"{split_label} | {report.page_count} | "
        f"{_score(detection.map50 if detection else None)} | "
        f"{_score(detection.map5095 if detection else None)} | "
        f"{_percent(model.reduction_rate if model else None)} | "
        f"{_percent(model.loss_rate if model else None)} | "
        f"{_percent(model.contamination_rate if model else None)} | "
        f"`{report.dataset_fingerprint}` | "
        f"{report.seed if report.seed is not None else '—'} | {link} |"
    )

    # Appended at the end so rows read oldest-first and the numbering ascends
    # downward. Inserting after the marker would reverse the ledger.
    path.write_text(text.rstrip("\n") + f"\n{row}\n", encoding="utf-8")
    return row


def append_note(
    title: str,
    body: str,
    log_path: Path = RESEARCH_LOG_PATH,
    timestamp: str | None = None,
) -> None:
    """Append a dated entry to the research log."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(RESEARCH_LOG_HEADER, encoding="utf-8")
    stamp = timestamp or datetime.now().strftime("%Y-%m-%d")
    existing = path.read_text(encoding="utf-8").rstrip()
    entry = f"\n\n## {stamp} — {title}\n\n{body.strip()}\n"
    path.write_text(existing + entry, encoding="utf-8")


def load_reports(directory: Path) -> list[EvaluationReport]:
    """Load every ``*.json`` evaluation report in a directory, newest name last."""
    reports: list[EvaluationReport] = []
    for path in sorted(Path(directory).glob("*.json")):
        if path.name.endswith("_predictions.json"):
            continue
        try:
            reports.append(EvaluationReport.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:
            # A directory of run outputs holds other JSON too; skip what is not a report.
            continue
    return reports


def parse_tags(values: list[str] | None) -> dict[str, str]:
    """Parse ``key=value`` CLI tags into the experiment cell coordinates."""
    tags: dict[str, str] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"Tag {raw!r} must be key=value, e.g. n=20")
        key, _, value = raw.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"Tag {raw!r} has an empty key")
        tags[key] = value.strip()
    return tags
