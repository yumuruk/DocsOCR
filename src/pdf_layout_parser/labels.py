"""Canonical label catalogue and label-space mapping.

Class ids MUST be issued from this module and nowhere else.

The previous behaviour derived class ids from the order of the ``labels`` array in
whichever annotation JSON happened to be loaded (``dataset.py`` ``label_to_id``).
That makes the id of a class a property of a *file* rather than of the project, so
two datasets exported at different times can disagree about what class ``3`` means.
Any cross-dataset, transfer, or "does more data help" comparison built on that is
silently meaningless. Hence: one fixed, ordered catalogue, one place to ask.
"""

from __future__ import annotations

import json
from pathlib import Path

from pdf_layout_parser.models import DEFAULT_LABEL_POLICIES, ExtractionAction, LabelPolicy

#: Canonical label order. Appending is safe; reordering or removing is a breaking
#: change that invalidates every previously trained checkpoint and exported label file.
CANONICAL_LABELS: tuple[str, ...] = (
    "title",
    "body",
    "table",
    "figure",
    "header",
    "footer",
    "ignore",
    "target_data",
)

LABEL_TO_ID: dict[str, int] = {label: index for index, label in enumerate(CANONICAL_LABELS)}
ID_TO_LABEL: dict[int, str] = {index: label for label, index in LABEL_TO_ID.items()}

#: Sentinel used by report tables for detections that matched no ground truth box.
BACKGROUND = "__background__"
#: Sentinel used by report tables for ground truth boxes that no detection matched.
MISSED = "__missed__"


class UnknownLabelError(KeyError):
    """Raised when a label is not part of the canonical catalogue."""


def canonical_id(label: str) -> int:
    """Return the canonical class id for ``label``.

    Raises rather than skipping: a silently dropped label is how ground truth
    quietly loses boxes (the previous ``dataset.py`` behaviour).
    """
    try:
        return LABEL_TO_ID[label]
    except KeyError as exc:
        raise UnknownLabelError(
            f"{label!r} is not a canonical label. Known labels: {', '.join(CANONICAL_LABELS)}"
        ) from exc


def is_canonical(label: str) -> bool:
    return label in LABEL_TO_ID


# ---------------------------------------------------------------------------
# DocStructBench -> canonical mapping
# ---------------------------------------------------------------------------

#: The DocStructBench fine-tuned DocLayout-YOLO checkpoint emits these 10 classes.
#: Without a mapping the adapter (``detectors.py``) only lowercases them, so a
#: zero-shot baseline is scored against a label space it never predicts in --
#: per-class AP and the confusion matrix are both meaningless in that state.
DOCSTRUCTBENCH_CLASSES: tuple[str, ...] = (
    "title",
    "plain text",
    "abandon",
    "figure",
    "figure_caption",
    "table",
    "table_caption",
    "table_footnote",
    "isolate_formula",
    "formula_caption",
)

#: Mapping to the canonical catalogue. ``None`` means "drop this prediction".
#:
#: Four of these are judgement calls that move the numbers, so they are stated
#: explicitly rather than buried in a lowercase() call:
#:
#: * ``abandon`` -> ``ignore``: DocStructBench uses ``abandon`` for exactly the
#:   discardable furniture this project calls ignore/header/footer (running heads,
#:   page numbers). It cannot be split into header vs footer without geometry, so
#:   it maps to ``ignore`` -- all three are SKIP under the default policy, which
#:   means the operational metrics are unaffected by the conflation while
#:   per-class AP for header/footer is necessarily 0 in zero-shot.
#: * ``*_caption`` / ``table_footnote`` -> ``body``: captions and footnotes are
#:   OCR-able prose, so treating them as body matches the extraction policy.
#: * ``isolate_formula`` -> ``figure``: Tesseract cannot read formulas, so the
#:   operationally correct action is SKIP, which is what ``figure`` gets.
#: * ``target_data`` has no DocStructBench counterpart. Its zero-shot AP is
#:   therefore structurally 0 and must not be read as a model failure --
#:   ``evaluation.report`` annotates this.
DOCSTRUCTBENCH_TO_CANONICAL: dict[str, str | None] = {
    "title": "title",
    "plain text": "body",
    "abandon": "ignore",
    "figure": "figure",
    "figure_caption": "body",
    "table": "table",
    "table_caption": "body",
    "table_footnote": "body",
    "isolate_formula": "figure",
    "formula_caption": "body",
}

#: Identity mapping, for checkpoints fine-tuned on the canonical catalogue.
#: A fine-tuned model learns the canonical labels directly, so mapping it through
#: DOCSTRUCTBENCH_TO_CANONICAL would corrupt its output.
IDENTITY_MAPPING: dict[str, str | None] = {label: label for label in CANONICAL_LABELS}

LABEL_MAPPINGS: dict[str, dict[str, str | None]] = {
    "identity": IDENTITY_MAPPING,
    "docstructbench": DOCSTRUCTBENCH_TO_CANONICAL,
}


def resolve_mapping(name_or_path: str | Path | None) -> tuple[str, dict[str, str | None]]:
    """Resolve a mapping by builtin name or JSON path.

    Returns ``(name, mapping)`` so the resolved identity can be recorded in the
    evaluation report -- which mapping was used is part of the result.
    """
    if name_or_path is None:
        return "identity", dict(IDENTITY_MAPPING)
    key = str(name_or_path)
    if key in LABEL_MAPPINGS:
        return key, dict(LABEL_MAPPINGS[key])
    path = Path(key)
    if not path.exists():
        raise ValueError(
            f"Unknown label mapping {key!r}. Use one of {sorted(LABEL_MAPPINGS)} or a path to a JSON file."
        )
    return path.as_posix(), load_label_mapping(path)


def load_label_mapping(path: Path) -> dict[str, str | None]:
    """Load a ``{raw_label: canonical_label_or_null}`` JSON mapping and validate it."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Label mapping must be a JSON object, got {type(data).__name__}")
    mapping: dict[str, str | None] = {}
    for raw, target in data.items():
        if target is None:
            mapping[str(raw)] = None
            continue
        target = str(target)
        if not is_canonical(target):
            raise UnknownLabelError(
                f"Mapping {raw!r} -> {target!r} targets a non-canonical label."
            )
        mapping[str(raw)] = target
    return mapping


def map_label(raw_label: str, mapping: dict[str, str | None]) -> str | None:
    """Map a raw detector label into the canonical space.

    Lookup is case-insensitive on the raw side because the DocLayout adapter
    lowercases model class names before they reach here.
    """
    if raw_label in mapping:
        return mapping[raw_label]
    lowered = raw_label.lower()
    if lowered in mapping:
        return mapping[lowered]
    return None


# ---------------------------------------------------------------------------
# keep / skip partition, for the operational metrics
# ---------------------------------------------------------------------------


def keep_labels(policies: dict[str, LabelPolicy] | None = None) -> frozenset[str]:
    """Canonical labels whose regions are sent to extraction (i.e. cost money).

    Derived from the region policies so there is a single source of truth: a
    label is "keep" unless its action is SKIP. ``table`` is a keep label even
    though its action is TABLE rather than OCR.
    """
    effective = policies or DEFAULT_LABEL_POLICIES
    keep = set()
    for label in CANONICAL_LABELS:
        policy = effective.get(label)
        if policy is None or policy.action != ExtractionAction.SKIP:
            keep.add(label)
    return frozenset(keep)


def skip_labels(policies: dict[str, LabelPolicy] | None = None) -> frozenset[str]:
    return frozenset(CANONICAL_LABELS) - keep_labels(policies)


def catalogue_fingerprint() -> str:
    """Short digest of the catalogue, recorded in reports to detect drift."""
    import hashlib

    joined = "|".join(f"{index}:{label}" for index, label in enumerate(CANONICAL_LABELS))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
