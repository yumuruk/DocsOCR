import {
  Braces,
  Download,
  Eraser,
  FileText,
  FlaskConical,
  Loader2,
  Play,
  ScanLine,
  Upload,
  Wand2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

const LABELS = [
  "title",
  "body",
  "table",
  "figure",
  "header",
  "footer",
  "ignore",
  "target_data",
];

// Plate colours. These are the only saturated colours in the UI; everything
// else is achromatic so that colour always means "this is a detected region".
// Drawn over the white page image, so they need no dark-theme variant.
const COLORS = {
  title: "#2563eb",
  body: "#047857",
  "plain text": "#047857",
  text: "#047857",
  table: "#b45309",
  figure: "#6d28d9",
  image: "#6d28d9",
  header: "#0e7490",
  footer: "#64748b",
  ignore: "#be123c",
  target_data: "#db2777",
  // DocStructBench class names emitted by the pretrained checkpoint. Captions use
  // a lighter shade of the class they caption; skipped furniture recedes.
  abandon: "#94a3b8",
  figure_caption: "#8b5cf6",
  table_caption: "#d97706",
  table_footnote: "#f59e0b",
  isolate_formula: "#0d9488",
  formula_caption: "#2dd4bf",
  manual: "#111827",
};

const ACTION_LABELS = {
  ocr: "extract",
  native_text: "native",
  table: "table",
  skip: "skip",
};

const DEFAULT_DOCLAYOUT_MODEL = "models/doclayout_yolo_docstructbench_imgsz1024.pt";

const DEVICES = ["cpu", "mps", "cuda:0"];

const ZOOM_OPTIONS = [
  { label: "Fit", value: "fit" },
  { label: "75%", value: "0.75" },
  { label: "100%", value: "1" },
];

export default function App() {
  const [file, setFile] = useState(null);
  const [detector, setDetector] = useState("doclayout-yolo");
  const [maxPages, setMaxPages] = useState("3");
  const [modelPath, setModelPath] = useState(DEFAULT_DOCLAYOUT_MODEL);
  const [device, setDevice] = useState("cpu");
  const [runOcr, setRunOcr] = useState(true);
  const [drawMode, setDrawMode] = useState(true);
  const [selectedLabel, setSelectedLabel] = useState("target_data");
  const [zoom, setZoom] = useState("fit");
  const [status, setStatus] = useState({ label: "Ready", tone: "idle" });
  const [result, setResult] = useState(null);
  const [elapsed, setElapsed] = useState(null);
  const [annotations, setAnnotations] = useState([]);
  const [hiddenLabels, setHiddenLabels] = useState(() => new Set());
  // One pointer gesture at a time: drawing a new box, moving a box, or resizing
  // by a corner handle. Selection is separate state so a box stays selected
  // (relabel, keyboard delete) after the gesture ends.
  const [gesture, setGesture] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  // Pages the annotator has marked finished. Exported as label_status so the
  // dataset side can tell "labelled and empty" from "not labelled yet" (C5).
  const [donePages, setDonePages] = useState(() => new Set());
  const [annotationNote, setAnnotationNote] = useState("");
  const [error, setError] = useState("");
  // Web quick-loop training: one job at a time, polled until it settles.
  const [finetune, setFinetune] = useState(null);
  const [epochs, setEpochs] = useState("20");
  const [trainImageSize, setTrainImageSize] = useState("1024");
  const nextAnnotationId = useRef(1);
  const importInputRef = useRef(null);
  const isParsing = status.tone === "busy";
  const usesModel = detector === "doclayout-yolo";

  const summary = useMemo(() => (result ? summarize(result) : null), [result]);

  const annotationPayload = useMemo(() => {
    if (!result) return null;
    const labels = Array.from(new Set([...LABELS, ...annotations.map((item) => item.label)]));
    return {
      labels,
      pages: result.pages.map((page) => ({
        image_path: page.image_path,
        width: page.width,
        height: page.height,
        page_number: page.page_number,
        // "in_progress" until the annotator marks the page done. The exporter
        // and evaluator skip non-complete pages, so an unfinished page can never
        // masquerade as an empty one.
        label_status: donePages.has(page.page_number) ? "complete" : "in_progress",
        annotations: annotations
          .filter((item) => item.page_number === page.page_number)
          .map((item) => ({ label: item.label, bbox: item.bbox, source: item.source })),
      })),
    };
  }, [annotations, donePages, result]);

  function resetRun() {
    setResult(null);
    setAnnotations([]);
    setHiddenLabels(new Set());
    setGesture(null);
    setSelectedId(null);
    setDonePages(new Set());
    setAnnotationNote("");
    setFinetune(null);
    setElapsed(null);
    setError("");
  }

  async function handleParse(event) {
    event.preventDefault();
    if (!file) return;

    setError("");
    setStatus({ label: "Parsing", tone: "busy" });
    const startedAt = performance.now();

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("detector", detector);
      formData.append("device", device);
      formData.append("run_ocr", runOcr ? "true" : "false");
      if (maxPages.trim()) formData.append("max_pages", maxPages.trim());
      if (usesModel && modelPath.trim()) formData.append("model_path", modelPath.trim());

      const response = await fetch("/documents/parse", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      setResult(await response.json());
      setAnnotations([]);
      setHiddenLabels(new Set());
      setGesture(null);
      setSelectedId(null);
      setDonePages(new Set());
      setAnnotationNote("");
      setFinetune(null);
      setElapsed((performance.now() - startedAt) / 1000);
      setStatus({ label: "Done", tone: "done" });
    } catch (caught) {
      setError(readableError(caught));
      setStatus({ label: "Failed", tone: "error" });
    }
  }

  function toggleLabel(label) {
    setHiddenLabels((current) => {
      const next = new Set(current);
      if (next.has(label)) next.delete(label);
      else next.add(label);
      return next;
    });
  }

  // A prediction that a human touched is no longer a prediction. The
  // manual/prediction/correction distinction is an experiment axis downstream,
  // so the transition happens at the moment of edit, not at export.
  function editedSource(source) {
    return source === "prediction" ? "correction" : source || "manual";
  }

  function handlePointerDown(event, page) {
    const point = getSvgPoint(event.currentTarget, event);
    // No transform yet means no usable origin, so never begin a gesture that
    // would have to guess where it started.
    if (!point) return;
    const pageAnnotations = annotations.filter((item) => item.page_number === page.page_number);

    // Corner handles of the selected box take priority over everything under them.
    const selected = pageAnnotations.find((item) => item.id === selectedId);
    if (selected) {
      const corner = hitHandle(point, selected.bbox);
      if (corner) {
        event.currentTarget.setPointerCapture(event.pointerId);
        setGesture({
          type: "resize",
          id: selected.id,
          pageNumber: page.page_number,
          pageWidth: page.width,
          pageHeight: page.height,
          anchor: oppositeCorner(selected.bbox, corner),
          origin: { ...selected.bbox },
        });
        return;
      }
    }

    // Topmost box under the pointer: last drawn wins, matching paint order.
    const hit = [...pageAnnotations].reverse().find((item) => containsPoint(item.bbox, point));
    if (hit) {
      setSelectedId(hit.id);
      event.currentTarget.setPointerCapture(event.pointerId);
      setGesture({
        type: "move",
        id: hit.id,
        pageNumber: page.page_number,
        pageWidth: page.width,
        pageHeight: page.height,
        start: point,
        origin: { ...hit.bbox },
      });
      return;
    }

    setSelectedId(null);
    if (!drawMode) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    setGesture({
      type: "draw",
      pageNumber: page.page_number,
      start: point,
      current: point,
    });
  }

  function handlePointerMove(event) {
    if (!gesture) return;
    // Read the point before handing it to the updater: React clears
    // `currentTarget` once the handler returns, and the updater runs later.
    const point = getSvgPoint(event.currentTarget, event);
    // Hold the last good position rather than dropping the gesture; the next
    // move event recovers it.
    if (!point) return;

    if (gesture.type === "draw") {
      setGesture((current) => (current ? { ...current, current: point } : current));
      return;
    }

    // Move and resize apply live so the box tracks the pointer.
    const { id, pageWidth, pageHeight } = gesture;
    setAnnotations((items) =>
      items.map((item) => {
        if (item.id !== id) return item;
        const bbox =
          gesture.type === "move"
            ? clampShift(gesture.origin, point.x - gesture.start.x, point.y - gesture.start.y, pageWidth, pageHeight)
            : clampBox(normalizeBox(gesture.anchor, point), pageWidth, pageHeight);
        return { ...item, bbox };
      })
    );
  }

  function handlePointerUp(event, page) {
    if (!gesture || gesture.pageNumber !== page.page_number) return;
    const active = gesture;
    // Clearing the gesture is unconditional: returning early would leave it
    // attached to a pointer that no longer exists.
    setGesture(null);

    if (active.type === "draw") {
      // Fall back to the last position the move handler managed to map; `start`
      // and `current` are real points for as long as the gesture exists.
      const end = getSvgPoint(event.currentTarget, event) ?? active.current;
      const bbox = clampBox(normalizeBox(active.start, end), page.width, page.height);
      if (bbox.width < 8 || bbox.height < 8) return;
      const id = nextAnnotationId.current++;
      setAnnotations((items) => [
        ...items,
        { id, page_number: page.page_number, label: selectedLabel, bbox, source: "manual" },
      ]);
      setSelectedId(id);
      return;
    }

    // Move/resize already applied live; finalise. A resize collapsed below the
    // minimum reverts to where it started rather than leaving a sliver.
    setAnnotations((items) =>
      items.map((item) => {
        if (item.id !== active.id) return item;
        const bbox =
          active.type === "resize" && (item.bbox.width < 8 || item.bbox.height < 8)
            ? active.origin
            : item.bbox;
        const moved =
          bbox.x !== active.origin.x ||
          bbox.y !== active.origin.y ||
          bbox.width !== active.origin.width ||
          bbox.height !== active.origin.height;
        return { ...item, bbox, source: moved ? editedSource(item.source) : item.source };
      })
    );
  }

  function removeAnnotation(id) {
    setAnnotations((items) => items.filter((item) => item.id !== id));
    setSelectedId((current) => (current === id ? null : current));
  }

  function relabelAnnotation(id, label) {
    setAnnotations((items) =>
      items.map((item) =>
        item.id === id ? { ...item, label, source: editedSource(item.source) } : item
      )
    );
  }

  function togglePageDone(pageNumber) {
    setDonePages((current) => {
      const next = new Set(current);
      if (next.has(pageNumber)) next.delete(pageNumber);
      else next.add(pageNumber);
      return next;
    });
  }

  function loadPredictions() {
    if (!result) return;
    let id = nextAnnotationId.current;
    const loaded = result.pages.flatMap((page) =>
      page.regions.map((region) => ({
        id: id++,
        page_number: page.page_number,
        label: canonicalizePrediction(region.label, region.bbox, page.height),
        bbox: {
          x: region.bbox.x,
          y: region.bbox.y,
          width: region.bbox.width,
          height: region.bbox.height,
        },
        source: "prediction",
      }))
    );
    nextAnnotationId.current = id;
    // Replace rather than append: the HITL scenario is "start from the model's
    // output and fix it", not "mix it into what I already drew".
    setAnnotations(loaded);
    setSelectedId(null);
    setAnnotationNote(`Loaded ${loaded.length} prediction(s) — edits become corrections`);
  }

  async function importAnnotations(file) {
    if (!file || !result) return;
    try {
      const data = JSON.parse(await file.text());
      const pagesByName = new Map(
        result.pages.map((page) => [baseName(page.image_path), page])
      );
      let id = nextAnnotationId.current;
      const loaded = [];
      const done = new Set();
      let skippedPages = 0;
      for (const page of data.pages || []) {
        const match = pagesByName.get(baseName(page.image_path || ""));
        // Coordinates only mean anything on the exact render they were drawn
        // over, so a page that is missing or was rendered at another size is
        // skipped, not guessed at.
        if (!match || match.width !== page.width || match.height !== page.height) {
          skippedPages += 1;
          continue;
        }
        if (page.label_status === "complete") done.add(match.page_number);
        for (const annotation of page.annotations || []) {
          const bbox = annotation.bbox || {};
          loaded.push({
            id: id++,
            page_number: match.page_number,
            label: annotation.label,
            bbox: { x: bbox.x, y: bbox.y, width: bbox.width, height: bbox.height },
            source: annotation.source || "manual",
          });
        }
      }
      nextAnnotationId.current = id;
      setAnnotations(loaded);
      setDonePages(done);
      setSelectedId(null);
      setAnnotationNote(
        `Imported ${loaded.length} box(es)` +
          (skippedPages ? ` · ${skippedPages} page(s) skipped (not in this run or size mismatch)` : "")
      );
    } catch (caught) {
      setAnnotationNote(`Import failed: ${readableError(caught)}`);
    }
  }

  function exportAnnotations() {
    if (!annotationPayload) return;
    downloadJson("annotations.json", annotationPayload);
  }

  const isTraining = finetune?.status === "queued" || finetune?.status === "running";

  async function startFinetune() {
    if (!annotationPayload || isTraining) return;
    setFinetune({ status: "queued", phase: "submitting" });
    try {
      const response = await fetch("/training/fine-tune", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          annotations: annotationPayload,
          epochs: Number(epochs) || 20,
          image_size: Number(trainImageSize) || 1024,
          device,
          // The current model path doubles as the fine-tune starting point; the
          // server falls back to the bundled DocStructBench checkpoint.
          pretrain: usesModel && modelPath.trim() ? modelPath.trim() : null,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Fine-tune request failed");
      setFinetune({ jobId: data.job_id, status: "queued", phase: "queued", startedAt: performance.now() });
    } catch (caught) {
      setFinetune({ status: "failed", error: readableError(caught) });
    }
  }

  // Poll the job until it settles. The interval restarts whenever the job id or
  // its status changes and tears down on unmount.
  useEffect(() => {
    if (!finetune?.jobId || (finetune.status !== "queued" && finetune.status !== "running")) return;
    const timer = setInterval(async () => {
      try {
        const response = await fetch(`/training/jobs/${finetune.jobId}`);
        if (!response.ok) return;
        const snapshot = await response.json();
        setFinetune((current) =>
          current && current.jobId === snapshot.job_id
            ? { ...current, status: snapshot.status, phase: snapshot.phase, result: snapshot.result, error: snapshot.error }
            : current
        );
      } catch {
        // transient network error: keep polling
      }
    }, 2500);
    return () => clearInterval(timer);
  }, [finetune?.jobId, finetune?.status]);

  function useTunedModel() {
    const path = finetune?.result?.best_model_relpath || finetune?.result?.best_model_path;
    if (!path) return;
    setDetector("doclayout-yolo");
    setModelPath(path);
  }

  // Keyboard editing: Escape deselects, Delete removes the selected box.
  // Ignored while typing in a form control.
  useEffect(() => {
    function onKeyDown(event) {
      if (/^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName)) return;
      if (event.key === "Escape") setSelectedId(null);
      if ((event.key === "Delete" || event.key === "Backspace") && selectedId !== null) {
        event.preventDefault();
        removeAnnotation(selectedId);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selectedId]);

  return (
    <div className="min-h-screen bg-paper text-ink">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line bg-surface px-5 py-3">
        <div className="flex items-baseline gap-3">
          <h1 className="text-[15px] font-semibold tracking-tight">DocsOCR</h1>
          <p className="eyebrow">Document layout proofing</p>
        </div>
        <StatusBadge status={status} />
      </header>

      <main className="grid items-start gap-4 p-4 lg:grid-cols-[320px_minmax(0,1fr)]">
        <aside className="grid content-start gap-3">
          <Panel title="Source">
            <form className="grid gap-3" onSubmit={handleParse}>
              <Field label="PDF file">
                <input
                  className="block w-full cursor-pointer rounded-md border border-line bg-raised px-2.5 py-2 font-mono text-[12px] text-muted transition hover:border-line-strong file:mr-2.5 file:cursor-pointer file:rounded file:border-0 file:bg-action file:px-2 file:py-1 file:font-sans file:text-[12px] file:font-semibold file:text-action-ink"
                  type="file"
                  accept="application/pdf"
                  onChange={(event) => {
                    setFile(event.target.files?.[0] || null);
                    resetRun();
                    setStatus({ label: "Ready", tone: "idle" });
                  }}
                  required
                />
              </Field>

              <div className="grid grid-cols-[minmax(0,1fr)_84px] gap-2">
                <Field label="Detector">
                  <select
                    className="input"
                    value={detector}
                    onChange={(event) => setDetector(event.target.value)}
                  >
                    <option value="heuristic">heuristic</option>
                    <option value="doclayout-yolo">doclayout-yolo</option>
                  </select>
                </Field>

                <Field label="Pages">
                  <input
                    className="input tabular-nums"
                    min="1"
                    max="300"
                    type="number"
                    value={maxPages}
                    onChange={(event) => setMaxPages(event.target.value)}
                  />
                </Field>
              </div>

              <Field label="Model path" hint={usesModel ? null : "heuristic needs no model"}>
                <input
                  className="input"
                  value={modelPath}
                  disabled={!usesModel}
                  placeholder={DEFAULT_DOCLAYOUT_MODEL}
                  onChange={(event) => setModelPath(event.target.value)}
                />
              </Field>

              <div className="grid grid-cols-[92px_minmax(0,1fr)] gap-2">
                <Field label="Device">
                  <select
                    className="input"
                    value={device}
                    onChange={(event) => setDevice(event.target.value)}
                  >
                    {DEVICES.map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label="Text extraction">
                  <Toggle checked={runOcr} label="Run OCR on text regions" onChange={setRunOcr} />
                </Field>
              </div>

              <button className="primary-button" type="submit" disabled={isParsing}>
                {isParsing ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
                Run parse
              </button>
            </form>
          </Panel>

          <Panel title="Annotate">
            <div className="grid gap-3">
              <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-2">
                <Field label="Label">
                  <select
                    className="input"
                    value={selectedLabel}
                    onChange={(event) => setSelectedLabel(event.target.value)}
                  >
                    {LABELS.map((label) => (
                      <option key={label} value={label}>
                        {label}
                      </option>
                    ))}
                  </select>
                </Field>
                <Toggle checked={drawMode} label="Draw" onChange={setDrawMode} />
              </div>

              {annotations.length > 0 ? (
                <ul className="grid max-h-52 gap-px overflow-auto rounded-md border border-line bg-line">
                  {annotations.map((annotation) => (
                    <li
                      className={`flex cursor-pointer items-center gap-2 px-2.5 py-1.5 ${
                        annotation.id === selectedId ? "bg-raised" : "bg-surface"
                      }`}
                      key={annotation.id}
                      onClick={() => setSelectedId(annotation.id)}
                    >
                      <span
                        className="swatch"
                        style={{ backgroundColor: COLORS[annotation.label] || COLORS.manual }}
                      />
                      <select
                        aria-label={`Label for box on page ${annotation.page_number}`}
                        className="w-24 rounded border border-transparent bg-transparent font-mono text-[11px] text-ink transition hover:border-line"
                        value={annotation.label}
                        onChange={(event) => relabelAnnotation(annotation.id, event.target.value)}
                        onClick={(event) => event.stopPropagation()}
                      >
                        {Array.from(new Set([...LABELS, annotation.label])).map((label) => (
                          <option key={label} value={label}>
                            {label}
                          </option>
                        ))}
                      </select>
                      <span className="ml-auto font-mono text-[10px] tabular-nums text-faint">
                        {annotation.source === "manual" ? "" : `${annotation.source} · `}p
                        {annotation.page_number} · {Math.round(annotation.bbox.width)}×
                        {Math.round(annotation.bbox.height)}
                      </span>
                      <button
                        aria-label={`Delete ${annotation.label} box on page ${annotation.page_number}`}
                        className="rounded p-0.5 text-faint transition hover:bg-raised hover:text-danger"
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          removeAnnotation(annotation.id);
                        }}
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="rounded-md border border-dashed border-line px-2.5 py-3 text-center font-mono text-[11px] text-faint">
                  {result ? "Drag on a page to add a box" : "No boxes yet"}
                </p>
              )}

              <div className="grid grid-cols-2 gap-2">
                <button
                  className="secondary-button"
                  type="button"
                  onClick={loadPredictions}
                  disabled={!result}
                  title="Replace boxes with the model's detections, then correct them"
                >
                  <Wand2 className="h-3.5 w-3.5" />
                  Predictions
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => importInputRef.current?.click()}
                  disabled={!result}
                  title="Load a previously exported annotations.json"
                >
                  <Upload className="h-3.5 w-3.5" />
                  Import
                </button>
                <input
                  ref={importInputRef}
                  className="hidden"
                  type="file"
                  accept="application/json"
                  onChange={(event) => {
                    importAnnotations(event.target.files?.[0]);
                    event.target.value = "";
                  }}
                />
                <button
                  className="secondary-button"
                  type="button"
                  onClick={exportAnnotations}
                  disabled={!annotationPayload}
                >
                  <Download className="h-3.5 w-3.5" />
                  Export
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => {
                    setAnnotations([]);
                    setSelectedId(null);
                  }}
                  disabled={annotations.length === 0}
                >
                  <Eraser className="h-3.5 w-3.5" />
                  Clear all
                </button>
              </div>

              {annotationNote ? (
                <p className="font-mono text-[10px] leading-4 text-faint">{annotationNote}</p>
              ) : (
                <p className="font-mono text-[10px] leading-4 text-faint">
                  Click a box to select · drag to move · corners resize · Del removes
                </p>
              )}
            </div>
          </Panel>

          <Panel title="Fine-tune">
            <div className="grid gap-3">
              <div className="grid grid-cols-2 gap-2">
                <Field label="Epochs">
                  <input
                    className="input tabular-nums"
                    min="1"
                    max="300"
                    type="number"
                    value={epochs}
                    disabled={isTraining}
                    onChange={(event) => setEpochs(event.target.value)}
                  />
                </Field>
                <Field label="Train img size">
                  <select
                    className="input"
                    value={trainImageSize}
                    disabled={isTraining}
                    onChange={(event) => setTrainImageSize(event.target.value)}
                  >
                    <option value="640">640 (fast)</option>
                    <option value="1024">1024</option>
                  </select>
                </Field>
              </div>

              <button
                className="primary-button"
                type="button"
                onClick={startFinetune}
                disabled={!result || donePages.size < 3 || annotations.length === 0 || isTraining}
                title={
                  donePages.size < 3
                    ? "Mark at least 3 pages as Labeled first — they are split into train/val/test"
                    : "Train on the Labeled pages, then score before vs after on held-out pages"
                }
              >
                {isTraining ? <Loader2 className="h-4 w-4 animate-spin" /> : <FlaskConical className="h-4 w-4" />}
                {isTraining ? `Training — ${finetune.phase}` : "Fine-tune & compare"}
              </button>

              {donePages.size < 3 && result ? (
                <p className="font-mono text-[10px] leading-4 text-faint">
                  {donePages.size}/3 pages marked Labeled — the job splits them into train/val/test,
                  trains on train, and scores both models on the held-out test pages.
                </p>
              ) : null}

              {finetune?.status === "failed" ? (
                <div className="rounded-md border border-danger bg-danger-bg px-2.5 py-2">
                  <p className="eyebrow text-danger">Fine-tune failed</p>
                  <p className="mt-1 max-h-24 overflow-auto whitespace-pre-wrap break-words font-mono text-[10px] leading-4 text-danger">
                    {finetune.error || "Unknown error"}
                  </p>
                </div>
              ) : null}

              {finetune?.status === "completed" && finetune.result ? (
                <FinetuneOutcome result={finetune.result} onUseTuned={useTunedModel} />
              ) : null}
            </div>
          </Panel>

          <Panel title="View">
            <div className="grid grid-cols-3 gap-1 rounded-md border border-line bg-raised p-1">
              {ZOOM_OPTIONS.map((option) => (
                <button
                  key={option.value}
                  className={`h-7 rounded font-mono text-[11px] transition ${
                    zoom === option.value
                      ? "bg-surface font-semibold text-ink shadow-sm"
                      : "text-muted hover:text-ink"
                  }`}
                  type="button"
                  onClick={() => setZoom(option.value)}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </Panel>

          <Panel title="Output">
            {result ? (
              <div className="grid gap-2">
                <a className="output-link" href={result.result_json_url} target="_blank" rel="noreferrer">
                  <Braces className="h-3.5 w-3.5 text-faint" />
                  result.json
                </a>
                <a className="output-link" href={result.result_markdown_url} target="_blank" rel="noreferrer">
                  <FileText className="h-3.5 w-3.5 text-faint" />
                  result.md
                </a>
              </div>
            ) : (
              <p className="font-mono text-[11px] text-faint">Nothing parsed yet</p>
            )}
            {error ? (
              <div className="mt-3 rounded-md border border-danger bg-danger-bg px-2.5 py-2">
                <p className="eyebrow text-danger">Parse failed</p>
                <p className="mt-1 break-words font-mono text-[11px] leading-5 text-danger">{error}</p>
              </div>
            ) : null}
          </Panel>
        </aside>

        <section className="grid min-w-0 gap-4">
          {isParsing ? (
            <div className="panel grid min-h-[380px] place-content-center p-6 text-center">
              <Loader2 className="mx-auto mb-3 h-6 w-6 animate-spin text-muted" />
              <strong className="text-sm font-semibold">Detecting layout regions</strong>
              <p className="mt-1.5 font-mono text-[11px] text-faint">{file?.name || "Selected PDF"}</p>
            </div>
          ) : !result ? (
            <div className="panel grid min-h-[380px] place-content-center border-dashed p-6 text-center">
              <ScanLine className="mx-auto mb-3 h-6 w-6 text-faint" />
              <strong className="text-sm font-semibold">
                {file ? "Ready to parse" : "Choose a PDF to inspect its layout"}
              </strong>
              <p className="mt-1.5 max-w-sm font-mono text-[11px] leading-5 text-faint">
                Each detected region is drawn as a coloured plate over the page, then either
                extracted or skipped by its label policy.
              </p>
            </div>
          ) : (
            <>
              <SummaryStrip
                elapsed={elapsed}
                hiddenLabels={hiddenLabels}
                onToggleLabel={toggleLabel}
                summary={summary}
              />
              {result.pages.map((page) => (
                <PageProof
                  annotations={annotations}
                  done={donePages.has(page.page_number)}
                  gesture={gesture}
                  hiddenLabels={hiddenLabels}
                  key={page.page_number}
                  onPointerDown={handlePointerDown}
                  onPointerMove={handlePointerMove}
                  onPointerUp={handlePointerUp}
                  onToggleDone={togglePageDone}
                  page={page}
                  selectedId={selectedId}
                  zoom={zoom}
                />
              ))}
            </>
          )}
        </section>
      </main>
    </div>
  );
}

function SummaryStrip({ elapsed, hiddenLabels, onToggleLabel, summary }) {
  return (
    <div className="panel sticky top-0 z-10 bg-surface/95 backdrop-blur">
      <div className="flex flex-wrap gap-x-8 gap-y-4 px-4 py-3">
        <Metric label="Pages" value={summary.pages} />
        <Metric label="Regions" value={summary.regions} />
        <Metric label="Extracted" value={summary.extracted} />
        <Metric label="Skipped" value={summary.skipped} />
        <Metric label="Characters" value={summary.characters.toLocaleString()} />
        <Metric label="Elapsed" value={elapsed ? `${elapsed.toFixed(1)}s` : "—"} />
      </div>
      <div className="border-t border-line px-4 py-3">
        <p className="eyebrow mb-2">Plates · click to hide</p>
        <div className="flex flex-wrap gap-1.5">
          {summary.legend.map((entry) => {
            const hidden = hiddenLabels.has(entry.label);
            return (
              <button
                aria-pressed={!hidden}
                className={`inline-flex items-center gap-1.5 rounded border px-2 py-1 font-mono text-[11px] transition ${
                  hidden
                    ? "border-line text-faint line-through"
                    : "border-line-strong text-ink hover:bg-raised"
                }`}
                key={entry.label}
                type="button"
                onClick={() => onToggleLabel(entry.label)}
              >
                <span className="swatch" style={{ backgroundColor: entry.color }} />
                {entry.label}
                <span className="tabular-nums text-faint">{entry.count}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function FinetuneOutcome({ onUseTuned, result }) {
  const pct = (value) => (value === null || value === undefined ? "—" : `${(value * 100).toFixed(1)}%`);
  const score = (value) => (value === null || value === undefined ? "—" : value.toFixed(3));
  const rows = result.summary || [];
  const testPages = result.split?.test_pages;
  const baseline = rows.find((row) => row.variant === "baseline");
  const tuned = rows.find((row) => row.variant === "tuned");
  const delta =
    baseline?.map50 != null && tuned?.map50 != null ? tuned.map50 - baseline.map50 : null;

  return (
    <div className="grid gap-2">
      <p className="eyebrow">
        Before vs after · {testPages ?? "?"} held-out page{testPages === 1 ? "" : "s"}
      </p>
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full font-mono text-[11px]">
          <thead>
            <tr className="border-b border-line bg-raised text-left text-faint">
              <th className="px-2 py-1.5 font-medium">model</th>
              <th className="px-2 py-1.5 font-medium">mAP@.5</th>
              <th className="px-2 py-1.5 font-medium" title="OCR area saved vs full-page">save</th>
              <th className="px-2 py-1.5 font-medium" title="ground-truth content discarded">loss</th>
              <th className="px-2 py-1.5 font-medium" title="extracted area that is not content">noise</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr className="border-b border-line last:border-b-0" key={row.variant}>
                <td className="px-2 py-1.5">{row.variant}</td>
                <td className="px-2 py-1.5 tabular-nums">{score(row.map50)}</td>
                <td className="px-2 py-1.5 tabular-nums">{pct(row.reduction_rate)}</td>
                <td className="px-2 py-1.5 tabular-nums">{pct(row.loss_rate)}</td>
                <td className="px-2 py-1.5 tabular-nums">{pct(row.contamination_rate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {delta !== null ? (
        <p className="font-mono text-[11px] text-muted">
          mAP@.5 {delta >= 0 ? "+" : ""}
          {delta.toFixed(3)} after fine-tuning
        </p>
      ) : null}
      <button className="secondary-button" type="button" onClick={onUseTuned}>
        <Play className="h-3.5 w-3.5" />
        Use tuned model
      </button>
      <p className="font-mono text-[10px] leading-4 text-faint">
        Held-out pages come from the same document (page-level split); numbers are not comparable
        with document-level CLI experiments. Set Model path &amp; Run parse again to see the tuned boxes.
      </p>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="grid gap-1.5">
      <span className="eyebrow">{label}</span>
      <span className="metric">{value}</span>
    </div>
  );
}

function PageProof({
  annotations,
  done,
  gesture,
  hiddenLabels,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onToggleDone,
  page,
  selectedId,
  zoom,
}) {
  const regions = page.regions.filter((region) => !hiddenLabels.has(region.label));
  const pageAnnotations = annotations.filter((item) => item.page_number === page.page_number);
  const displayWidth = getDisplayWidth(page, zoom);
  const previewBox =
    gesture?.type === "draw" && gesture.pageNumber === page.page_number
      ? normalizeBox(gesture.start, gesture.current)
      : null;

  return (
    <article className="panel overflow-hidden">
      <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
        <h2 className="font-mono text-[12px] font-semibold">Page {page.page_number}</h2>
        <span className="ml-auto font-mono text-[11px] tabular-nums text-faint">
          {page.width} × {page.height} px · {regions.length} regions
        </span>
        <label
          className={`flex cursor-pointer items-center gap-1.5 rounded border px-2 py-1 font-mono text-[11px] transition ${
            done ? "border-transparent bg-action text-action-ink" : "border-line text-muted hover:border-line-strong"
          }`}
          title="Exported as label_status: complete — unfinished pages are excluded from training and evaluation"
        >
          <input
            checked={done}
            className="h-3 w-3 accent-[rgb(var(--action))]"
            type="checkbox"
            onChange={() => onToggleDone(page.page_number)}
          />
          Labeled
        </label>
      </div>

      <div className="grid lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="max-h-[72vh] overflow-auto border-b border-line bg-paper p-5 text-center lg:border-b-0 lg:border-r">
          <div
            className="relative inline-block min-w-[260px] bg-white shadow-proof"
            style={{ width: `min(100%, ${displayWidth}px)` }}
          >
            <img
              className="block h-auto w-full select-none"
              src={page.image_url}
              alt={`Page ${page.page_number}`}
            />
            <svg
              className="absolute inset-0 h-full w-full touch-none"
              viewBox={`0 0 ${page.width} ${page.height}`}
              onPointerDown={(event) => onPointerDown(event, page)}
              onPointerMove={onPointerMove}
              onPointerUp={(event) => onPointerUp(event, page)}
            >
              {regions.map((region) => (
                <RegionBox
                  key={region.region_id}
                  bbox={region.bbox}
                  confidence={region.confidence}
                  label={region.label}
                />
              ))}
              {pageAnnotations.map((annotation) => (
                <RegionBox
                  bbox={annotation.bbox}
                  key={annotation.id}
                  label={annotation.label}
                  manual
                  selected={annotation.id === selectedId}
                  source={annotation.source}
                />
              ))}
              {previewBox ? <RegionBox bbox={previewBox} label="manual" manual /> : null}
            </svg>
          </div>
        </div>

        <div className="max-h-[72vh] overflow-auto bg-surface">
          {regions.length === 0 ? (
            <p className="px-4 py-6 text-center font-mono text-[11px] text-faint">
              All plates hidden
            </p>
          ) : (
            regions.map((region) => <RegionRow key={region.region_id} region={region} />)
          )}
        </div>
      </div>
    </article>
  );
}

function RegionBox({ bbox, confidence = null, label, manual = false, selected = false, source = "manual" }) {
  const color = COLORS[label] || COLORS.manual;
  return (
    <g>
      <rect
        fill={manual ? "rgba(17, 24, 39, 0.07)" : "transparent"}
        height={bbox.height}
        stroke={color}
        // Selection reads as "solid and handled"; unselected editable boxes stay
        // dashed so they are visually distinct from model plates.
        strokeDasharray={manual && !selected ? "10 8" : undefined}
        strokeWidth={manual ? (selected ? 5 : 4) : 3}
        width={bbox.width}
        x={bbox.x}
        y={bbox.y}
      />
      <text
        fill={color}
        fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
        fontSize="17"
        fontWeight="600"
        x={bbox.x + 6}
        y={Math.max(17, bbox.y + 17)}
      >
        {manual ? `${label} · ${source}` : `${label}${confidence === null ? "" : ` ${Number(confidence).toFixed(2)}`}`}
      </text>
      {selected
        ? cornerPoints(bbox).map((corner) => (
            <rect
              key={corner.name}
              x={corner.x - HANDLE_SIZE / 2}
              y={corner.y - HANDLE_SIZE / 2}
              width={HANDLE_SIZE}
              height={HANDLE_SIZE}
              fill="white"
              stroke={color}
              strokeWidth={3}
            />
          ))
        : null}
    </g>
  );
}

function RegionRow({ region }) {
  const color = COLORS[region.label] || COLORS.manual;
  const skipped = region.action === "skip";
  return (
    <div className={`border-b border-line px-3.5 py-2.5 last:border-b-0 ${skipped ? "bg-raised" : ""}`}>
      <div className="flex items-center gap-2">
        <span className="swatch" style={{ backgroundColor: color }} />
        <span className={`font-mono text-[12px] ${skipped ? "text-muted" : "text-ink"}`}>
          {region.label}
        </span>
        <span className="ml-auto flex items-center gap-2">
          <ActionChip action={region.action} />
          <span className="font-mono text-[11px] tabular-nums text-faint">
            {Number(region.confidence).toFixed(2)}
          </span>
        </span>
      </div>
      {skipped ? (
        <p className="mt-1.5 font-mono text-[11px] text-faint">
          {region.notes[0] || "Skipped by region policy"}
        </p>
      ) : (
        <pre className="mt-2 max-h-28 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-5 text-muted">
          {region.text || region.notes.join("\n") || "No text extracted"}
        </pre>
      )}
    </div>
  );
}

function ActionChip({ action }) {
  const skipped = action === "skip";
  return (
    <span
      className={`rounded px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.08em] ${
        skipped
          ? "border border-dashed border-line-strong text-faint"
          : "bg-raised text-muted ring-1 ring-inset ring-line"
      }`}
    >
      {ACTION_LABELS[action] || action}
    </span>
  );
}

function Panel({ children, title }) {
  return (
    <section className="panel p-3.5">
      <h2 className="eyebrow mb-3">{title}</h2>
      {children}
    </section>
  );
}

function Field({ children, hint, label }) {
  return (
    <label className="grid gap-1.5">
      <span className="eyebrow">{label}</span>
      {children}
      {hint ? <span className="font-mono text-[10px] text-faint">{hint}</span> : null}
    </label>
  );
}

function Toggle({ checked, label, onChange }) {
  return (
    <span
      className={`flex h-9 cursor-pointer items-center gap-2 rounded-md border px-2.5 transition ${
        checked ? "border-line-strong bg-raised" : "border-line"
      }`}
      onClick={() => onChange(!checked)}
    >
      <input
        checked={checked}
        className="h-3.5 w-3.5 accent-[rgb(var(--action))]"
        type="checkbox"
        onChange={(event) => onChange(event.target.checked)}
        onClick={(event) => event.stopPropagation()}
      />
      <span className="truncate font-mono text-[11px] text-muted">{label}</span>
    </span>
  );
}

function StatusBadge({ status }) {
  const tone = {
    idle: "border-line text-faint",
    busy: "border-line-strong text-ink",
    done: "border-transparent bg-action text-action-ink",
    error: "border-danger text-danger",
  }[status.tone || "idle"];

  return (
    <div
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[11px] ${tone}`}
    >
      {status.tone === "busy" ? <Loader2 className="h-3 w-3 animate-spin" /> : null}
      {status.label}
    </div>
  );
}

function summarize(result) {
  const counts = new Map();
  let regions = 0;
  let extracted = 0;
  let characters = 0;

  for (const page of result.pages) {
    for (const region of page.regions) {
      regions += 1;
      if (region.action !== "skip") extracted += 1;
      characters += (region.text || "").length;
      counts.set(region.label, (counts.get(region.label) || 0) + 1);
    }
  }

  const legend = [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([label, count]) => ({ label, count, color: COLORS[label] || COLORS.manual }));

  return {
    pages: result.pages.length,
    regions,
    extracted,
    skipped: regions - extracted,
    characters,
    legend,
  };
}

function readableError(caught) {
  const raw = caught?.message || "Parse failed";
  try {
    const parsed = JSON.parse(raw);
    if (parsed?.detail) return typeof parsed.detail === "string" ? parsed.detail : JSON.stringify(parsed.detail);
  } catch {
    // not JSON; fall through to the raw text
  }
  return raw.length > 400 ? `${raw.slice(0, 400)}…` : raw;
}

function getDisplayWidth(page, zoom) {
  if (zoom === "fit") return Math.min(page.width, 680);
  return Math.max(320, page.width * Number(zoom || 1));
}

// DocStructBench class names -> the canonical labels the annotator corrects in.
// Mirrors labels.DOCSTRUCTBENCH_TO_CANONICAL on the Python side; without this,
// every preloaded box arrives as "plain text"/"table_caption"/... and the
// annotator renames dozens of boxes per page by hand.
const PREDICTION_LABEL_MAP = {
  "plain text": "body",
  figure_caption: "body",
  table_caption: "body",
  table_footnote: "body",
  formula_caption: "body",
  isolate_formula: "figure",
};

// `abandon` is DocStructBench's one bucket for all discardable furniture. The
// evaluation mapping folds it into `ignore` (a mapping table cannot split one
// label into three), but here a HUMAN verifies every box afterwards, so a
// position heuristic may pre-sort it: top band -> header, bottom band ->
// footer. Wrong guesses cost one relabel click; right ones cost nothing.
function canonicalizePrediction(label, bbox, pageHeight) {
  if (label === "abandon") {
    const centerY = bbox.y + bbox.height / 2;
    if (centerY < pageHeight * 0.2) return "header";
    if (centerY > pageHeight * 0.8) return "footer";
    return "ignore";
  }
  return PREDICTION_LABEL_MAP[label] || label;
}

// Returns null when the pointer cannot be mapped into page coordinates.
// getScreenCTM() has no matrix to give while the SVG is unrendered -- not yet laid
// out, or inside a collapsed container -- and returns null, so the transform has
// to be treated as unavailable rather than assumed. Callers must handle null; a
// dropped point is recoverable, a thrown one kills the drag.
function getSvgPoint(svg, event) {
  if (!svg) return null;
  const matrix = svg.getScreenCTM();
  if (!matrix) return null;
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  return point.matrixTransform(matrix.inverse());
}

function normalizeBox(start, end) {
  const x = Math.min(start.x, end.x);
  const y = Math.min(start.y, end.y);
  return {
    x,
    y,
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y),
  };
}

function clampBox(bbox, width, height) {
  const x = Math.max(0, Math.min(bbox.x, width));
  const y = Math.max(0, Math.min(bbox.y, height));
  const x2 = Math.max(0, Math.min(bbox.x + bbox.width, width));
  const y2 = Math.max(0, Math.min(bbox.y + bbox.height, height));
  return {
    x: Number(x.toFixed(2)),
    y: Number(y.toFixed(2)),
    width: Number(Math.max(0, x2 - x).toFixed(2)),
    height: Number(Math.max(0, y2 - y).toFixed(2)),
  };
}

// Moving must never change a box's size: instead of clamping the corners (which
// crops), the translation itself is limited so the whole box stays on the page.
function clampShift(origin, dx, dy, width, height) {
  const x = Math.max(0, Math.min(origin.x + dx, width - origin.width));
  const y = Math.max(0, Math.min(origin.y + dy, height - origin.height));
  return {
    x: Number(x.toFixed(2)),
    y: Number(y.toFixed(2)),
    width: origin.width,
    height: origin.height,
  };
}

function containsPoint(bbox, point) {
  return (
    point.x >= bbox.x &&
    point.x <= bbox.x + bbox.width &&
    point.y >= bbox.y &&
    point.y <= bbox.y + bbox.height
  );
}

// Rendered size of a corner handle, in page pixels (dpi 180 renders, so ~14px
// is comfortably visible without swallowing small boxes).
const HANDLE_SIZE = 14;
// Grab radius around a handle centre. Larger than the visual so corners are
// forgiving to hit at any zoom.
const HANDLE_HIT = 16;

function cornerPoints(bbox) {
  return [
    { name: "nw", x: bbox.x, y: bbox.y },
    { name: "ne", x: bbox.x + bbox.width, y: bbox.y },
    { name: "sw", x: bbox.x, y: bbox.y + bbox.height },
    { name: "se", x: bbox.x + bbox.width, y: bbox.y + bbox.height },
  ];
}

function hitHandle(point, bbox) {
  for (const corner of cornerPoints(bbox)) {
    if (Math.abs(point.x - corner.x) <= HANDLE_HIT && Math.abs(point.y - corner.y) <= HANDLE_HIT) {
      return corner.name;
    }
  }
  return null;
}

// Resizing drags one corner while the opposite corner stays anchored.
function oppositeCorner(bbox, corner) {
  const x = corner.includes("w") ? bbox.x + bbox.width : bbox.x;
  const y = corner.includes("n") ? bbox.y + bbox.height : bbox.y;
  return { x, y };
}

function baseName(path) {
  const parts = String(path).split(/[\\/]/);
  return parts[parts.length - 1] || "";
}

function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
