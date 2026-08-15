import {
  Braces,
  Download,
  Eraser,
  FileText,
  Loader2,
  Play,
  ScanLine,
  X,
} from "lucide-react";
import { useMemo, useState } from "react";

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
  const [drawing, setDrawing] = useState(null);
  const [error, setError] = useState("");
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
        annotations: annotations
          .filter((item) => item.page_number === page.page_number)
          .map((item) => ({ label: item.label, bbox: item.bbox })),
      })),
    };
  }, [annotations, result]);

  function resetRun() {
    setResult(null);
    setAnnotations([]);
    setHiddenLabels(new Set());
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

  function handlePointerDown(event, page) {
    if (!drawMode) return;
    const start = getSvgPoint(event.currentTarget, event);
    event.currentTarget.setPointerCapture(event.pointerId);
    setDrawing({
      pageNumber: page.page_number,
      start,
      current: start,
    });
  }

  function handlePointerMove(event) {
    if (!drawing) return;
    // Read the point before handing it to the updater: React clears
    // `currentTarget` once the handler returns, and the updater runs later.
    const point = getSvgPoint(event.currentTarget, event);
    setDrawing((current) => {
      if (!current) return current;
      return {
        ...current,
        current: point,
      };
    });
  }

  function handlePointerUp(event, page) {
    if (!drawing || drawing.pageNumber !== page.page_number) return;
    const end = getSvgPoint(event.currentTarget, event);
    const bbox = clampBox(normalizeBox(drawing.start, end), page.width, page.height);
    setDrawing(null);

    if (bbox.width < 8 || bbox.height < 8) return;
    setAnnotations((items) => [
      ...items,
      {
        page_number: page.page_number,
        label: selectedLabel,
        bbox,
      },
    ]);
  }

  function removeAnnotation(index) {
    setAnnotations((items) => items.filter((_, position) => position !== index));
  }

  function exportAnnotations() {
    if (!annotationPayload) return;
    downloadJson("annotations.json", annotationPayload);
  }

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
                  {annotations.map((annotation, index) => (
                    <li
                      className="flex items-center gap-2 bg-surface px-2.5 py-1.5"
                      key={`${annotation.page_number}-${index}`}
                    >
                      <span
                        className="swatch"
                        style={{ backgroundColor: COLORS[annotation.label] || COLORS.manual }}
                      />
                      <span className="truncate font-mono text-[11px]">{annotation.label}</span>
                      <span className="ml-auto font-mono text-[10px] tabular-nums text-faint">
                        p{annotation.page_number} · {Math.round(annotation.bbox.width)}×
                        {Math.round(annotation.bbox.height)}
                      </span>
                      <button
                        aria-label={`Delete ${annotation.label} box on page ${annotation.page_number}`}
                        className="rounded p-0.5 text-faint transition hover:bg-raised hover:text-danger"
                        type="button"
                        onClick={() => removeAnnotation(index)}
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
                  onClick={exportAnnotations}
                  disabled={!annotationPayload}
                >
                  <Download className="h-3.5 w-3.5" />
                  Export
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => setAnnotations([])}
                  disabled={annotations.length === 0}
                >
                  <Eraser className="h-3.5 w-3.5" />
                  Clear all
                </button>
              </div>
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
                  drawing={drawing}
                  hiddenLabels={hiddenLabels}
                  key={page.page_number}
                  onPointerDown={handlePointerDown}
                  onPointerMove={handlePointerMove}
                  onPointerUp={handlePointerUp}
                  page={page}
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
  drawing,
  hiddenLabels,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  page,
  zoom,
}) {
  const regions = page.regions.filter((region) => !hiddenLabels.has(region.label));
  const pageAnnotations = annotations.filter((item) => item.page_number === page.page_number);
  const displayWidth = getDisplayWidth(page, zoom);
  const previewBox =
    drawing?.pageNumber === page.page_number ? normalizeBox(drawing.start, drawing.current) : null;

  return (
    <article className="panel overflow-hidden">
      <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
        <h2 className="font-mono text-[12px] font-semibold">Page {page.page_number}</h2>
        <span className="font-mono text-[11px] tabular-nums text-faint">
          {page.width} × {page.height} px · {regions.length} regions
        </span>
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
              {pageAnnotations.map((annotation, index) => (
                <RegionBox
                  bbox={annotation.bbox}
                  key={`${annotation.page_number}-${index}`}
                  label={annotation.label}
                  manual
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

function RegionBox({ bbox, confidence = null, label, manual = false }) {
  const color = COLORS[label] || COLORS.manual;
  return (
    <g>
      <rect
        fill={manual ? "rgba(17, 24, 39, 0.07)" : "transparent"}
        height={bbox.height}
        stroke={color}
        strokeDasharray={manual ? "10 8" : undefined}
        strokeWidth={manual ? 4 : 3}
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
        {manual ? `${label} · manual` : `${label}${confidence === null ? "" : ` ${Number(confidence).toFixed(2)}`}`}
      </text>
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

function getSvgPoint(svg, event) {
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  return point.matrixTransform(svg.getScreenCTM().inverse());
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

function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
