import {
  Braces,
  Download,
  Eraser,
  FileText,
  Loader2,
  Play,
  ScanLine,
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

const DEFAULT_DOCLAYOUT_MODEL = "models/doclayout_yolo_docstructbench_imgsz1024.pt";

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
  const [runOcr, setRunOcr] = useState(true);
  const [drawMode, setDrawMode] = useState(true);
  const [selectedLabel, setSelectedLabel] = useState("target_data");
  const [zoom, setZoom] = useState("fit");
  const [status, setStatus] = useState({ label: "Ready", tone: "idle" });
  const [result, setResult] = useState(null);
  const [annotations, setAnnotations] = useState([]);
  const [drawing, setDrawing] = useState(null);
  const [error, setError] = useState("");
  const isParsing = status.tone === "busy";

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

  async function handleParse(event) {
    event.preventDefault();
    if (!file) return;

    setError("");
    setStatus({ label: "Parsing", tone: "busy" });

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("detector", detector);
      formData.append("device", "cpu");
      formData.append("run_ocr", runOcr ? "true" : "false");
      if (maxPages.trim()) formData.append("max_pages", maxPages.trim());
      if (modelPath.trim()) formData.append("model_path", modelPath.trim());

      const response = await fetch("/documents/parse", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      setResult(await response.json());
      setAnnotations([]);
      setStatus({ label: "Done", tone: "done" });
    } catch (caught) {
      setError(caught.message || "Parse failed");
      setStatus({ label: "Error", tone: "error" });
    }
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

  function exportAnnotations() {
    if (!annotationPayload) return;
    downloadJson("annotations.json", annotationPayload);
  }

  return (
    <div className="min-h-screen bg-slate-100 text-ink">
      <header className="flex flex-col gap-3 border-b border-slate-200 bg-white px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold leading-tight">DocsOCR Layout Parser</h1>
          <p className="mt-1 text-sm text-slate-500">Annotation-driven PDF parsing prototype</p>
        </div>
        <StatusBadge status={status} />
      </header>

      <main className="grid gap-4 p-4 lg:grid-cols-[340px_minmax(0,1fr)]">
        <aside className="grid content-start gap-3">
          <Panel title="Parse">
            <form className="grid gap-3" onSubmit={handleParse}>
              <Field label="PDF file">
                <input
                  className="input"
                  type="file"
                  accept="application/pdf"
                  onChange={(event) => {
                    setFile(event.target.files?.[0] || null);
                    setResult(null);
                    setAnnotations([]);
                    setError("");
                    setStatus({ label: "Ready", tone: "idle" });
                  }}
                  required
                />
              </Field>

              <div className="grid grid-cols-[minmax(0,1fr)_96px] gap-2">
                <Field label="Detector">
                  <select className="input" value={detector} onChange={(event) => setDetector(event.target.value)}>
                    <option value="heuristic">heuristic</option>
                    <option value="doclayout-yolo">doclayout-yolo</option>
                  </select>
                </Field>

                <Field label="Pages">
                  <input
                    className="input"
                    min="1"
                    max="300"
                    type="number"
                    value={maxPages}
                    onChange={(event) => setMaxPages(event.target.value)}
                  />
                </Field>
              </div>

              <Field label="Model path">
                <input
                  className="input"
                  value={modelPath}
                  placeholder={DEFAULT_DOCLAYOUT_MODEL}
                  onChange={(event) => setModelPath(event.target.value)}
                />
              </Field>
              <p className="-mt-2 text-xs leading-5 text-slate-500">Local default: <code>models/doclayout_yolo_docstructbench_imgsz1024.pt</code></p>

              <div className="grid grid-cols-2 gap-2">
                <Toggle checked={runOcr} label="OCR" onChange={setRunOcr} />
                <Toggle checked={drawMode} label="Draw" onChange={setDrawMode} />
              </div>

              <button className="primary-button" type="submit" disabled={status.tone === "busy"}>
                {status.tone === "busy" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                Run parse
              </button>
            </form>
          </Panel>

          <Panel title="Annotation">
            <div className="grid gap-3">
              <Field label="Label">
                <select className="input" value={selectedLabel} onChange={(event) => setSelectedLabel(event.target.value)}>
                  {LABELS.map((label) => (
                    <option key={label} value={label}>
                      {label}
                    </option>
                  ))}
                </select>
              </Field>

              <div className="grid grid-cols-2 gap-2">
                <button className="secondary-button" type="button" onClick={exportAnnotations} disabled={!annotationPayload}>
                  <Download className="h-4 w-4" />
                  Export
                </button>
                <button className="secondary-button" type="button" onClick={() => setAnnotations([])}>
                  <Eraser className="h-4 w-4" />
                  Clear
                </button>
              </div>
              <p className="text-xs text-slate-500">{annotations.length} annotations</p>
            </div>
          </Panel>

          <Panel title="View">
            <div className="grid grid-cols-3 gap-1 rounded-lg border border-slate-200 bg-slate-100 p-1">
              {ZOOM_OPTIONS.map((option) => (
                <button
                  key={option.value}
                  className={`h-8 rounded-md text-xs font-bold ${
                    zoom === option.value ? "bg-white text-blue-600 shadow-sm" : "text-slate-500"
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
              <div className="grid gap-2 text-sm">
                <a className="output-link" href={result.result_json_url} target="_blank" rel="noreferrer">
                  <Braces className="h-4 w-4" />
                  result.json
                </a>
                <a className="output-link" href={result.result_markdown_url} target="_blank" rel="noreferrer">
                  <FileText className="h-4 w-4" />
                  result.md
                </a>
              </div>
            ) : (
              <p className="text-xs leading-5 text-slate-500">No output yet</p>
            )}
            {error ? <p className="mt-3 rounded-md bg-rose-50 p-2 text-xs text-rose-700">{error}</p> : null}
          </Panel>
        </aside>

        <section className="min-w-0">
          {isParsing ? (
            <div className="grid min-h-[360px] place-content-center rounded-lg border border-slate-200 bg-white p-6 text-center shadow-panel">
              <Loader2 className="mx-auto mb-3 h-8 w-8 animate-spin text-blue-600" />
              <strong className="text-sm">Parsing PDF layout...</strong>
              <p className="mt-2 text-xs text-slate-500">{file?.name || "Selected PDF"}</p>
            </div>
          ) : !result ? (
            <div className="grid min-h-[360px] place-content-center rounded-lg border border-dashed border-slate-300 bg-white p-6 text-center shadow-panel">
              <ScanLine className="mx-auto mb-3 h-8 w-8 text-slate-400" />
              <strong className="text-sm">{file ? "PDF is ready to parse." : "Upload a PDF to inspect layout regions."}</strong>
            </div>
          ) : (
            <div className="grid gap-4">
              {result.pages.map((page) => (
                <PageViewer
                  annotations={annotations}
                  drawing={drawing}
                  key={page.page_number}
                  onPointerDown={handlePointerDown}
                  onPointerMove={handlePointerMove}
                  onPointerUp={handlePointerUp}
                  page={page}
                  zoom={zoom}
                />
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

function PageViewer({ annotations, drawing, onPointerDown, onPointerMove, onPointerUp, page, zoom }) {
  const pageAnnotations = annotations.filter((item) => item.page_number === page.page_number);
  const displayWidth = getDisplayWidth(page, zoom);
  const previewBox =
    drawing?.pageNumber === page.page_number ? normalizeBox(drawing.start, drawing.current) : null;

  return (
    <article className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-panel">
      <div className="flex items-center justify-between gap-3 border-b border-slate-200 px-4 py-3">
        <h2 className="text-sm font-bold">Page {page.page_number}</h2>
        <span className="text-xs text-slate-500">
          {page.width} x {page.height}
        </span>
      </div>

      <div className="grid lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="max-h-[72vh] overflow-auto border-b border-slate-200 bg-slate-200 p-4 text-center lg:border-b-0 lg:border-r">
          <div
            className="relative inline-block min-w-[260px] bg-white shadow-xl"
            style={{ width: `min(100%, ${displayWidth}px)` }}
          >
            <img className="block h-auto w-full select-none" src={page.image_url} alt={`Page ${page.page_number}`} />
            <svg
              className="absolute inset-0 h-full w-full touch-none"
              viewBox={`0 0 ${page.width} ${page.height}`}
              onPointerDown={(event) => onPointerDown(event, page)}
              onPointerMove={onPointerMove}
              onPointerUp={(event) => onPointerUp(event, page)}
            >
              {page.regions.map((region) => (
                <RegionBox key={region.region_id} bbox={region.bbox} confidence={region.confidence} label={region.label} />
              ))}
              {pageAnnotations.map((annotation, index) => (
                <RegionBox bbox={annotation.bbox} key={`${annotation.page_number}-${index}`} label={annotation.label} manual />
              ))}
              {previewBox ? <RegionBox bbox={previewBox} label="manual" manual /> : null}
            </svg>
          </div>
        </div>

        <div className="max-h-[72vh] overflow-auto bg-slate-50 p-3">
          <div className="grid gap-2">
            {page.regions.map((region) => (
              <RegionListItem key={region.region_id} region={region} />
            ))}
          </div>
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
      <text fill={color} fontSize="18" fontWeight="700" x={bbox.x + 6} y={Math.max(18, bbox.y + 18)}>
        {manual ? `${label} manual` : `${label}${confidence === null ? "" : ` ${Number(confidence).toFixed(2)}`}`}
      </text>
    </g>
  );
}

function RegionListItem({ region }) {
  const color = COLORS[region.label] || COLORS.manual;
  return (
    <div className="rounded-md border border-slate-200 bg-white p-3">
      <div className="mb-2 flex items-center justify-between gap-2 text-xs font-bold">
        <span className="rounded-full px-2 py-1 text-white" style={{ backgroundColor: color }}>
          {region.label}
        </span>
        <span className="text-slate-500">
          {region.action} {Number(region.confidence).toFixed(2)}
        </span>
      </div>
      <pre className="max-h-24 overflow-auto whitespace-pre-wrap text-xs leading-5 text-slate-500">
        {region.text || region.notes.join("\n") || "No extracted text"}
      </pre>
    </div>
  );
}

function Panel({ children, title }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-panel">
      <h2 className="mb-3 text-sm font-bold">{title}</h2>
      {children}
    </section>
  );
}

function Field({ children, label }) {
  return (
    <label className="grid gap-1.5 text-xs font-bold text-slate-500">
      <span>{label}</span>
      {children}
    </label>
  );
}

function Toggle({ checked, label, onChange }) {
  return (
    <label className="flex h-9 items-center gap-2 rounded-md border border-slate-200 px-3 text-sm font-semibold">
      <input checked={checked} type="checkbox" onChange={(event) => onChange(event.target.checked)} />
      {label}
    </label>
  );
}

function StatusBadge({ status }) {
  const tone = {
    idle: "border-slate-200 bg-slate-50 text-slate-500",
    busy: "border-amber-200 bg-amber-50 text-amber-700",
    done: "border-emerald-200 bg-emerald-50 text-emerald-700",
    error: "border-rose-200 bg-rose-50 text-rose-700",
  }[status.tone || "idle"];

  return <div className={`rounded-full border px-3 py-1.5 text-xs font-bold ${tone}`}>{status.label}</div>;
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

