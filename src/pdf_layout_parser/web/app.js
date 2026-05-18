const parseForm = document.querySelector("#parseForm");
const statusBadge = document.querySelector("#statusBadge");
const pagesEl = document.querySelector("#pages");
const emptyState = document.querySelector("#emptyState");
const pageTemplate = document.querySelector("#pageTemplate");
const outputLinks = document.querySelector("#outputLinks");
const annotationCount = document.querySelector("#annotationCount");
const exportAnnotationsBtn = document.querySelector("#exportAnnotations");
const clearAnnotationsBtn = document.querySelector("#clearAnnotations");
const zoomButtons = document.querySelectorAll(".zoom-button");

const state = {
  result: null,
  annotations: [],
  drawing: null,
  zoom: "fit",
};

const colors = {
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
  manual: "#111827",
};

parseForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = document.querySelector("#pdfFile").files[0];
  if (!file) return;

  setStatus("Parsing", "busy");
  setFormDisabled(true);

  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("detector", document.querySelector("#detector").value);
    formData.append("device", "cpu");
    formData.append("run_ocr", document.querySelector("#runOcr").checked ? "true" : "false");

    const maxPages = document.querySelector("#maxPages").value.trim();
    const modelPath = document.querySelector("#modelPath").value.trim();
    if (maxPages) formData.append("max_pages", maxPages);
    if (modelPath) formData.append("model_path", modelPath);

    const response = await fetch("/documents/parse", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    state.result = await response.json();
    state.annotations = [];
    renderResult();
    setStatus("Done", "done");
  } catch (error) {
    console.error(error);
    setStatus("Error", "error");
    outputLinks.textContent = error.message || "파싱 중 오류가 발생했습니다.";
  } finally {
    setFormDisabled(false);
  }
});

exportAnnotationsBtn.addEventListener("click", () => {
  if (!state.result) return;
  const labels = Array.from(new Set([
    "title",
    "body",
    "table",
    "figure",
    "header",
    "footer",
    "ignore",
    "target_data",
    ...state.annotations.map((annotation) => annotation.label),
  ]));

  const pages = state.result.pages.map((page) => ({
    image_path: page.image_path,
    width: page.width,
    height: page.height,
    annotations: state.annotations
      .filter((annotation) => annotation.page_number === page.page_number)
      .map((annotation) => ({
        label: annotation.label,
        bbox: annotation.bbox,
      })),
  }));

  downloadJson("annotations.json", { labels, pages });
});

clearAnnotationsBtn.addEventListener("click", () => {
  state.annotations = [];
  renderResult();
});

zoomButtons.forEach((button) => {
  button.addEventListener("click", () => {
    state.zoom = button.dataset.zoom;
    zoomButtons.forEach((item) => item.classList.toggle("active", item === button));
    applyZoom();
  });
});

function renderResult() {
  pagesEl.replaceChildren();
  emptyState.style.display = state.result ? "none" : "grid";
  updateOutputLinks();
  updateAnnotationCount();

  if (!state.result) return;

  for (const page of state.result.pages) {
    const pageNode = pageTemplate.content.cloneNode(true);
    const article = pageNode.querySelector(".page-card");
    const heading = pageNode.querySelector("h3");
    const meta = pageNode.querySelector(".page-meta");
    const pageCanvas = pageNode.querySelector(".page-canvas");
    const image = pageNode.querySelector("img");
    const overlay = pageNode.querySelector(".overlay");
    const regionList = pageNode.querySelector(".region-list");

    article.dataset.pageNumber = page.page_number;
    heading.textContent = `Page ${page.page_number}`;
    meta.textContent = `${page.width} x ${page.height}`;
    image.src = page.image_url;
    image.alt = `Page ${page.page_number}`;
    overlay.setAttribute("viewBox", `0 0 ${page.width} ${page.height}`);
    overlay.dataset.pageNumber = page.page_number;
    pageCanvas.style.setProperty("--page-display-width", `${displayWidthForPage(page)}px`);

    renderOverlay(overlay, page);
    bindAnnotationDrawing(overlay, page);
    renderRegionList(regionList, page);

    pagesEl.appendChild(pageNode);
  }
}

function applyZoom() {
  if (!state.result) return;
  document.querySelectorAll(".page-card").forEach((article) => {
    const pageNumber = Number(article.dataset.pageNumber);
    const page = state.result.pages.find((item) => item.page_number === pageNumber);
    if (!page) return;
    article
      .querySelector(".page-canvas")
      .style.setProperty("--page-display-width", `${displayWidthForPage(page)}px`);
  });
}

function displayWidthForPage(page) {
  if (state.zoom === "fit") {
    return Math.min(page.width, 680);
  }
  return Math.max(320, page.width * Number(state.zoom || 1));
}

function renderOverlay(svg, page) {
  svg.replaceChildren();

  for (const region of page.regions) {
    svg.appendChild(createBox(region.bbox, region.label, region.confidence, false));
  }

  const pageAnnotations = state.annotations.filter(
    (annotation) => annotation.page_number === page.page_number,
  );
  for (const annotation of pageAnnotations) {
    svg.appendChild(createBox(annotation.bbox, annotation.label, null, true));
  }
}

function createBox(bbox, label, confidence, manual) {
  const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const color = colors[label] || colors.manual;

  const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  rect.setAttribute("x", bbox.x);
  rect.setAttribute("y", bbox.y);
  rect.setAttribute("width", bbox.width);
  rect.setAttribute("height", bbox.height);
  rect.setAttribute("fill", manual ? "rgba(17, 24, 39, 0.07)" : "transparent");
  rect.setAttribute("stroke", color);
  rect.setAttribute("stroke-width", manual ? "4" : "3");
  if (manual) rect.setAttribute("stroke-dasharray", "10 8");

  const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
  text.setAttribute("x", bbox.x + 6);
  text.setAttribute("y", Math.max(18, bbox.y + 18));
  text.setAttribute("font-size", "18");
  text.setAttribute("font-weight", "700");
  text.setAttribute("fill", color);
  text.textContent = manual
    ? `${label} manual`
    : `${label}${confidence === null ? "" : ` ${Number(confidence).toFixed(2)}`}`;

  group.append(rect, text);
  return group;
}

function bindAnnotationDrawing(svg, page) {
  svg.addEventListener("pointerdown", (event) => {
    if (!document.querySelector("#drawMode").checked) return;
    const start = svgPoint(svg, event);
    state.drawing = {
      svg,
      page,
      start,
      preview: createBox({ x: start.x, y: start.y, width: 1, height: 1 }, "manual", null, true),
    };
    svg.appendChild(state.drawing.preview);
    svg.setPointerCapture(event.pointerId);
  });

  svg.addEventListener("pointermove", (event) => {
    if (!state.drawing || state.drawing.svg !== svg) return;
    const current = svgPoint(svg, event);
    const bbox = normalizeBox(state.drawing.start, current);
    state.drawing.preview.replaceWith(createBox(bbox, "manual", null, true));
    state.drawing.preview = svg.lastElementChild;
  });

  svg.addEventListener("pointerup", (event) => {
    if (!state.drawing || state.drawing.svg !== svg) return;
    const end = svgPoint(svg, event);
    const bbox = normalizeBox(state.drawing.start, end);
    state.drawing.preview.remove();
    state.drawing = null;

    if (bbox.width < 8 || bbox.height < 8) {
      renderResult();
      return;
    }

    state.annotations.push({
      page_number: page.page_number,
      label: document.querySelector("#labelSelect").value,
      bbox: clampBox(bbox, page.width, page.height),
    });
    renderResult();
  });
}

function renderRegionList(container, page) {
  container.replaceChildren();
  if (!page.regions.length) {
    container.textContent = "탐지된 영역이 없습니다.";
    container.classList.add("empty-region");
    return;
  }

  for (const region of page.regions) {
    const item = document.createElement("div");
    item.className = "region-item";

    const title = document.createElement("div");
    title.className = "region-title";

    const label = document.createElement("span");
    label.className = "region-pill";
    label.style.background = colors[region.label] || colors.manual;
    label.textContent = region.label;

    const action = document.createElement("span");
    action.textContent = `${region.action} ${Number(region.confidence).toFixed(2)}`;

    const text = document.createElement("div");
    text.className = "region-text";
    text.textContent = region.text || region.notes.join("\n") || "No extracted text";

    title.append(label, action);
    item.append(title, text);
    container.appendChild(item);
  }
}

function updateOutputLinks() {
  outputLinks.replaceChildren();
  if (!state.result) {
    outputLinks.textContent = "파싱 후 결과 링크가 표시됩니다.";
    outputLinks.classList.add("muted");
    return;
  }

  const jsonLink = document.createElement("a");
  jsonLink.href = state.result.result_json_url;
  jsonLink.target = "_blank";
  jsonLink.textContent = "result.json";

  const mdLink = document.createElement("a");
  mdLink.href = state.result.result_markdown_url;
  mdLink.target = "_blank";
  mdLink.textContent = "result.md";

  outputLinks.append(jsonLink, mdLink);
}

function updateAnnotationCount() {
  annotationCount.textContent = `${state.annotations.length} annotations`;
}

function setStatus(text, mode) {
  statusBadge.textContent = text;
  statusBadge.className = `status-badge ${mode || ""}`.trim();
}

function setFormDisabled(disabled) {
  parseForm.querySelectorAll("button, input, select").forEach((element) => {
    element.disabled = disabled;
  });
}

function svgPoint(svg, event) {
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
