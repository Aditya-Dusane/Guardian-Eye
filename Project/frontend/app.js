/**
 * app.js — Guardian Eye Anomaly Detection Frontend
 * Connects to the FastAPI backend at http://localhost:8000
 */

const API_BASE = "http://localhost:8000";

// ── DOM refs ─────────────────────────────────────────────────
const dropZone       = document.getElementById("drop-zone");
const fileInput      = document.getElementById("file-input");
const filePreview    = document.getElementById("file-preview");
const fileName       = document.getElementById("file-name");
const fileSize       = document.getElementById("file-size");
const clearFileBtn   = document.getElementById("clear-file");
const analyzeBtn     = document.getElementById("analyze-btn");
const btnText        = analyzeBtn.querySelector(".btn-text");
const btnSpinner     = analyzeBtn.querySelector(".btn-spinner");

const serverBadge    = document.getElementById("server-badge");
const badgeText      = serverBadge.querySelector(".badge-text");
const thresholdVal   = document.getElementById("threshold-val");
const deviceVal      = document.getElementById("device-val");

// Results panels
const resultsIdle    = document.getElementById("results-idle");
const resultsLoading = document.getElementById("results-loading");
const resultsOutput  = document.getElementById("results-output");
const resultsError   = document.getElementById("results-error");
const errorMessage   = document.getElementById("error-message");
const retryBtn       = document.getElementById("retry-btn");

// Result elements
const statusBadge    = document.getElementById("status-badge");
const statusLabel    = document.getElementById("status-label");
const gaugeFill      = document.getElementById("gauge-fill");
const gaugeScore     = document.getElementById("gauge-score");
const metricScore    = document.getElementById("metric-score");
const metricMax      = document.getElementById("metric-max");
const metricThresh   = document.getElementById("metric-threshold");
const metricClips    = document.getElementById("metric-clips");
const confidencePct  = document.getElementById("confidence-pct");
const confidenceBar  = document.getElementById("confidence-bar");
const resultFilename = document.getElementById("result-filename");
const resultFrames   = document.getElementById("result-frames");

// ── State ─────────────────────────────────────────────────────
let selectedFile = null;

// ── Server status check ────────────────────────────────────────
async function checkServerStatus() {
  try {
    const res = await fetch(`${API_BASE}/model-status`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    serverBadge.className = "server-badge connected";
    badgeText.textContent = "Backend Ready";

    thresholdVal.textContent = data.threshold ?? "—";
    deviceVal.textContent    = data.device    ?? "—";
  } catch (e) {
    serverBadge.className = "server-badge error";
    badgeText.textContent = "Backend Offline";
    console.warn("[Guardian Eye] Backend not reachable:", e.message);
  }
}

checkServerStatus();
setInterval(checkServerStatus, 15000); // Re-check every 15s

// ── File handling ──────────────────────────────────────────────
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function setFile(file) {
  if (!file) return;
  const allowed = [".mp4", ".avi", ".mov", ".mkv"];
  const ext = "." + file.name.split(".").pop().toLowerCase();
  if (!allowed.includes(ext)) {
    showError(`Unsupported file type "${ext}". Please upload an MP4, AVI, MOV, or MKV file.`);
    return;
  }
  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  filePreview.classList.remove("hidden");
  analyzeBtn.disabled = false;
  showPanel("idle");
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  filePreview.classList.add("hidden");
  analyzeBtn.disabled = true;
  showPanel("idle");
}

// Drop zone events
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });

dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

clearFileBtn.addEventListener("click", clearFile);
retryBtn.addEventListener("click", () => { showPanel("idle"); });

// ── Panel helpers ──────────────────────────────────────────────
function showPanel(name) {
  resultsIdle.classList.add("hidden");
  resultsLoading.classList.add("hidden");
  resultsOutput.classList.add("hidden");
  resultsError.classList.add("hidden");

  if (name === "idle")    resultsIdle.classList.remove("hidden");
  if (name === "loading") resultsLoading.classList.remove("hidden");
  if (name === "output")  resultsOutput.classList.remove("hidden");
  if (name === "error")   resultsError.classList.remove("hidden");
}

function showError(msg) {
  errorMessage.textContent = msg;
  showPanel("error");
}

function setAnalyzing(loading) {
  if (loading) {
    btnText.textContent = "Analyzing…";
    btnSpinner.classList.remove("hidden");
    analyzeBtn.disabled = true;
  } else {
    btnText.textContent = "Analyze Video";
    btnSpinner.classList.add("hidden");
    analyzeBtn.disabled = !selectedFile;
  }
}

// ── Gauge rendering ────────────────────────────────────────────
/**
 * Arc path total length for the half-circle gauge:
 *   r=50, arc from (10,65) to (110,65) → semi-circle → length = π*50 ≈ 157
 */
const GAUGE_FULL = 157;

function renderGauge(score, threshold, isAnomaly) {
  // Normalise score to 0-1 relative to 2x threshold as max
  const maxRef = threshold * 2 || 1;
  const ratio = Math.min(1, score / maxRef);
  const filled = ratio * GAUGE_FULL;

  gaugeFill.setAttribute("stroke-dasharray", `${filled.toFixed(2)} ${(GAUGE_FULL - filled).toFixed(2)}`);
  gaugeFill.classList.toggle("anomaly-fill", isAnomaly);
  gaugeScore.textContent = score.toFixed(5);
}

// ── Inference ──────────────────────────────────────────────────
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  setAnalyzing(true);
  showPanel("loading");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || `Server error (${res.status})`);
      setAnalyzing(false);
      return;
    }

    renderResults(data);
    showPanel("output");

  } catch (e) {
    console.error("[Guardian Eye] Predict failed:", e);
    showError("Could not reach the backend. Make sure the server is running at " + API_BASE);
    showPanel("error");
  } finally {
    setAnalyzing(false);
  }
});

function renderResults(data) {
  const isAnomaly = data.is_anomaly;

  // Status badge
  statusBadge.className = `status-badge ${isAnomaly ? "anomaly" : "normal"}`;
  statusLabel.textContent = isAnomaly ? "⚠ Anomaly Detected" : "✓ Normal";

  // Gauge
  renderGauge(data.anomaly_score, data.threshold, isAnomaly);

  // Metrics
  metricScore.textContent  = data.anomaly_score.toFixed(5);
  metricMax.textContent    = data.max_clip_score.toFixed(5);
  metricThresh.textContent = data.threshold.toFixed(5);
  metricClips.textContent  = data.clip_count;

  // Confidence bar
  const conf = Math.min(100, data.confidence);
  confidencePct.textContent = `${conf.toFixed(1)}%`;
  confidenceBar.style.width = `${conf}%`;
  confidenceBar.classList.toggle("anomaly-bar", isAnomaly);

  // File info
  resultFilename.textContent = data.filename || "—";
  resultFrames.textContent   = data.frame_count;

  // Update server info in case it changed
  thresholdVal.textContent = data.threshold;
}
