const form = document.querySelector("#scan-form");
const inputPath = document.querySelector("#input-path");
const workspace = document.querySelector("#workspace");
const submitButton = document.querySelector("#submit-button");
const errorBox = document.querySelector("#error-box");
const emptyState = document.querySelector("#empty-state");
const jobState = document.querySelector("#job-state");
const statusChip = document.querySelector("#status-chip");
const jobTitle = document.querySelector("#job-title");
const jobMessage = document.querySelector("#job-message");
const jobTime = document.querySelector("#job-time");
const progressBar = document.querySelector("#progress-bar");
const sourceSummary = document.querySelector("#source-summary");
const workspaceSummary = document.querySelector("#workspace-summary");
const results = document.querySelector("#results");
const filesBody = document.querySelector("#files-body");
const fileFilter = document.querySelector("#file-filter");
const tableNote = document.querySelector("#table-note");
const newScanButton = document.querySelector("#new-scan-button");
const serverState = document.querySelector("#server-state");
const recentJobs = document.querySelector("#recent-jobs");
const recentJobsList = document.querySelector("#recent-jobs-list");
const refreshJobs = document.querySelector("#refresh-jobs");

let currentJob = null;
let currentAssets = [];
let startedAt = null;
let timerId = null;
let serverReady = false;

const statusLabels = {
  queued: "В очереди",
  running: "Выполняется",
  completed: "Готово",
  failed: "Ошибка",
};

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    throw new Error(payload?.detail || `Ошибка HTTP ${response.status}`);
  }
  return payload;
}

async function checkHealth() {
  try {
    const health = await requestJson("/api/health");
    serverReady = health.ready;
    serverState.classList.toggle("online", health.ready);
    serverState.classList.toggle("offline", !health.ready);
    serverState.querySelector("span:last-child").textContent = health.ready
      ? health.status === "ok"
        ? "Сервер готов"
        : "FFmpeg требует настройки"
      : "FFprobe не найден";
    submitButton.disabled = !health.ready;
    if (!health.ready) {
      showError(
        health.ffprobe.error ||
          "FFprobe недоступен. Проверьте PATH или TRAVELMOVIEAI_FFPROBE_BINARY.",
      );
    }
  } catch (error) {
    serverReady = false;
    serverState.classList.add("offline");
    serverState.classList.remove("online");
    serverState.querySelector("span:last-child").textContent = "Нет связи";
    submitButton.disabled = true;
    showError(error.message);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideError();
  results.classList.add("hidden");
  if (!serverReady) {
    showError("Сервер не готов к анализу. Проверьте FFprobe.");
    return;
  }
  submitButton.disabled = true;

  try {
    currentJob = await requestJson("/api/scans", {
      method: "POST",
      body: JSON.stringify({
        input_path: inputPath.value.trim(),
        workspace: workspace.value.trim() || null,
      }),
    });
    startedAt = new Date();
    showJob(currentJob);
    startTimer();
    await pollJob(currentJob.id);
  } catch (error) {
    showError(error.message);
    submitButton.disabled = false;
  }
});

async function pollJob(jobId) {
  while (currentJob && currentJob.id === jobId) {
    await sleep(700);
    try {
      currentJob = await requestJson(`/api/scans/${jobId}`);
      showJob(currentJob);

      if (currentJob.status === "completed") {
        const report = await requestJson(`/api/scans/${jobId}/result`);
        showResults(report);
        submitButton.disabled = false;
        stopTimer();
        await loadHistory();
        return;
      }

      if (currentJob.status === "failed") {
        showError(currentJob.error || currentJob.message);
        submitButton.disabled = false;
        stopTimer();
        return;
      }
    } catch (error) {
      showError(error.message);
      submitButton.disabled = false;
      stopTimer();
      return;
    }
  }
}

async function loadHistory() {
  try {
    const history = await requestJson("/api/scans?limit=6");
    renderHistory(history.jobs || []);
  } catch {
    recentJobs.classList.add("hidden");
  }
}

function renderHistory(jobs) {
  recentJobsList.replaceChildren();
  recentJobs.classList.toggle("hidden", jobs.length === 0);

  for (const job of jobs) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "recent-job";
    const dot = document.createElement("span");
    dot.className = `recent-job-dot ${job.status}`;
    const name = document.createElement("span");
    name.className = "recent-job-name";
    name.textContent = lastPathPart(job.input_path);
    name.title = job.input_path;
    const status = document.createElement("span");
    status.className = "recent-job-status";
    status.textContent = statusLabels[job.status] || job.status;
    button.append(dot, name, status);
    button.addEventListener("click", () => openHistoryJob(job));
    recentJobsList.append(button);
  }
}

async function openHistoryJob(job) {
  currentJob = job;
  startedAt = new Date(job.started_at || job.created_at);
  showJob(job);
  hideError();

  if (job.status === "completed") {
    try {
      const report = await requestJson(`/api/scans/${job.id}/result`);
      showResults(report);
    } catch (error) {
      showError(error.message);
    }
  } else if (job.status === "failed") {
    showError(job.error || job.message);
  } else {
    startTimer();
    await pollJob(job.id);
  }
}

function showJob(job) {
  emptyState.classList.add("hidden");
  jobState.classList.remove("hidden");
  statusChip.textContent = statusLabels[job.status] || job.status;
  statusChip.className = `status-chip ${job.status}`;
  jobTitle.textContent =
    job.status === "completed"
      ? "Медиатека проиндексирована"
      : job.status === "failed"
        ? "Не удалось завершить анализ"
        : "Анализ материалов";
  jobMessage.textContent = job.message;
  sourceSummary.textContent = job.input_path;
  sourceSummary.title = job.input_path;
  workspaceSummary.textContent = job.workspace;
  workspaceSummary.title = job.workspace;
  progressBar.classList.toggle(
    "indeterminate",
    job.status === "queued" || job.status === "running",
  );
  progressBar.style.width = job.status === "completed" ? "100%" : "";
}

function showResults(report) {
  currentAssets = report.assets || [];
  document.querySelector("#stat-discovered").textContent = report.discovered_count;
  document.querySelector("#stat-probed").textContent = report.probed_count;
  document.querySelector("#stat-cached").textContent = report.cached_count;
  document.querySelector("#stat-errors").textContent = report.error_count;
  document.querySelector("#files-caption").textContent =
    `${report.discovered_count} файлов · ${formatDate(report.scanned_at)}`;
  results.classList.remove("hidden");
  renderFiles(currentAssets);
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderFiles(assets) {
  filesBody.replaceChildren();
  const visible = assets.slice(0, 250);

  for (const asset of visible) {
    const row = document.createElement("tr");
    row.append(
      cell(asset.relative_path, asset.scan_error || asset.path),
      pillCell(asset.media_type),
      cell(formatBytes(asset.size_bytes)),
      cell(formatDuration(asset.duration_seconds)),
      cell(formatResolution(asset)),
      statusCell(asset.scan_error),
    );
    filesBody.append(row);
  }

  const hiddenCount = assets.length - visible.length;
  tableNote.classList.toggle("hidden", hiddenCount <= 0);
  tableNote.textContent =
    hiddenCount > 0 ? `Показаны первые 250 файлов. Ещё скрыто: ${hiddenCount}.` : "";
}

function cell(value, title = "") {
  const element = document.createElement("td");
  element.textContent = value ?? "—";
  if (title) element.title = title;
  return element;
}

function pillCell(type) {
  const element = document.createElement("td");
  const pill = document.createElement("span");
  pill.className = "type-pill";
  pill.textContent = type;
  element.append(pill);
  return element;
}

function statusCell(error) {
  const element = document.createElement("td");
  const badge = document.createElement("span");
  badge.className = `file-status ${error ? "error" : "ok"}`;
  badge.textContent = error ? "Ошибка" : "Готово";
  if (error) badge.title = error;
  element.append(badge);
  return element;
}

fileFilter.addEventListener("input", () => {
  const query = fileFilter.value.trim().toLocaleLowerCase("ru");
  const filtered = currentAssets.filter((asset) =>
    asset.relative_path.toLocaleLowerCase("ru").includes(query),
  );
  renderFiles(filtered);
});

newScanButton.addEventListener("click", () => {
  currentJob = null;
  currentAssets = [];
  results.classList.add("hidden");
  jobState.classList.add("hidden");
  emptyState.classList.remove("hidden");
  fileFilter.value = "";
  hideError();
  inputPath.focus();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function hideError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

function startTimer() {
  stopTimer();
  updateTimer();
  timerId = window.setInterval(updateTimer, 1000);
}

function stopTimer() {
  if (timerId) window.clearInterval(timerId);
  timerId = null;
  updateTimer();
}

function updateTimer() {
  if (!startedAt) return;
  const seconds = Math.max(0, Math.floor((Date.now() - startedAt.getTime()) / 1000));
  const minutes = Math.floor(seconds / 60);
  jobTime.textContent = `${String(minutes).padStart(2, "0")}:${String(
    seconds % 60,
  ).padStart(2, "0")}`;
}

function formatBytes(bytes) {
  if (bytes === 0) return "0 Б";
  if (!bytes) return "—";
  const units = ["Б", "КБ", "МБ", "ГБ", "ТБ"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** index;
  return `${value.toLocaleString("ru-RU", { maximumFractionDigits: 1 })} ${units[index]}`;
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  return `${String(minutes).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function formatResolution(asset) {
  return asset.width && asset.height ? `${asset.width}×${asset.height}` : "—";
}

function formatDate(value) {
  return new Intl.DateTimeFormat("ru-RU", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function lastPathPart(value) {
  const normalized = value.replace(/[\\/]+$/, "");
  return normalized.split(/[\\/]/).pop() || value;
}

refreshJobs.addEventListener("click", loadHistory);
submitButton.disabled = true;
checkHealth();
loadHistory();
