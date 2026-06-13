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
const movieButton = document.querySelector("#movie-button");
const movieDuration = document.querySelector("#movie-duration");
const clipDuration = document.querySelector("#clip-duration");
const photoDuration = document.querySelector("#photo-duration");
const storyStyle = document.querySelector("#story-style");
const visionModel = document.querySelector("#vision-model");
const renderDevice = document.querySelector("#render-device");
const transitionType = document.querySelector("#transition-type");
const semanticAnalysis = document.querySelector("#semantic-analysis");
const qualityAnalysis = document.querySelector("#quality-analysis");
const musicMode = document.querySelector("#music-mode");
const musicProfile = document.querySelector("#music-profile");
const musicVolume = document.querySelector("#music-volume");
const musicVolumeValue = document.querySelector("#music-volume-value");
const musicPath = document.querySelector("#music-path");
const capabilityList = document.querySelector("#capability-list");
const movieProgress = document.querySelector("#movie-progress");
const movieStatus = document.querySelector("#movie-status");
const movieProgressTitle = document.querySelector("#movie-progress-title");
const movieProgressMessage = document.querySelector("#movie-progress-message");
const movieProgressBar = document.querySelector("#movie-progress-bar");
const movieResult = document.querySelector("#movie-result");
const movieResultSummary = document.querySelector("#movie-result-summary");
const movieDownload = document.querySelector("#movie-download");
const moviePreview = document.querySelector("#movie-preview");

let currentJob = null;
let currentAssets = [];
let startedAt = null;
let timerId = null;
let serverReady = false;
let movieReady = false;
let currentMovieJob = null;

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
    movieReady = health.ffmpeg.available && health.ffprobe.available;
    serverState.classList.toggle("online", health.ready);
    serverState.classList.toggle("offline", !health.ready);
    serverState.querySelector("span:last-child").textContent = health.ready
      ? health.status === "ok"
        ? "Сервер готов"
        : "FFmpeg требует настройки"
      : "FFprobe не найден";
    submitButton.disabled = !health.ready;
    movieButton.disabled = !movieReady;
    if (!health.ready) {
      showError(
        health.ffprobe.error ||
          "FFprobe недоступен. Проверьте PATH или TRAVELMOVIEAI_FFPROBE_BINARY.",
      );
    }
  } catch (error) {
    serverReady = false;
    movieReady = false;
    serverState.classList.add("offline");
    serverState.classList.remove("online");
    serverState.querySelector("span:last-child").textContent = "Нет связи";
    submitButton.disabled = true;
    movieButton.disabled = true;
    showError(error.message);
  }
}

async function loadCapabilities() {
  try {
    const capabilities = await requestJson("/api/capabilities");
    renderCapabilities(capabilities);
    populateModels(capabilities.ai);
    if (!capabilities.cuda.ffmpeg_nvenc && renderDevice.value === "cuda") {
      renderDevice.value = "auto";
    }
  } catch {
    capabilityList.replaceChildren(capabilityChip("AI/GPU: нет данных", false));
    visionModel.replaceChildren(new Option("Модели недоступны", ""));
  }
}

function renderCapabilities(capabilities) {
  capabilityList.replaceChildren(
    capabilityChip(
      capabilities.ai.available
        ? `LM Studio · ${capabilities.ai.models.length} моделей`
        : "LM Studio недоступен",
      capabilities.ai.available,
    ),
    capabilityChip(
      capabilities.cuda.available
        ? `${capabilities.cuda.gpu_name} · ${capabilities.cuda.memory_mb} MB`
        : "NVIDIA GPU не найдена",
      capabilities.cuda.available,
    ),
    capabilityChip(
      capabilities.cuda.ffmpeg_nvenc ? "NVENC готов" : "NVENC недоступен",
      capabilities.cuda.ffmpeg_nvenc,
    ),
    capabilityChip(
      capabilities.opencv_available ? "OpenCV готов" : "OpenCV fallback: Pillow",
      capabilities.opencv_available,
    ),
  );
}

function capabilityChip(label, available) {
  const chip = document.createElement("span");
  chip.className = `capability-chip ${available ? "ready" : "warning"}`;
  chip.textContent = label;
  return chip;
}

function populateModels(ai) {
  visionModel.replaceChildren();
  if (!ai.models.length) {
    visionModel.append(new Option(ai.configured_model || "Нет моделей", ai.configured_model));
    return;
  }
  for (const model of ai.models) {
    const suffix = model.likely_vision ? " · vision" : " · совместимость не проверена";
    const option = new Option(`${model.id}${suffix}`, model.id);
    option.selected = model.recommended;
    visionModel.append(option);
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
  movieButton.disabled = !movieReady;
  renderFiles(currentAssets);
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

movieButton.addEventListener("click", async () => {
  if (!currentJob || currentJob.status !== "completed") {
    showError("Сначала завершите анализ медиатеки.");
    return;
  }
  if (!movieReady) {
    showError("Для монтажа требуется FFmpeg.");
    return;
  }

  hideError();
  movieResult.classList.add("hidden");
  movieProgress.classList.remove("hidden");
  movieButton.disabled = true;
  movieStatus.textContent = "В очереди";
  movieStatus.className = "status-chip running";
  movieProgressTitle.textContent = "Подготовка фильма";
  movieProgressMessage.textContent = "Монтаж ожидает запуска.";
  movieProgressBar.style.width = "2%";

  try {
    currentMovieJob = await requestJson("/api/movies", {
      method: "POST",
      body: JSON.stringify({
        input_path: currentJob.input_path,
        workspace: currentJob.workspace,
        settings: {
          target_duration_seconds: Number(movieDuration.value),
          max_video_clip_seconds: Number(clipDuration.value),
          photo_duration_seconds: Number(photoDuration.value),
          semantic_analysis: semanticAnalysis.checked,
          quality_analysis: qualityAnalysis.checked,
          vision_model: visionModel.value || null,
          render_device: renderDevice.value,
          story_style: storyStyle.value,
          transition: transitionType.value,
          music_enabled: musicMode.value !== "none",
          music_mode: musicMode.value,
          music_profile: musicProfile.value,
          music_volume: Number(musicVolume.value) / 100,
          music_path: musicPath.value.trim() || null,
        },
      }),
    });
    await pollMovie(currentMovieJob.id);
  } catch (error) {
    showError(error.message);
    movieProgress.classList.add("hidden");
    movieButton.disabled = false;
  }
});

async function pollMovie(jobId) {
  while (currentMovieJob && currentMovieJob.id === jobId) {
    await sleep(800);
    try {
      currentMovieJob = await requestJson(`/api/movies/${jobId}`);
      showMovieProgress(currentMovieJob);
      if (currentMovieJob.status === "completed") {
        showMovieResult(currentMovieJob);
        movieButton.disabled = false;
        return;
      }
      if (currentMovieJob.status === "failed") {
        showError(currentMovieJob.error || currentMovieJob.message);
        movieButton.disabled = false;
        return;
      }
    } catch (error) {
      showError(error.message);
      movieButton.disabled = false;
      return;
    }
  }
}

function showMovieProgress(job) {
  movieProgress.classList.remove("hidden");
  movieStatus.textContent = statusLabels[job.status] || job.status;
  movieStatus.className = `status-chip ${job.status}`;
  movieProgressTitle.textContent =
    job.status === "completed"
      ? "Фильм готов"
      : job.status === "failed"
        ? "Ошибка монтажа"
        : "Идёт монтаж";
  movieProgressMessage.textContent = job.message;
  const percent =
    job.progress_total > 0
      ? Math.max(2, Math.round((job.progress_current / job.progress_total) * 100))
      : 2;
  movieProgressBar.style.width = `${percent}%`;
}

function showMovieResult(job) {
  const downloadUrl = `/api/movies/${job.id}/download`;
  movieResultSummary.textContent =
    `${job.clip_count} фрагментов · ${formatDuration(job.duration_seconds)} · ${
      job.selection_mode === "semantic" ? "AI-отбор" : "быстрый режим"
    } · ${job.render_encoder || "кодировщик неизвестен"} · ${
      job.music_profile || job.music_mode || "без музыки"
    }`;
  movieDownload.href = downloadUrl;
  moviePreview.src = downloadUrl;
  movieResult.classList.remove("hidden");
  movieResult.scrollIntoView({ behavior: "smooth", block: "center" });
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
  movieProgress.classList.add("hidden");
  movieResult.classList.add("hidden");
  moviePreview.removeAttribute("src");
  moviePreview.load();
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
musicMode.addEventListener("change", () => {
  musicPath.disabled = musicMode.value !== "manual";
  musicProfile.disabled = ["manual", "library", "none"].includes(musicMode.value);
});
musicVolume.addEventListener("input", () => {
  musicVolumeValue.textContent = `${musicVolume.value}%`;
});
submitButton.disabled = true;
checkHealth();
loadCapabilities();
loadHistory();
