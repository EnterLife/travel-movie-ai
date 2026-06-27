const form = document.querySelector("#scan-form");
const inputPath = document.querySelector("#input-path");
const workspace = document.querySelector("#workspace");
const browseInputPath = document.querySelector("#browse-input-path");
const browseWorkspace = document.querySelector("#browse-workspace");
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
const analysisQualityMode = document.querySelector("#analysis-quality-mode");
const visionProvider = document.querySelector("#vision-provider");
const visionModel = document.querySelector("#vision-model");
const visionModelSource = document.querySelector("#vision-model-source");
const renderDevice = document.querySelector("#render-device");
const transitionType = document.querySelector("#transition-type");
const previewMode = document.querySelector("#preview-mode");
const semanticAnalysis = document.querySelector("#semantic-analysis");
const qualityAnalysis = document.querySelector("#quality-analysis");
const speechAnalysis = document.querySelector("#speech-analysis");
const musicMode = document.querySelector("#music-mode");
const musicEngine = document.querySelector("#music-engine");
const musicModel = document.querySelector("#music-model");
const musicProfile = document.querySelector("#music-profile");
const musicSync = document.querySelector("#music-sync");
const musicVolume = document.querySelector("#music-volume");
const musicVolumeValue = document.querySelector("#music-volume-value");
const musicPath = document.querySelector("#music-path");
const capabilityList = document.querySelector("#capability-list");
const movieProgress = document.querySelector("#movie-progress");
const movieStatus = document.querySelector("#movie-status");
const movieProgressTitle = document.querySelector("#movie-progress-title");
const movieProgressMessage = document.querySelector("#movie-progress-message");
const movieProgressBar = document.querySelector("#movie-progress-bar");
const movieProgressPercent = document.querySelector("#movie-progress-percent");
const movieProgressPhase = document.querySelector("#movie-progress-phase");
const movieProgressElapsed = document.querySelector("#movie-progress-elapsed");
const movieProgressEta = document.querySelector("#movie-progress-eta");
const movieProgressResources = document.querySelector("#movie-progress-resources");
const movieSubtasksList = document.querySelector("#movie-subtasks-list");
const movieSubtasksSummary = document.querySelector("#movie-subtasks-summary");
const movieLog = document.querySelector("#movie-log");
const movieLogCount = document.querySelector("#movie-log-count");
const movieResult = document.querySelector("#movie-result");
const movieResultSummary = document.querySelector("#movie-result-summary");
const sceneReview = document.querySelector("#scene-review");
const sceneGrid = document.querySelector("#scene-grid");
const movieDownload = document.querySelector("#movie-download");
const moviePreview = document.querySelector("#movie-preview");
const moviePauseButton = document.querySelector("#movie-pause-button");
const movieCancelButton = document.querySelector("#movie-cancel-button");

let currentJob = null;
let currentAssets = [];
let startedAt = null;
let timerId = null;
let serverReady = false;
let movieReady = false;
let currentMovieJob = null;
let loadedCapabilities = null;
let defaultWorkspaceRoot = "";
let workspaceIsAutomatic = true;

const statusLabels = {
  queued: "Queued",
  running: "Running",
  paused: "Paused",
  cancelled: "Stopped",
  completed: "Complete",
  failed: "Failed",
};

const phaseLabels = {
  queued: "Waiting",
  preparing: "Preparing",
  media_scan: "Media scan",
  scene_detection: "Scene detection",
  frame_sampling: "Frame extraction",
  quality_analysis: "OpenCV analysis",
  vision_analysis: "Vision AI",
  speech_analysis: "Speech recognition",
  audio_analysis: "Audio Analysis",
  story_builder: "Story and selection",
  music: "Music",
  timeline: "Timeline",
  rendering: "Rendering",
  validation: "Validation",
  completed: "Complete",
  failed: "Failed",
  processing: "Processing",
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
    throw new Error(payload?.detail || `HTTP error ${response.status}`);
  }
  return payload;
}

async function pickDirectory(purpose, field, button) {
  hideError();
  button.disabled = true;
  const originalLabel = button.textContent;
  button.textContent = "Opening...";
  try {
    const payload = await requestJson("/api/dialogs/directory", {
      method: "POST",
      body: JSON.stringify({
        purpose,
        initial_path: field.value.trim() || null,
      }),
    });
    if (payload.selected_path) {
      field.value = payload.selected_path;
      if (purpose === "workspace") workspaceIsAutomatic = false;
      field.dispatchEvent(new Event("change", { bubbles: true }));
    }
  } catch (error) {
    showError(`Could not open the folder picker: ${error.message}`);
  } finally {
    button.disabled = false;
    button.textContent = originalLabel;
  }
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
        ? "Server ready"
        : "FFmpeg needs configuration"
      : "FFprobe not found";
    submitButton.disabled = !health.ready;
    movieButton.disabled = !movieReady;
    if (!health.ready) {
      showError(
        health.ffprobe.error ||
          "FFprobe is unavailable. Check PATH or ffprobe_binary in configs/settings.toml.",
      );
    }
  } catch (error) {
    serverReady = false;
    movieReady = false;
    serverState.classList.add("offline");
    serverState.classList.remove("online");
    serverState.querySelector("span:last-child").textContent = "Offline";
    submitButton.disabled = true;
    movieButton.disabled = true;
    showError(error.message);
  }
}

async function loadCapabilities() {
  try {
    const capabilities = await requestJson("/api/capabilities");
    loadedCapabilities = capabilities;
    defaultWorkspaceRoot = capabilities.default_workspace_root || "";
    updateAutomaticWorkspace();
    renderCapabilities(capabilities);
    populateModels(capabilities);
    if (!capabilities.cuda.ffmpeg_nvenc && renderDevice.value === "cuda") {
      renderDevice.value = "auto";
    }
  } catch {
    capabilityList.replaceChildren(capabilityChip("AI/GPU: unavailable", false));
    visionModel.replaceChildren(new Option("Models unavailable", ""));
  }
}

function renderCapabilities(capabilities) {
  capabilityList.replaceChildren(
    capabilityChip(
      capabilities.local_ai.available
        ? `Local AI · ${shortModelName(capabilities.local_ai.resolved_model)}`
        : "Local AI dependencies missing",
      capabilities.local_ai.available,
    ),
    capabilityChip(
      capabilities.local_ai.downloads_enabled
        ? "Models auto-download"
        : "Models: cache only",
      capabilities.local_ai.downloads_enabled,
    ),
    capabilityChip(
      capabilities.cuda.available
        ? `${capabilities.cuda.gpu_name} · ${capabilities.cuda.memory_mb} MB`
        : "NVIDIA GPU not found",
      capabilities.cuda.available,
    ),
    capabilityChip(
      capabilities.cuda.ffmpeg_nvenc ? "NVENC ready" : "NVENC unavailable",
      capabilities.cuda.ffmpeg_nvenc,
    ),
    capabilityChip(
      capabilities.cuda.torch_cuda ? "Vision AI · CUDA" : "Vision AI · CPU",
      capabilities.cuda.torch_cuda,
    ),
    capabilityChip(
      capabilities.music_ai.runtime_installed
        ? `Music AI · ${shortModelName(capabilities.music_ai.resolved_model)}`
        : "Music AI · installs on first run",
      capabilities.music_ai.available,
    ),
    capabilityChip(
      capabilities.opencv_available ? "OpenCV ready" : "OpenCV fallback: Pillow",
      capabilities.opencv_available,
    ),
    capabilityChip(
      `${capabilities.resources.logical_cores} CPU · frames ${capabilities.resources.frame_workers}x · render ${capabilities.resources.render_workers}x`,
      true,
    ),
  );
}

function capabilityChip(label, available) {
  const chip = document.createElement("span");
  chip.className = `capability-chip ${available ? "ready" : "warning"}`;
  chip.textContent = label;
  return chip;
}

function populateModels(capabilities) {
  populateMusicModels(capabilities.music_ai);
  if (visionProvider.value === "florence") {
    populateFlorenceModels();
    return;
  }
  visionModelSource.textContent = "local";
  visionModel.replaceChildren(
    new Option(
      `Auto · ${shortModelName(capabilities.local_ai.resolved_model)}`,
      "auto",
      true,
      capabilities.local_ai.configured_model === "auto",
    ),
  );
  for (const model of capabilities.local_ai.models) {
    const option = new Option(localModelLabel(model.id), model.id);
    option.selected = capabilities.local_ai.configured_model === model.id;
    visionModel.append(option);
  }
}

function populateMusicModels(musicAi) {
  musicModel.replaceChildren(
    new Option(
      `Auto · ${shortModelName(musicAi.resolved_model)}`,
      "auto",
      true,
      musicAi.configured_model === "auto",
    ),
  );
  for (const model of musicAi.models) {
    const option = new Option(`${shortModelName(model.id)} · high quality`, model.id);
    option.selected = musicAi.configured_model === model.id;
    musicModel.append(option);
  }
}

function populateFlorenceModels() {
  visionModelSource.textContent = "local";
  visionModel.replaceChildren(
    new Option("microsoft/Florence-2-large", "microsoft/Florence-2-large", true, true),
    new Option("microsoft/Florence-2-base", "microsoft/Florence-2-base"),
  );
}

function shortModelName(model) {
  return model.split("/").pop();
}

function localModelLabel(model) {
  const name = shortModelName(model);
  if (name.includes("3B")) return `${name} · fast, 6 GB VRAM`;
  if (name.includes("7B")) return `${name} · higher quality, GPU + RAM`;
  if (name.includes("32B")) return `${name} · maximum quality, very slow`;
  return name;
}

function updateAutomaticWorkspace() {
  if (!workspaceIsAutomatic || !defaultWorkspaceRoot) return;
  const sourceName = lastPathPart(inputPath.value.trim());
  const separator = defaultWorkspaceRoot.includes("\\") ? "\\" : "/";
  const root = defaultWorkspaceRoot.replace(/[\\/]+$/, "");
  workspace.value = sourceName ? `${root}${separator}${sourceName}` : root;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideError();
  results.classList.add("hidden");
  if (!serverReady) {
    showError("The server is not ready. Check FFprobe.");
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
      ? "Media library indexed"
      : job.status === "failed"
        ? "Scan failed"
        : "Scanning media";
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
    `${report.discovered_count} files · ${formatDate(report.scanned_at)}`;
  results.classList.remove("hidden");
  movieButton.disabled = !movieReady;
  renderFiles(currentAssets);
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

movieButton.addEventListener("click", async () => {
  if (!currentJob || currentJob.status !== "completed") {
    showError("Complete a media scan first.");
    return;
  }
  if (!movieReady) {
    showError("FFmpeg is required to create a movie.");
    return;
  }

  hideError();
  movieResult.classList.add("hidden");
  movieProgress.classList.remove("hidden");
  movieButton.disabled = true;
  movieStatus.textContent = "Queued";
  movieStatus.className = "status-chip running";
  movieProgressTitle.textContent = "Preparing film";
  movieProgressMessage.textContent = "The edit is waiting to start.";
  movieProgressBar.style.width = "2%";
  movieProgressPercent.textContent = "0%";
  movieProgressPhase.textContent = "Waiting";
  movieProgressElapsed.textContent = "00:00";
  movieProgressEta.textContent = "—";
  movieProgressResources.textContent = "Detecting...";
  movieSubtasksList.replaceChildren();
  movieSubtasksSummary.textContent = "0 / 0";
  movieLog.replaceChildren();
  movieLogCount.textContent = "0 messages";

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
          speech_analysis: speechAnalysis.checked,
          audio_analysis: true,
          vision_provider: visionProvider.value,
          vision_model: visionModel.value || null,
          render_device: renderDevice.value,
          story_style: storyStyle.value,
          analysis_quality_mode: analysisQualityMode.value,
          transition: transitionType.value,
          preview_mode: previewMode.checked,
          music_enabled: musicMode.value !== "none",
          music_mode: musicMode.value,
          music_engine: musicEngine.value,
          music_model: musicModel.value || null,
          music_profile: musicProfile.value,
          music_sync: musicSync.checked,
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
    await sleep(500);
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
      if (currentMovieJob.status === "cancelled") {
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
      ? "Film ready"
      : job.status === "failed"
        ? "Edit failed"
        : job.status === "paused"
          ? "Edit paused"
          : job.status === "cancelled"
            ? "Edit stopped"
        : "Editing";
  moviePauseButton.disabled = !["running", "paused", "queued"].includes(job.status);
  moviePauseButton.textContent = job.status === "paused" ? "Resume" : "Pause";
  movieCancelButton.disabled = !["running", "paused", "queued"].includes(job.status);
  movieProgressMessage.textContent = job.message;
  const percent = Math.max(0, Math.min(100, job.progress_percent || 0));
  movieProgressBar.style.width = `${Math.max(2, percent)}%`;
  movieProgressPercent.textContent = `${Math.round(percent)}%`;
  movieProgressPhase.textContent = phaseLabels[job.phase] || job.phase;
  movieProgressElapsed.textContent = formatClock(job.elapsed_seconds);
  movieProgressEta.textContent =
    job.eta_seconds == null ? "—" : `≈ ${formatClock(job.eta_seconds)}`;
  movieProgressResources.textContent = job.resources?.summary || "Detecting...";
  renderMovieSubtasks(job.subtasks || []);
  renderMovieLogs(job.logs || []);
}

function renderMovieSubtasks(subtasks) {
  movieSubtasksList.replaceChildren();
  const finished = subtasks.filter((task) =>
    ["completed", "skipped"].includes(task.status),
  ).length;
  movieSubtasksSummary.textContent = `${finished} / ${subtasks.length}`;

  for (const task of subtasks) {
    const row = document.createElement("article");
    row.className = `movie-subtask ${task.status}`;

    const header = document.createElement("div");
    header.className = "movie-subtask-header";
    const label = document.createElement("strong");
    label.textContent = task.label;
    const state = document.createElement("span");
    state.textContent =
      task.status === "skipped"
        ? "Disabled"
        : task.status === "completed"
          ? "Done"
          : task.status === "failed"
            ? "Failed"
            : `${Math.round(task.progress_percent)}%`;
    header.append(label, state);

    const track = document.createElement("div");
    track.className = "movie-subtask-track";
    const bar = document.createElement("span");
    bar.style.width = `${Math.max(0, Math.min(100, task.progress_percent))}%`;
    track.append(bar);

    const message = document.createElement("p");
    message.textContent = task.message;
    row.append(header, track, message);
    movieSubtasksList.append(row);
  }
}

function renderMovieLogs(logs) {
  const keepPinned =
    movieLog.scrollHeight - movieLog.scrollTop - movieLog.clientHeight < 36;
  movieLog.replaceChildren();
  for (const entry of logs) {
    const row = document.createElement("div");
    row.className = `movie-log-row ${entry.level}`;
    const time = document.createElement("time");
    time.textContent = new Date(entry.timestamp).toLocaleTimeString("ru-RU");
    const phase = document.createElement("span");
    phase.textContent = `${Math.round(entry.progress_percent)}%`;
    const message = document.createElement("p");
    message.textContent = entry.message;
    row.append(time, phase, message);
    movieLog.append(row);
  }
  movieLogCount.textContent = `${logs.length} ${logs.length === 1 ? "message" : "messages"}`;
  if (keepPinned) movieLog.scrollTop = movieLog.scrollHeight;
}

function showMovieResult(job) {
  const downloadUrl = `/api/movies/${job.id}/download`;
  movieResultSummary.textContent =
    `${job.clip_count} clips · ${formatDuration(job.duration_seconds)} · ${
      job.selection_mode === "semantic" ? "AI selection" : "quick mode"
    } · ${job.render_encoder || "unknown encoder"} · ${
      job.music_profile || job.music_mode || "no music"
    } · ${job.music_generator || "music file"}`;
  movieDownload.href = downloadUrl;
  moviePreview.src = downloadUrl;
  movieResult.classList.remove("hidden");
  loadSceneReview().catch((error) => showError(error.message));
  movieResult.scrollIntoView({ behavior: "smooth", block: "center" });
}

async function loadSceneReview() {
  if (!currentJob || !semanticAnalysis.checked) {
    sceneReview.classList.add("hidden");
    return;
  }
  const query = new URLSearchParams({
    input_path: currentJob.input_path,
    workspace: currentJob.workspace,
  });
  const payload = await requestJson(`/api/scenes?${query}`);
  renderSceneReview(payload.scenes || []);
}

function renderSceneReview(scenes) {
  sceneGrid.replaceChildren();
  for (const scene of scenes.slice(0, 120)) {
    const card = document.createElement("article");
    card.className = "scene-card";
    const query = new URLSearchParams({
      input_path: currentJob.input_path,
      workspace: currentJob.workspace,
    });
    const image = document.createElement("img");
    image.loading = "lazy";
    image.alt = scene.caption || "Scene frames";
    image.src = `/api/scenes/${scene.id}/thumbnail?${query}`;

    const copy = document.createElement("div");
    copy.className = "scene-card-copy";
    const title = document.createElement("strong");
    title.textContent = scene.caption || "Untitled scene";
    const metrics = document.createElement("small");
    const rank = scene.metadata?.ranking_score;
    metrics.textContent =
      `AI ${formatScore(scene.importance_score)} · quality ${formatScore(
        scene.quality_score,
      )}${rank == null ? "" : ` · final ${formatScore(rank)}`}`;
    const reasons = document.createElement("p");
    reasons.textContent = sceneDecisionSummary(scene);
    copy.append(title, metrics, reasons);

    const actions = document.createElement("div");
    actions.className = "scene-actions";
    for (const [decision, label] of [
      ["auto", "Auto"],
      ["include", "Include"],
      ["exclude", "Exclude"],
    ]) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      const active = (scene.metadata?.selection_override || "auto") === decision;
      button.className = active ? `active ${decision}` : decision;
      button.addEventListener("click", async () => {
        await updateSceneDecision(scene.id, decision);
        scene.metadata = { ...(scene.metadata || {}) };
        if (decision === "auto") {
          delete scene.metadata.selection_override;
        } else {
          scene.metadata.selection_override = decision;
        }
        renderSceneReview(scenes);
      });
      actions.append(button);
    }
    card.append(image, copy, actions);
    sceneGrid.append(card);
  }
  sceneReview.classList.toggle("hidden", scenes.length === 0);
}

async function updateSceneDecision(sceneId, decision) {
  await requestJson(`/api/scenes/${sceneId}`, {
    method: "PATCH",
    body: JSON.stringify({
      input_path: currentJob.input_path,
      workspace: currentJob.workspace,
      decision,
    }),
  });
}

function sceneDecisionSummary(scene) {
  if (scene.metadata?.duplicate_status === "duplicate") {
    return "Similar scene: skipped automatically unless you mark it as included.";
  }
  const technical = scene.metadata?.technical_rejection_reasons || [];
  if (technical.length) {
    return `Technical issues: ${technical.join(", ")}.`;
  }
  const reasons = scene.metadata?.ranking_reasons || [];
  return reasons.join(" · ") || "A decision will appear after semantic analysis.";
}

function formatScore(value) {
  return value == null ? "—" : Math.round(value);
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
    hiddenCount > 0 ? `Showing the first 250 files. Hidden: ${hiddenCount}.` : "";
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
  badge.textContent = error ? "Error" : "Ready";
  if (error) badge.title = error;
  element.append(badge);
  return element;
}

fileFilter.addEventListener("input", () => {
  const query = fileFilter.value.trim().toLocaleLowerCase("en-US");
  const filtered = currentAssets.filter((asset) =>
    asset.relative_path.toLocaleLowerCase("en-US").includes(query),
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
  if (bytes === 0) return "0 B";
  if (!bytes) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** index;
  return `${value.toLocaleString("en-US", { maximumFractionDigits: 1 })} ${units[index]}`;
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  return `${String(minutes).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function formatClock(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const tail = `${String(minutes).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
  return hours > 0 ? `${String(hours).padStart(2, "0")}:${tail}` : tail;
}

function formatResolution(asset) {
  return asset.width && asset.height ? `${asset.width}×${asset.height}` : "—";
}

function formatDate(value) {
  return new Intl.DateTimeFormat("en-US", {
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
browseInputPath.addEventListener("click", () =>
  pickDirectory("input", inputPath, browseInputPath),
);
browseWorkspace.addEventListener("click", () =>
  pickDirectory("workspace", workspace, browseWorkspace),
);
inputPath.addEventListener("input", updateAutomaticWorkspace);
inputPath.addEventListener("change", updateAutomaticWorkspace);
workspace.addEventListener("input", () => {
  workspaceIsAutomatic = false;
});
moviePauseButton.addEventListener("click", async () => {
  if (!currentMovieJob) return;
  try {
    const action = currentMovieJob.status === "paused" ? "resume" : "pause";
    currentMovieJob = await requestJson(
      `/api/movies/${currentMovieJob.id}/${action}`,
      { method: "POST" },
    );
    showMovieProgress(currentMovieJob);
  } catch (error) {
    showError(error.message);
  }
});
movieCancelButton.addEventListener("click", async () => {
  if (!currentMovieJob) return;
  if (!window.confirm("Stop the edit? Existing cache files will be kept.")) {
    return;
  }
  try {
    currentMovieJob = await requestJson(
      `/api/movies/${currentMovieJob.id}/cancel`,
      { method: "POST" },
    );
    showMovieProgress(currentMovieJob);
  } catch (error) {
    showError(error.message);
  }
});
visionProvider.addEventListener("change", () => {
  if (!loadedCapabilities) {
    loadCapabilities();
    return;
  }
  populateModels(loadedCapabilities);
});
musicMode.addEventListener("change", () => {
  musicPath.disabled = musicMode.value !== "manual";
  musicProfile.disabled = ["manual", "library", "none"].includes(musicMode.value);
  musicSync.disabled = ["manual", "library", "none"].includes(musicMode.value);
  musicEngine.disabled = ["manual", "library", "none"].includes(musicMode.value);
  musicModel.disabled =
    ["manual", "library", "none"].includes(musicMode.value) ||
    musicEngine.value === "procedural";
});
musicEngine.addEventListener("change", () => {
  musicModel.disabled = musicEngine.value === "procedural";
});
musicVolume.addEventListener("input", () => {
  musicVolumeValue.textContent = `${musicVolume.value}%`;
});
submitButton.disabled = true;
checkHealth();
loadCapabilities();
loadHistory();
