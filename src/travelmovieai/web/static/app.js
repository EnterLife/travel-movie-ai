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
const movieVariant = document.querySelector("#movie-variant");
const movieDuration = document.querySelector("#movie-duration");
const clipDuration = document.querySelector("#clip-duration");
const photoDuration = document.querySelector("#photo-duration");
const storyStyle = document.querySelector("#story-style");
const analysisQualityMode = document.querySelector("#analysis-quality-mode");
const visionProvider = document.querySelector("#vision-provider");
const visionModel = document.querySelector("#vision-model");
const visionModelSource = document.querySelector("#vision-model-source");
const renderDevice = document.querySelector("#render-device");
const transition = document.querySelector("#transition");
const preserveChronology = document.querySelector("#preserve-chronology");
const previewMode = document.querySelector("#preview-mode");
const semanticAnalysis = document.querySelector("#semantic-analysis");
const qualityAnalysis = document.querySelector("#quality-analysis");
const speechAnalysis = document.querySelector("#speech-analysis");
const narrationEnabled = document.querySelector("#narration-enabled");
const framingMode = document.querySelector("#framing-mode");
const verticalVideoLayout = document.querySelector("#vertical-video-layout");
const photoMotion = document.querySelector("#photo-motion");
const colorNormalization = document.querySelector("#color-normalization");
const hdrToSdr = document.querySelector("#hdr-to-sdr");
const eventTitlesEnabled = document.querySelector("#event-titles-enabled");
const sceneSubtitlesEnabled = document.querySelector("#scene-subtitles-enabled");
const creditsText = document.querySelector("#credits-text");
const musicMode = document.querySelector("#music-mode");
const musicEngine = document.querySelector("#music-engine");
const musicModel = document.querySelector("#music-model");
const musicProfile = document.querySelector("#music-profile");
const musicSync = document.querySelector("#music-sync");
const musicVolume = document.querySelector("#music-volume");
const musicVolumeValue = document.querySelector("#music-volume-value");
const musicPath = document.querySelector("#music-path");
const musicBpmAnalysis = document.querySelector("#music-bpm-analysis");
const musicVolumeEnvelope = document.querySelector("#music-volume-envelope");
const narrationVolume = document.querySelector("#narration-volume");
const backgroundVolume = document.querySelector("#background-volume");
const sourceAudioVolume = document.querySelector("#source-audio-volume");
const capabilityList = document.querySelector("#capability-list");
const capabilityGuidance = document.querySelector("#capability-guidance");
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
const sceneEventFilter = document.querySelector("#scene-event-filter");
const scenePageStatus = document.querySelector("#scene-page-status");
const loadMoreScenes = document.querySelector("#load-more-scenes");
const movieDownload = document.querySelector("#movie-download");
const moviePreview = document.querySelector("#movie-preview");
const moviePauseButton = document.querySelector("#movie-pause-button");
const movieCancelButton = document.querySelector("#movie-cancel-button");
const editWorkspace = document.querySelector("#edit-workspace");
const eventList = document.querySelector("#event-list");
const refreshEdits = document.querySelector("#refresh-edits");
const versionBefore = document.querySelector("#version-before");
const versionAfter = document.querySelector("#version-after");
const compareVersions = document.querySelector("#compare-versions");
const versionComparison = document.querySelector("#version-comparison");

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
let currentScenes = [];
let currentSceneTotal = 0;
let scenePageLoading = false;
let currentEvents = [];
let currentVersions = [];
const scenePageSize = 60;

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
    const requestId = response.headers.get("X-Request-ID") || payload?.request_id;
    const detail = formatErrorDetail(payload?.detail) || `HTTP error ${response.status}`;
    const error = new Error(requestId ? `${detail} (request ${requestId})` : detail);
    error.status = response.status;
    error.requestId = requestId || null;
    throw error;
  }
  return payload;
}

function formatErrorDetail(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((issue) => {
        if (typeof issue === "string") return issue;
        const location = Array.isArray(issue?.loc) ? issue.loc.slice(1).join(" → ") : "input";
        return `${location || "input"}: ${issue?.msg || "invalid value"}`;
      })
      .join("; ");
  }
  if (detail && typeof detail === "object") {
    return detail.message || JSON.stringify(detail);
  }
  return "";
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
    serverReady = health.ffprobe.available;
    movieReady = health.ffmpeg.available && health.ffprobe.available;
    serverState.classList.toggle("online", serverReady);
    serverState.classList.toggle("offline", !serverReady);
    const statusText = !health.ffprobe.available
      ? "FFprobe not found"
      : health.ffmpeg.available
        ? "Server ready"
        : "Scans ready · FFmpeg not found";
    serverState.querySelector("span:last-child").textContent = statusText;
    submitButton.disabled =
      !serverReady || ["queued", "running"].includes(currentJob?.status);
    movieButton.disabled =
      !movieReady || ["queued", "running", "paused"].includes(currentMovieJob?.status);
    if (!serverReady) {
      showError(
        health.ffprobe.error ||
          "FFprobe is unavailable. Check PATH or ffprobe_binary in configs/settings.toml.",
      );
    } else if (!health.ffmpeg.available) {
      showError(
        health.ffmpeg.error ||
          "FFmpeg is unavailable. Media scans can run, but movie creation is disabled.",
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
    applyCapabilityAvailability(capabilities);
    renderDevice.value = capabilities.recommended_render_device || "cpu";
  } catch {
    capabilityList.replaceChildren(capabilityChip("AI/GPU: unavailable", false));
    capabilityGuidance.textContent =
      "Capability detection failed. AI options are disabled; quick local editing remains available.";
    disableCapabilityControl(semanticAnalysis, "Local vision capability is unknown.");
    disableCapabilityControl(speechAnalysis, "Speech capability is unknown.");
    disableCapabilityControl(narrationEnabled, "Narration capability is unknown.");
    updateDependentCapabilityControls();
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
        ? `${capabilities.cuda.gpu_name} · ${capabilities.cuda.free_memory_mb ?? "?"} / ${capabilities.cuda.memory_mb} MB free`
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
        : "Music AI unavailable · procedural music ready",
      capabilities.music_ai.available,
    ),
    capabilityChip(
      capabilities.speech.available ? "Faster Whisper ready" : "Speech unavailable",
      capabilities.speech.available,
    ),
    capabilityChip(
      capabilities.narration.available ? "Piper narration ready" : "Narration unavailable",
      capabilities.narration.available,
    ),
    capabilityChip(
      capabilities.opencv_available ? "OpenCV ready" : "OpenCV fallback: Pillow",
      capabilities.opencv_available,
    ),
    capabilityChip(
      `${capabilities.resources.logical_cores} CPU · frames ${capabilities.resources.frame_workers}x · render ${capabilities.resources.render_workers}x`,
      true,
    ),
    capabilityChip(
      `Default · ${capabilities.resources.device.toUpperCase()} / ${capabilities.recommended_resource_mode}`,
      true,
    ),
  );
}

function applyCapabilityAvailability(capabilities) {
  setCapabilityControl(
    semanticAnalysis,
    capabilities.local_ai.available,
    capabilityDetail(capabilities.local_ai),
  );
  setCapabilityControl(
    speechAnalysis,
    capabilities.speech.available,
    capabilityDetail(capabilities.speech),
  );
  setCapabilityControl(
    narrationEnabled,
    capabilities.narration.available,
    capabilityDetail(capabilities.narration),
  );
  visionProvider.disabled = !capabilities.local_ai.available;
  visionModel.disabled = !capabilities.local_ai.available;
  const cudaRenderOption = renderDevice.querySelector('option[value="cuda"]');
  if (cudaRenderOption) cudaRenderOption.disabled = !capabilities.cuda.ffmpeg_nvenc;
  if (!capabilities.cuda.ffmpeg_nvenc && renderDevice.value === "cuda") {
    renderDevice.value = "cpu";
  }
  updateDependentCapabilityControls();

  const unavailable = [
    ["Semantic selection", capabilities.local_ai],
    ["Speech", capabilities.speech],
    ["Narration", capabilities.narration],
  ].filter(([, capability]) => !capability.available);
  capabilityGuidance.textContent = unavailable.length
    ? unavailable
        .map(([label, capability]) => `${label}: ${capabilityDetail(capability)}`)
        .join(" ")
    : "All local AI editing features are available. Enable only the analysis you need.";
}

function capabilityDetail(capability) {
  return [capability.reason, capability.action].filter(Boolean).join(" ");
}

function setCapabilityControl(control, available, detail) {
  control.dataset.capabilityAvailable = String(available);
  if (!available) {
    disableCapabilityControl(control, detail);
    return;
  }
  control.disabled = false;
  control.title = "";
  const label = control.closest("label");
  if (label) label.title = "";
}

function disableCapabilityControl(control, detail) {
  control.checked = false;
  control.disabled = true;
  control.title = detail || "This local capability is unavailable.";
  const label = control.closest("label");
  if (label) label.title = control.title;
}

function updateDependentCapabilityControls() {
  const semanticReady =
    semanticAnalysis.dataset.capabilityAvailable === "true" && semanticAnalysis.checked;
  for (const control of [speechAnalysis, narrationEnabled]) {
    const capabilityReady = control.dataset.capabilityAvailable === "true";
    control.disabled = !semanticReady || !capabilityReady;
    if (!semanticReady) control.checked = false;
  }
  const smartFraming = framingMode.querySelector('option[value="smart"]');
  if (smartFraming) smartFraming.disabled = !semanticReady;
  if (!semanticReady && framingMode.value === "smart") framingMode.value = "fit";
  for (const control of [colorNormalization, eventTitlesEnabled, sceneSubtitlesEnabled]) {
    control.disabled = !semanticReady;
    if (!semanticReady) control.checked = false;
    control.title = semanticReady ? "" : "Requires semantic scene analysis.";
  }
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
  workspace.value = "";
  workspace.placeholder = sourceName
    ? `${root}${separator}${sourceName}-<source-id>`
    : `${root}${separator}<automatic-project>`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideError();
  results.classList.add("hidden");
  if (!serverReady) {
    showError("The scanner is not ready. Check FFprobe.");
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
  let retryDelay = 700;
  while (currentJob && currentJob.id === jobId) {
    await sleep(retryDelay);
    try {
      currentJob = await requestJson(`/api/scans/${jobId}`);
      retryDelay = 700;
      hideError();
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
      if (error.status === 404) {
        currentJob = null;
        submitButton.disabled = false;
        stopTimer();
        showError("The active scan is no longer available. Start a new scan.");
        return;
      }
      retryDelay = Math.min(retryDelay * 2, 8000);
      showError(`${error.message} Retrying the active scan…`);
    }
  }
}

async function loadHistory(attempt = 0) {
  try {
    const history = await requestJson("/api/scans?limit=6");
    const jobs = history.jobs || [];
    renderHistory(jobs);
    const active = jobs.find((job) => ["queued", "running"].includes(job.status));
    if (!currentJob && active) {
      submitButton.disabled = true;
      await openHistoryJob(active);
    }
  } catch (error) {
    recentJobs.classList.add("hidden");
    if (attempt < 4 && !currentJob) {
      await sleep(Math.min(500 * 2 ** attempt, 4000));
      return loadHistory(attempt + 1);
    }
    showError(`${error.message} Could not restore scan history.`);
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
  editWorkspace.classList.add("hidden");
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
        variant_name: movieVariant.value.trim() || "Default",
        settings: {
          target_duration_seconds: Number(movieDuration.value),
          max_video_clip_seconds: Number(clipDuration.value),
          photo_duration_seconds: Number(photoDuration.value),
          semantic_analysis: semanticAnalysis.checked,
          quality_analysis: qualityAnalysis.checked,
          speech_analysis: speechAnalysis.checked,
          narration_enabled: narrationEnabled.checked,
          narration_volume: Number(narrationVolume.value),
          background_volume_during_narration: Number(backgroundVolume.value),
          source_audio_volume: Number(sourceAudioVolume.value),
          audio_analysis: true,
          vision_provider: visionProvider.value,
          vision_model: visionModel.value || null,
          render_device: renderDevice.value,
          transition: transition.value,
          preserve_chronology: preserveChronology.checked,
          story_style: storyStyle.value,
          analysis_quality_mode: analysisQualityMode.value,
          preview_mode: previewMode.checked,
          framing_mode: framingMode.value,
          vertical_video_layout: verticalVideoLayout.value,
          photo_motion: photoMotion.value,
          color_normalization: colorNormalization.checked,
          hdr_to_sdr: hdrToSdr.checked,
          event_titles_enabled: eventTitlesEnabled.checked,
          scene_subtitles_enabled: sceneSubtitlesEnabled.checked,
          credits_text: creditsText.value.trim() || null,
          music_enabled: musicMode.value !== "none",
          music_mode: musicMode.value,
          music_engine: musicEngine.value,
          music_model: musicModel.value || null,
          music_profile: musicProfile.value,
          music_sync: musicSync.checked,
          music_bpm_analysis: musicBpmAnalysis.checked,
          music_volume_envelope: musicVolumeEnvelope.checked,
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
  let retryDelay = 500;
  while (currentMovieJob && currentMovieJob.id === jobId) {
    await sleep(retryDelay);
    try {
      currentMovieJob = await requestJson(`/api/movies/${jobId}`);
      retryDelay = 500;
      hideError();
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
      if (error.status === 404) {
        currentMovieJob = null;
        movieButton.disabled = false;
        showError("The active edit is no longer available. Start a new edit.");
        return;
      }
      retryDelay = Math.min(retryDelay * 2, 8000);
      showError(`${error.message} The edit is still running; reconnecting…`);
    }
  }
}

async function restoreActiveMovieJob(attempt = 0) {
  try {
    const history = await requestJson("/api/movies?limit=20");
    const active = (history.jobs || []).find((job) =>
      ["queued", "running", "paused"].includes(job.status),
    );
    if (!active || currentMovieJob) return;
    currentMovieJob = active;
    currentJob = {
      status: "completed",
      input_path: active.input_path,
      workspace: active.workspace,
    };
    movieButton.disabled = true;
    showMovieProgress(active);
    await pollMovie(active.id);
  } catch (error) {
    if (attempt < 4 && !currentMovieJob) {
      showError(`${error.message} Could not restore the active edit yet; retrying…`);
      await sleep(Math.min(500 * 2 ** attempt, 4000));
      return restoreActiveMovieJob(attempt + 1);
    }
    showError(`${error.message} Could not restore the active edit.`);
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
  const qualityDetails = [
    job.quality_gate_status ? `gate ${job.quality_gate_status}` : null,
    job.semantic_score_p10 == null ? null : `semantic p10 ${Math.round(job.semantic_score_p10)}`,
    job.dominant_event_ratio == null
      ? null
      : `dominant event ${Math.round(job.dominant_event_ratio * 100)}%`,
    job.adjacent_source_repeat_ratio == null
      ? null
      : `adjacent repeats ${Math.round(job.adjacent_source_repeat_ratio * 100)}%`,
    job.center_cut_ratio == null
      ? null
      : `center cuts ${Math.round(job.center_cut_ratio * 100)}%`,
    job.full_media_qa_completed ? "full-media QA" : null,
  ].filter(Boolean);
  movieResultSummary.textContent =
    `${job.clip_count} clips · ${formatDuration(job.duration_seconds)} · ${
      job.selection_mode === "semantic" ? "AI selection" : "quick mode"
    } · ${job.render_encoder || "unknown encoder"} · ${
      job.music_profile || job.music_mode || "no music"
    } · ${job.music_generator || "music file"} · quality ${
      job.quality_score == null ? "n/a" : Math.round(job.quality_score)
    }/100 (${job.quality_issue_count || 0} issues)${
      qualityDetails.length ? ` · ${qualityDetails.join(" · ")}` : ""
    }`;
  movieDownload.href = downloadUrl;
  moviePreview.src = downloadUrl;
  movieResult.classList.remove("hidden");
  loadEditingWorkspace().catch((error) => showError(error.message));
  movieResult.scrollIntoView({ behavior: "smooth", block: "center" });
}

async function loadSceneReview(reset = true) {
  if (!currentJob || !semanticAnalysis.checked) {
    currentScenes = [];
    currentSceneTotal = 0;
    sceneReview.classList.add("hidden");
    return;
  }
  if (scenePageLoading) return;
  if (reset) {
    currentScenes = [];
    currentSceneTotal = 0;
  }
  scenePageLoading = true;
  loadMoreScenes.disabled = true;
  const query = new URLSearchParams({
    input_path: currentJob.input_path,
    workspace: currentJob.workspace,
    offset: String(currentScenes.length),
    limit: String(scenePageSize),
  });
  if (sceneEventFilter.value) query.set("event_id", sceneEventFilter.value);
  try {
    const payload = await requestJson(`/api/scenes?${query}`);
    const known = new Set(currentScenes.map((scene) => scene.id));
    currentScenes.push(...(payload.scenes || []).filter((scene) => !known.has(scene.id)));
    currentSceneTotal = payload.total || 0;
    renderSceneReview(currentScenes);
  } finally {
    scenePageLoading = false;
    loadMoreScenes.disabled = false;
  }
}

function renderSceneReview(scenes) {
  sceneGrid.replaceChildren();
  for (const scene of scenes) {
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

    const editFields = document.createElement("div");
    editFields.className = "scene-edit-fields";
    const caption = document.createElement("input");
    caption.value = scene.caption || "";
    caption.placeholder = "Scene caption";
    const transcript = document.createElement("textarea");
    transcript.value = scene.transcript || "";
    transcript.placeholder = "Transcript";
    const landmarks = document.createElement("input");
    landmarks.value = (scene.landmarks || []).join(", ");
    landmarks.placeholder = "Landmarks, comma separated";
    const editButtons = document.createElement("div");
    editButtons.className = "scene-edit-buttons";
    const save = document.createElement("button");
    save.type = "button";
    save.className = "secondary-button";
    save.textContent = "Save text";
    save.addEventListener("click", async () => {
      try {
        const payload = await requestJson(`/api/scenes/${scene.id}`, {
          method: "PATCH",
          body: JSON.stringify({
            input_path: currentJob.input_path,
            workspace: currentJob.workspace,
            expected_version: scene.edit_version,
            caption: caption.value.trim() || null,
            transcript: transcript.value.trim() || null,
            landmarks: commaValues(landmarks.value),
          }),
        });
        Object.assign(scene, payload.scenes[0]);
        renderSceneReview(scenes);
      } catch (error) {
        showError(error.message);
      }
    });
    editButtons.append(save);
    for (const [direction, label] of [[-1, "Move up"], [1, "Move down"]]) {
      const move = document.createElement("button");
      move.type = "button";
      move.className = "secondary-button";
      move.textContent = label;
      move.addEventListener("click", () =>
        reorderScene(scene, direction).catch((error) => showError(error.message)),
      );
      editButtons.append(move);
    }
    editFields.append(caption, transcript, landmarks, editButtons);
    card.append(image, copy, actions, editFields);
    sceneGrid.append(card);
  }
  scenePageStatus.textContent = `${scenes.length} of ${currentSceneTotal} scenes loaded`;
  loadMoreScenes.classList.toggle("hidden", scenes.length >= currentSceneTotal);
  sceneReview.classList.toggle("hidden", scenes.length === 0);
}

async function loadEditingWorkspace() {
  await Promise.all([loadSceneReview(), loadEventReview(), loadTimelineVersions()]);
  editWorkspace.classList.toggle(
    "hidden",
    currentEvents.length === 0 && currentVersions.length === 0,
  );
}

function projectQuery() {
  return new URLSearchParams({
    input_path: currentJob.input_path,
    workspace: currentJob.workspace,
  });
}

async function loadEventReview() {
  const payload = await requestJson(`/api/events?${projectQuery()}`);
  currentEvents = payload.events || [];
  const selectedEvent = sceneEventFilter.value;
  sceneEventFilter.replaceChildren(new Option("All events", ""));
  for (const event of currentEvents) {
    sceneEventFilter.append(new Option(event.title || "Untitled event", event.id));
  }
  sceneEventFilter.value = currentEvents.some((event) => event.id === selectedEvent)
    ? selectedEvent
    : "";
  renderEventReview();
}

function renderEventReview() {
  eventList.replaceChildren();
  currentEvents.forEach((event, index) => {
    const card = document.createElement("article");
    card.className = "event-card";
    const title = document.createElement("input");
    title.value = event.title;
    const summary = document.createElement("textarea");
    summary.value = event.summary || "";
    const landmarks = document.createElement("input");
    landmarks.value = (event.landmarks || []).join(", ");
    landmarks.placeholder = "Landmarks, comma separated";
    const actions = document.createElement("div");
    actions.className = "event-actions";
    const save = document.createElement("button");
    save.type = "button";
    save.className = "secondary-button";
    save.textContent = "Save event";
    save.addEventListener("click", async () => {
      try {
        const payload = await requestJson(`/api/events/${event.id}`, {
          method: "PATCH",
          body: JSON.stringify({
            input_path: currentJob.input_path,
            workspace: currentJob.workspace,
            expected_version: event.edit_version,
            title: title.value.trim(),
            summary: summary.value.trim(),
            landmarks: commaValues(landmarks.value),
          }),
        });
        Object.assign(event, payload.events[0]);
        renderEventReview();
      } catch (error) {
        showError(error.message);
      }
    });
    actions.append(save);
    for (const [direction, label] of [[-1, "Move up"], [1, "Move down"]]) {
      const move = document.createElement("button");
      move.type = "button";
      move.className = "secondary-button";
      move.textContent = label;
      move.disabled = index + direction < 0 || index + direction >= currentEvents.length;
      move.addEventListener("click", () =>
        reorderEvent(index, direction).catch((error) => showError(error.message)),
      );
      actions.append(move);
    }
    card.append(title, summary, landmarks, actions);
    eventList.append(card);
  });
}

async function reorderEvent(index, direction) {
  const ordered = [...currentEvents];
  const target = index + direction;
  [ordered[index], ordered[target]] = [ordered[target], ordered[index]];
  const payload = await requestJson("/api/events/order", {
    method: "PUT",
    body: JSON.stringify({
      input_path: currentJob.input_path,
      workspace: currentJob.workspace,
      ordered_ids: ordered.map((event) => event.id),
      expected_versions: Object.fromEntries(
        currentEvents.map((event) => [event.id, event.edit_version]),
      ),
    }),
  });
  currentEvents = payload.events;
  renderEventReview();
  await loadSceneReview();
}

async function reorderScene(scene, direction) {
  const eventId = scene.metadata?.event_id;
  if (!eventId) {
    showError("Run event detection before reordering scenes.");
    return;
  }
  const eventScenes = await loadAllScenesForEvent(eventId);
  const index = eventScenes.findIndex((item) => item.id === scene.id);
  const target = index + direction;
  if (index < 0 || target < 0 || target >= eventScenes.length) return;
  [eventScenes[index], eventScenes[target]] = [eventScenes[target], eventScenes[index]];
  await requestJson(`/api/events/${eventId}/scenes/order`, {
    method: "PUT",
    body: JSON.stringify({
      input_path: currentJob.input_path,
      workspace: currentJob.workspace,
      ordered_ids: eventScenes.map((item) => item.id),
      expected_versions: Object.fromEntries(
        eventScenes.map((item) => [item.id, item.edit_version]),
      ),
    }),
  });
  await Promise.all([loadSceneReview(), loadEventReview()]);
}

async function loadAllScenesForEvent(eventId) {
  const scenes = [];
  let total = 0;
  do {
    const query = projectQuery();
    query.set("event_id", eventId);
    query.set("offset", String(scenes.length));
    query.set("limit", "500");
    const payload = await requestJson(`/api/scenes?${query}`);
    const page = payload.scenes || [];
    if (page.length === 0 && scenes.length < (payload.total || 0)) {
      throw new Error("The scene list changed while it was being loaded. Refresh and retry.");
    }
    scenes.push(...page);
    total = payload.total || 0;
  } while (scenes.length < total);
  return scenes;
}

async function loadTimelineVersions() {
  const payload = await requestJson(`/api/timeline-versions?${projectQuery()}`);
  currentVersions = payload.versions || [];
  versionBefore.replaceChildren();
  versionAfter.replaceChildren();
  currentVersions.forEach((version, index) => {
    const label = `${version.variant_name} · ${version.phase} · ${new Date(
      version.created_at,
    ).toLocaleString()}`;
    for (const select of [versionBefore, versionAfter]) {
      const option = document.createElement("option");
      option.value = version.id;
      option.textContent = label;
      select.append(option);
    }
    if (index === 1) versionBefore.value = version.id;
    if (index === 0) versionAfter.value = version.id;
  });
  compareVersions.disabled = currentVersions.length < 2;
}

async function compareTimelineVersions() {
  const query = projectQuery();
  query.set("before_id", versionBefore.value);
  query.set("after_id", versionAfter.value);
  const payload = await requestJson(`/api/timeline-versions/compare?${query}`);
  const comparison = payload.comparison;
  versionComparison.textContent =
    `${comparison.selected_scene_ids_added.length} added · ` +
    `${comparison.selected_scene_ids_removed.length} removed · ` +
    `${comparison.order_changes.length} scene moves · ` +
    `${comparison.clip_keys_added.length} clip additions · ` +
    `${comparison.clip_keys_removed.length} clip removals · ` +
    `${comparison.clip_order_changes.length} clip moves · ` +
    `${comparison.clip_changes.length} clip edits · settings: ` +
    `${Object.keys(comparison.settings_changes).join(", ") || "unchanged"} · plan: ` +
    `${Object.keys(comparison.plan_changes).join(", ") || "unchanged"}`;
}

function commaValues(value) {
  return [...new Set(value.split(",").map((item) => item.trim()).filter(Boolean))];
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
refreshEdits.addEventListener("click", () =>
  loadEditingWorkspace().catch((error) => showError(error.message)),
);
compareVersions.addEventListener("click", () =>
  compareTimelineVersions().catch((error) => showError(error.message)),
);
loadMoreScenes.addEventListener("click", () =>
  loadSceneReview(false).catch((error) => showError(error.message)),
);
sceneEventFilter.addEventListener("change", () =>
  loadSceneReview(true).catch((error) => showError(error.message)),
);
browseInputPath.addEventListener("click", () =>
  pickDirectory("input", inputPath, browseInputPath),
);
browseWorkspace.addEventListener("click", () =>
  pickDirectory("workspace", workspace, browseWorkspace),
);
inputPath.addEventListener("input", updateAutomaticWorkspace);
inputPath.addEventListener("change", updateAutomaticWorkspace);
workspace.addEventListener("input", () => {
  workspaceIsAutomatic = workspace.value.trim() === "";
  if (workspaceIsAutomatic) updateAutomaticWorkspace();
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
semanticAnalysis.addEventListener("change", updateDependentCapabilityControls);
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
restoreActiveMovieJob();
