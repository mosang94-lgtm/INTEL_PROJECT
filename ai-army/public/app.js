// ── CarInspect Main Application ──
// SPA router, state management, event delegation, page rendering.

// ── State ──
const state = {
  currentPage: "capture",    // "capture" | "result" | "history" | "settings"
  isInspecting: false,
  cameraActive: false,
  lastResult: null,          // InspectionResult from API
  lastImageUrl: null,        // data URL of captured/uploaded image
  history: [],               // loaded from localStorage
  settings: { apiUrl: "" }
};

// ── DOM References ──
const $ = function (sel) { return document.querySelector(sel); };
const $$ = function (sel) { return document.querySelectorAll(sel); };

// ── Init ──
function init() {
  state.history = loadHistory();
  state.settings = loadSettings();

  bindNavigation();
  bindPageEvents();
  navigateTo("capture");
}

// ── Navigation ──
function bindNavigation() {
  const nav = $(".bottom-nav");
  if (!nav) return;

  nav.addEventListener("click", function (e) {
    const btn = e.target.closest("[data-page]");
    if (!btn) return;
    e.preventDefault();
    const page = btn.getAttribute("data-page");
    if (page) navigateTo(page);
  });
}

function navigateTo(page) {
  // Stop camera when leaving capture page
  if (state.currentPage === "capture" && page !== "capture") {
    Camera.stop();
    state.cameraActive = false;
  }

  state.currentPage = page;

  // Update nav active state
  $$(".bottom-nav [data-page]").forEach(function (btn) {
    btn.classList.toggle("active", btn.getAttribute("data-page") === page);
  });

  // Show/hide pages
  $$(".page").forEach(function (el) {
    el.classList.toggle("hidden", el.id !== "page-" + page);
  });

  // Page-specific init
  if (page === "capture") initCapturePage();
  if (page === "result") renderResultPage();
  if (page === "history") renderHistoryPage();
  if (page === "settings") renderSettingsPage();
}

// ── Capture Page ──
function initCapturePage() {
  const video = $("#camera-video");
  const fallback = $("#file-fallback");
  const captureBtn = $("#btn-capture");
  const statusEl = $("#camera-status");

  if (!video) return;

  if (Camera.isSupported()) {
    if (statusEl) statusEl.textContent = "카메라 연결 중...";
    Camera.start(video).then(function (ok) {
      state.cameraActive = ok;
      if (ok) {
        video.classList.remove("hidden");
        if (fallback) fallback.classList.add("hidden");
        if (captureBtn) captureBtn.disabled = false;
        if (statusEl) statusEl.textContent = "";
      } else {
        video.classList.add("hidden");
        if (fallback) fallback.classList.remove("hidden");
        if (captureBtn) captureBtn.disabled = true;
        if (statusEl) statusEl.textContent = "카메라 접근 불가 — 파일을 선택해주세요";
      }
    });
  } else {
    video.classList.add("hidden");
    if (fallback) fallback.classList.remove("hidden");
    if (captureBtn) captureBtn.disabled = true;
    if (statusEl) statusEl.textContent = "이 브라우저는 카메라를 지원하지 않습니다";
  }
}

function bindPageEvents() {
  // Capture button
  const captureBtn = $("#btn-capture");
  if (captureBtn) {
    captureBtn.addEventListener("click", handleCapture);
  }

  // File input fallback
  const fileInput = $("#file-input");
  if (fileInput) {
    fileInput.addEventListener("change", handleFileSelect);
  }

  // Result page: re-inspect / save
  const resultArea = $("#page-result");
  if (resultArea) {
    resultArea.addEventListener("click", handleResultActions);
  }

  // History page: delegation
  const historyArea = $("#page-history");
  if (historyArea) {
    historyArea.addEventListener("click", handleHistoryActions);
  }

  // Settings page: save
  const settingsForm = $("#settings-form");
  if (settingsForm) {
    settingsForm.addEventListener("submit", handleSettingsSave);
  }
}

async function handleCapture() {
  if (state.isInspecting) return;

  try {
    const result = await Camera.capture();
    if (!result) {
      showToast("촬영 실패. 다시 시도해주세요.", "error");
      return;
    }
    await processInspection(result.blob, result.dataUrl);
  } catch (err) {
    showToast("촬영 오류: " + err.message, "error");
  }
}

async function handleFileSelect(e) {
  const file = e.target.files && e.target.files[0];
  if (!file) return;

  try {
    const imgData = await fileToImageData(file);
    await processInspection(imgData.blob, imgData.dataUrl);
  } catch (err) {
    showToast("파일 처리 오류: " + err.message, "error");
  }

  // Reset input so same file can be selected again
  e.target.value = "";
}

async function processInspection(blob, dataUrl) {
  state.isInspecting = true;
  showLoading(true);

  try {
    const resizedBlob = await resizeImageBlob(blob, 1920);
    const result = await inspectImage(resizedBlob, dataUrl);

    state.lastResult = result;
    state.lastImageUrl = dataUrl;

    // Auto-save to history
    addToHistory(result);

    navigateTo("result");
  } catch (err) {
    showToast("검사 실패: " + err.message, "error");
  } finally {
    state.isInspecting = false;
    showLoading(false);
  }
}

// ── Result Page ──
function renderResultPage() {
  const container = $("#result-content");
  if (!container || !state.lastResult) {
    if (container) container.innerHTML = '<p class="empty-state">검사 결과가 없습니다. 촬영 먼저 해주세요.</p>';
    return;
  }

  const r = state.lastResult;
  const grade = r.overall_grade;
  const icon = GRADE_ICONS[grade] || "❓";
  const label = GRADE_LABELS[grade] || "알 수 없음";

  let html = '<div class="result-card">';

  // Image with overlay canvas
  html += '<div class="result-image-wrap">';
  html += '<img id="result-img" src="' + escapeAttr(r.image_url) + '" alt="검사 이미지" />';
  html += '<canvas id="overlay-canvas"></canvas>';
  html += '</div>';

  // Grade card
  html += '<div class="grade-card grade-' + grade + '">';
  html += '<span class="grade-icon">' + icon + '</span>';
  html += '<div class="grade-info">';
  html += '<span class="grade-letter">' + grade + '등급</span>';
  html += '<span class="grade-label">' + label + '</span>';
  html += '</div>';
  html += '<div class="grade-score">심각도 ' + (r.severity_score * 100).toFixed(1) + '%</div>';
  html += '</div>';

  // Metadata
  html += '<div class="result-meta">';
  html += '<span>모델: ' + escapeHtml(r.model_id) + '</span>';
  html += '<span>추론: ' + r.inference_ms + 'ms</span>';
  html += '<span>' + formatDate(r.created_at) + '</span>';
  html += '</div>';

  // Detections list
  if (r.detections.length === 0) {
    html += '<div class="detection-empty">결함이 발견되지 않았습니다 ✨</div>';
  } else {
    html += '<div class="detection-list">';
    html += '<h3>발견된 결함 (' + r.detections.length + '건)</h3>';
    r.detections.forEach(function (det, i) {
      const conf = (det.confidence * 100).toFixed(0);
      html += '<div class="detection-item">';
      html += '<span class="det-index">' + (i + 1) + '</span>';
      html += '<span class="det-class">' + escapeHtml(det.class_name) + '</span>';
      if (det.part_name) {
        html += '<span class="det-part">' + escapeHtml(det.part_name) + '</span>';
      }
      html += '<span class="det-conf">' + conf + '%</span>';
      html += '</div>';
    });
    html += '</div>';
  }

  // Actions
  html += '<div class="result-actions">';
  html += '<button class="btn btn-primary" data-action="new-inspect">새 검사</button>';
  html += '</div>';

  html += '</div>';
  container.innerHTML = html;

  // Draw bbox overlay after image loads
  const img = $("#result-img");
  if (img) {
    if (img.complete) {
      drawOverlay(r.detections);
    } else {
      img.addEventListener("load", function () {
        drawOverlay(r.detections);
      });
    }
  }
}

function drawOverlay(detections) {
  const canvas = $("#overlay-canvas");
  const img = $("#result-img");
  if (!canvas || !img) return;

  const rect = img.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  canvas.style.width = rect.width + "px";
  canvas.style.height = rect.height + "px";

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const colors = [
    "#FF4444", "#FF8800", "#FFCC00", "#44BB44",
    "#4488FF", "#8844FF", "#FF44AA", "#00CCCC",
    "#FF6644", "#88CC00", "#CC44FF", "#44CCFF"
  ];

  detections.forEach(function (det, i) {
    if (!det.bbox || det.bbox.length < 4) return;

    const x1 = det.bbox[0] * canvas.width;
    const y1 = det.bbox[1] * canvas.height;
    const x2 = det.bbox[2] * canvas.width;
    const y2 = det.bbox[3] * canvas.height;
    const color = colors[det.class_id % colors.length];

    // Draw bbox
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Draw label background
    const labelText = det.class_name + " " + (det.confidence * 100).toFixed(0) + "%";
    ctx.font = "bold 12px sans-serif";
    const textWidth = ctx.measureText(labelText).width;
    ctx.fillStyle = color;
    ctx.fillRect(x1, y1 - 18, textWidth + 8, 18);

    // Draw label text
    ctx.fillStyle = "#FFFFFF";
    ctx.fillText(labelText, x1 + 4, y1 - 5);
  });
}

function handleResultActions(e) {
  const btn = e.target.closest("[data-action]");
  if (!btn) return;

  var action = btn.getAttribute("data-action");
  if (action === "new-inspect") {
    navigateTo("capture");
  }
}

// ── History ──
function addToHistory(result) {
  state.history.unshift(result);
  if (state.history.length > MAX_HISTORY_ITEMS) {
    state.history = state.history.slice(0, MAX_HISTORY_ITEMS);
  }
  saveHistory(state.history);
}

function renderHistoryPage() {
  var container = $("#history-content");
  if (!container) return;

  if (state.history.length === 0) {
    container.innerHTML = '<p class="empty-state">검사 이력이 없습니다.<br>카메라로 부품을 촬영해보세요.</p>';
    return;
  }

  var html = '<div class="history-header">';
  html += '<h3>검사 이력 (' + state.history.length + '건)</h3>';
  html += '<button class="btn btn-sm btn-danger" data-action="clear-history">전체 삭제</button>';
  html += '</div>';
  html += '<div class="history-list">';

  state.history.forEach(function (item, i) {
    var grade = item.overall_grade || "?";
    var icon = GRADE_ICONS[grade] || "❓";
    html += '<div class="history-item" data-action="view-history" data-index="' + i + '">';
    html += '<div class="history-thumb">';
    if (item.image_url) {
      html += '<img src="' + escapeAttr(item.image_url) + '" alt="검사 이미지" loading="lazy" />';
    }
    html += '</div>';
    html += '<div class="history-info">';
    html += '<span class="history-grade grade-' + grade + '">' + icon + ' ' + grade + '등급</span>';
    html += '<span class="history-defects">결함 ' + (item.detections ? item.detections.length : 0) + '건</span>';
    html += '<span class="history-date">' + formatDate(item.created_at) + '</span>';
    html += '</div>';
    html += '</div>';
  });

  html += '</div>';
  container.innerHTML = html;
}

function handleHistoryActions(e) {
  var btn = e.target.closest("[data-action]");
  if (!btn) return;

  var action = btn.getAttribute("data-action");

  if (action === "clear-history") {
    if (confirm("전체 검사 이력을 삭제하시겠습니까?")) {
      state.history = [];
      saveHistory(state.history);
      renderHistoryPage();
      showToast("이력이 삭제되었습니다.", "success");
    }
  }

  if (action === "view-history") {
    var index = parseInt(btn.getAttribute("data-index"), 10);
    if (!isNaN(index) && state.history[index]) {
      state.lastResult = state.history[index];
      state.lastImageUrl = state.history[index].image_url;
      navigateTo("result");
    }
  }
}

// ── Settings Page ──
function renderSettingsPage() {
  var urlInput = $("#settings-api-url");
  var modeLabel = $("#settings-mode");

  if (urlInput) urlInput.value = state.settings.apiUrl || "";
  if (modeLabel) {
    modeLabel.textContent = isMockMode() ? "🟡 Mock 모드" : "🟢 서버 연결";
  }
}

function handleSettingsSave(e) {
  e.preventDefault();
  var urlInput = $("#settings-api-url");
  if (!urlInput) return;

  state.settings.apiUrl = urlInput.value.trim();
  saveSettings(state.settings);
  renderSettingsPage();
  showToast("설정이 저장되었습니다.", "success");
}

// ── UI Utilities ──
function showLoading(visible) {
  var overlay = $("#loading-overlay");
  if (overlay) {
    overlay.classList.toggle("hidden", !visible);
  }
}

function showToast(message, type) {
  var container = $("#toast-container");
  if (!container) return;

  var toast = document.createElement("div");
  toast.className = "toast toast-" + (type || "info");
  toast.textContent = message;
  toast.setAttribute("role", "alert");
  container.appendChild(toast);

  // Trigger reflow for animation
  toast.offsetHeight;
  toast.classList.add("show");

  setTimeout(function () {
    toast.classList.remove("show");
    setTimeout(function () {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 300);
  }, 3000);
}

function escapeHtml(str) {
  var div = document.createElement("div");
  div.textContent = str || "";
  return div.innerHTML;
}

function escapeAttr(str) {
  return (str || "").replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/'/g, "&#39;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ── Bootstrap ──
document.addEventListener("DOMContentLoaded", init);