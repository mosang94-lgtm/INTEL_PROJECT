// ── CarInspect Data & Configuration ──
// 등급 산정 룰, 부품/불량 클래스, 유틸리티 함수

// 등급 산정 룰 (하드코딩 금지 — 이 객체에서만 읽음)
const GRADING_RULES = {
  classWeights: {
    "스크래치": 0.3,
    "외관손상": 0.5,
    "단차": 0.7,
    "장착불량": 0.6,
    "고정불량": 0.7,
    "고정핀불량": 0.7,
    "연계불량": 0.6,
    "유격불량": 0.6,
    "체결불량": 0.9,
    "실링불량": 0.8,
    "헤밍불량": 0.8,
    "홀변형": 1.0
  },
  gradeThresholds: [
    { max: 0.2, grade: "A", label: "양호" },
    { max: 0.4, grade: "B", label: "경미" },
    { max: 0.7, grade: "C", label: "주의" },
    { max: 1.0, grade: "D", label: "심각" }
  ]
};

const PART_CLASSES = [
  "도어", "라디에이터그릴", "루프사이드", "배선", "범퍼",
  "카울커버", "커넥터", "테일램프", "프레임", "헤드램프", "휀더"
];

const DEFECT_TYPES = [
  "스크래치", "외관손상", "단차", "장착불량", "고정불량", "고정핀불량",
  "연계불량", "유격불량", "체결불량", "실링불량", "헤밍불량", "홀변형"
];

const MOCK_MODEL = {
  model_id: "carinspect-v1-efficientdet-d3",
  model_version: "1.0.0-mock"
};

const GRADE_ICONS = { A: "✅", B: "⚠️", C: "🔶", D: "🔴" };
const GRADE_LABELS = { A: "양호", B: "경미", C: "주의", D: "심각" };

const STORAGE_KEYS = {
  history: "carinspect_history",
  settings: "carinspect_settings"
};

const MAX_HISTORY_ITEMS = 50;

// ── Utility Functions ──

function generateUUID() {
  if (crypto && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randFloat(min, max) {
  return min + Math.random() * (max - min);
}

/**
 * Calculate severity score and overall grade from detections.
 * Uses GRADING_RULES — no hardcoded values.
 */
function calculateGrade(detections) {
  if (!detections || detections.length === 0) {
    return { severity_score: 0.0, overall_grade: "A" };
  }

  const weights = GRADING_RULES.classWeights;
  let weightedSum = 0;

  for (const det of detections) {
    const classWeight = weights[det.class_name] || 0.5;
    weightedSum += det.confidence * classWeight;
  }

  const severityScore = Math.min(1.0, weightedSum / detections.length);
  let overallGrade = "D";

  for (const threshold of GRADING_RULES.gradeThresholds) {
    if (severityScore <= threshold.max) {
      overallGrade = threshold.grade;
      break;
    }
  }

  return {
    severity_score: Math.round(severityScore * 1000) / 1000,
    overall_grade: overallGrade
  };
}

/**
 * Generate a random bbox within valid bounds.
 * Returns [x1, y1, x2, y2] normalized 0~1.
 */
function generateRandomBbox() {
  const x1 = randFloat(0.05, 0.6);
  const y1 = randFloat(0.05, 0.6);
  const width = randFloat(0.1, 0.35);
  const height = randFloat(0.1, 0.35);
  const x2 = Math.min(0.95, x1 + width);
  const y2 = Math.min(0.95, y1 + height);
  return [
    Math.round(x1 * 1000) / 1000,
    Math.round(y1 * 1000) / 1000,
    Math.round(x2 * 1000) / 1000,
    Math.round(y2 * 1000) / 1000
  ];
}

/**
 * Format ISO date string to human-readable Korean format.
 */
function formatDate(isoString) {
  try {
    const d = new Date(isoString);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    const hour = String(d.getHours()).padStart(2, "0");
    const min = String(d.getMinutes()).padStart(2, "0");
    return `${year}.${month}.${day} ${hour}:${min}`;
  } catch {
    return isoString;
  }
}

/**
 * Resize image blob to max dimension, returns new Blob (JPEG 80%).
 */
async function resizeImageBlob(blob, maxDimension = 1920) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(blob);

    img.onload = function () {
      URL.revokeObjectURL(url);
      let { width, height } = img;

      if (width <= maxDimension && height <= maxDimension) {
        resolve(blob);
        return;
      }

      const ratio = Math.min(maxDimension / width, maxDimension / height);
      width = Math.round(width * ratio);
      height = Math.round(height * ratio);

      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, width, height);

      canvas.toBlob(
        function (resizedBlob) {
          resolve(resizedBlob || blob);
        },
        "image/jpeg",
        0.8
      );
    };

    img.onerror = function () {
      URL.revokeObjectURL(url);
      reject(new Error("Image load failed during resize"));
    };

    img.src = url;
  });
}

// ── localStorage Helpers ──

function loadHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.history);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveHistory(history) {
  try {
    // FIFO: keep only latest MAX_HISTORY_ITEMS
    const trimmed = history.slice(0, MAX_HISTORY_ITEMS);
    localStorage.setItem(STORAGE_KEYS.history, JSON.stringify(trimmed));
  } catch (e) {
    // localStorage full — remove oldest 10 and retry
    try {
      const trimmed = history.slice(0, MAX_HISTORY_ITEMS - 10);
      localStorage.setItem(STORAGE_KEYS.history, JSON.stringify(trimmed));
    } catch {
      console.warn("localStorage save failed:", e);
    }
  }
}

function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.settings);
    if (!raw) return { apiUrl: "" };
    return JSON.parse(raw);
  } catch {
    return { apiUrl: "" };
  }
}

function saveSettings(settings) {
  try {
    localStorage.setItem(STORAGE_KEYS.settings, JSON.stringify(settings));
  } catch (e) {
    console.warn("Settings save failed:", e);
  }
}