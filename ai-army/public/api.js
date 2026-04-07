// ── CarInspect API Layer ──
// Mock API that mirrors FastAPI /inspect endpoint response format.
// To switch to real server: change API_BASE_URL in settings.

/**
 * Check if running in mock mode (no server URL configured).
 */
function isMockMode() {
  const settings = loadSettings();
  return !settings.apiUrl || settings.apiUrl.trim() === "";
}

/**
 * Get configured API base URL.
 */
function getApiBaseUrl() {
  const settings = loadSettings();
  return (settings.apiUrl || "").trim().replace(/\/+$/, "");
}

/**
 * Delay helper for mock latency simulation.
 */
function delay(ms) {
  return new Promise(function (resolve) {
    setTimeout(resolve, ms);
  });
}

/**
 * Generate mock detection data.
 * 20% chance of zero defects (good part), otherwise 1-4 defects.
 */
function generateMockDetections() {
  const isGoodPart = Math.random() < 0.2;
  if (isGoodPart) return [];

  const count = randInt(1, 4);
  const detections = [];

  for (let i = 0; i < count; i++) {
    const classId = randInt(0, DEFECT_TYPES.length - 1);
    detections.push({
      class_id: classId,
      class_name: DEFECT_TYPES[classId],
      part_name: PART_CLASSES[randInt(0, PART_CLASSES.length - 1)],
      confidence: Math.round(randFloat(0.5, 0.99) * 100) / 100,
      bbox: generateRandomBbox(),
      polygon: null
    });
  }

  return detections;
}

/**
 * Generate a complete mock InspectionResult.
 */
function generateMockResult(imageUrl) {
  const detections = generateMockDetections();
  const { severity_score, overall_grade } = calculateGrade(detections);

  return {
    inspection_id: generateUUID(),
    model_id: MOCK_MODEL.model_id,
    model_version: MOCK_MODEL.model_version,
    overall_grade: overall_grade,
    severity_score: severity_score,
    inference_ms: 300 + Math.round(Math.random() * 400),
    detections: detections,
    image_url: imageUrl,
    overlay_url: null,
    created_at: new Date().toISOString()
  };
}

/**
 * Main API function: inspect an image for defects.
 * In mock mode: generates fake data after simulated delay.
 * In real mode: POSTs to FastAPI /inspect endpoint.
 *
 * @param {Blob} blob - Image blob to inspect
 * @param {string} imageDataUrl - Base64 data URL of image (for local display)
 * @returns {Promise<Object>} InspectionResult
 */
async function inspectImage(blob, imageDataUrl) {
  if (!blob || blob.size === 0) {
    throw new Error("빈 이미지입니다. 다시 촬영해주세요.");
  }

  if (isMockMode()) {
    // Mock mode: simulate network delay
    await delay(300 + Math.random() * 400);
    return generateMockResult(imageDataUrl);
  }

  // Real server mode
  const baseUrl = getApiBaseUrl();
  const formData = new FormData();
  formData.append("image", blob, "capture.jpg");

  try {
    const response = await fetch(baseUrl + "/inspect", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text().catch(function () {
        return "Unknown error";
      });
      throw new Error(
        "서버 오류 (" + response.status + "): " + errorText
      );
    }

    const result = await response.json();

    // Ensure required fields exist (defensive)
    if (!result.model_id || !result.model_version) {
      result.model_id = result.model_id || "unknown";
      result.model_version = result.model_version || "unknown";
    }
    if (!result.inspection_id) {
      result.inspection_id = generateUUID();
    }
    if (!result.created_at) {
      result.created_at = new Date().toISOString();
    }

    // Store local image reference for display
    result.image_url = result.image_url || imageDataUrl;

    return result;
  } catch (e) {
    if (e.name === "TypeError" && e.message.includes("fetch")) {
      throw new Error(
        "서버에 연결할 수 없습니다. 네트워크를 확인해주세요."
      );
    }
    throw e;
  }
}

/**
 * Fetch registered models from server.
 * In mock mode: returns single mock model.
 */
async function fetchModels() {
  if (isMockMode()) {
    await delay(100);
    return [
      {
        model_id: MOCK_MODEL.model_id,
        version: MOCK_MODEL.model_version,
        task: "object_detection",
        is_default: true
      }
    ];
  }

  const baseUrl = getApiBaseUrl();
  const response = await fetch(baseUrl + "/models");
  if (!response.ok) throw new Error("모델 목록 조회 실패");
  return response.json();
}

/**
 * Health check.
 */
async function healthCheck() {
  if (isMockMode()) {
    return { status: "ok", mode: "mock", model_id: MOCK_MODEL.model_id };
  }

  const baseUrl = getApiBaseUrl();
  const response = await fetch(baseUrl + "/healthz");
  if (!response.ok) throw new Error("Health check failed");
  return response.json();
}