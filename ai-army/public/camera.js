// ── CarInspect Camera Module ──
// getUserMedia wrapper with rear-camera preference and file input fallback.

/**
 * Camera controller — manages video stream lifecycle.
 */
const Camera = {
  stream: null,
  videoEl: null,

  /**
   * Start camera stream and attach to video element.
   * Prefers rear-facing camera on mobile.
   * @param {HTMLVideoElement} videoElement
   * @returns {Promise<boolean>} true if getUserMedia succeeded
   */
  async start(videoElement) {
    if (!videoElement) {
      console.error("Camera.start: videoElement is null");
      return false;
    }
    this.videoEl = videoElement;

    // Stop any existing stream first
    this.stop();

    // 모바일: 후면 카메라 선호 / 데스크탑(USB 카메라 등): facingMode 없이 시도
    const isMobile = /Android|iPhone|iPad/i.test(navigator.userAgent);
    const constraints = isMobile
      ? { video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false }
      : { video: { width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      videoElement.srcObject = this.stream;
      await videoElement.play();
      return true;
    } catch (err) {
      console.warn("Camera access failed, retrying with basic constraints:", err.name, err.message);
      // 고해상도 실패 시 기본 설정으로 재시도
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        videoElement.srcObject = this.stream;
        await videoElement.play();
        return true;
      } catch (err2) {
        console.warn("Camera access failed:", err2.name, err2.message);
        this.stream = null;
        return false;
      }
    }
  },

  /**
   * Stop the active camera stream and release resources.
   */
  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach(function (track) {
        track.stop();
      });
      this.stream = null;
    }
    if (this.videoEl) {
      this.videoEl.srcObject = null;
    }
  },

  /**
   * Capture current video frame as Blob + dataURL.
   * @returns {Promise<{blob: Blob, dataUrl: string}|null>}
   */
  async capture() {
    if (!this.videoEl || !this.stream) {
      return null;
    }

    const video = this.videoEl;
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;

    if (width === 0 || height === 0) {
      return null;
    }

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");

    if (!ctx) return null;

    ctx.drawImage(video, 0, 0, width, height);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.85);

    return new Promise(function (resolve) {
      canvas.toBlob(
        function (blob) {
          if (!blob) {
            resolve(null);
            return;
          }
          resolve({ blob: blob, dataUrl: dataUrl });
        },
        "image/jpeg",
        0.85
      );
    });
  },

  /**
   * Check if getUserMedia is supported.
   * @returns {boolean}
   */
  isSupported() {
    return !!(
      navigator.mediaDevices && navigator.mediaDevices.getUserMedia
    );
  }
};

/**
 * Convert a File from <input type="file"> to {blob, dataUrl}.
 * Applies resizing to keep images manageable.
 * @param {File} file
 * @returns {Promise<{blob: Blob, dataUrl: string}>}
 */
async function fileToImageData(file) {
  if (!file || file.size === 0) {
    throw new Error("파일이 비어있습니다.");
  }

  // Validate file type
  if (!file.type.startsWith("image/")) {
    throw new Error("이미지 파일만 업로드 가능합니다.");
  }

  // Resize if needed
  const resizedBlob = await resizeImageBlob(file, 1920);

  return new Promise(function (resolve, reject) {
    const reader = new FileReader();
    reader.onload = function (e) {
      resolve({
        blob: resizedBlob,
        dataUrl: e.target.result
      });
    };
    reader.onerror = function () {
      reject(new Error("파일 읽기 실패"));
    };
    reader.readAsDataURL(resizedBlob);
  });
}