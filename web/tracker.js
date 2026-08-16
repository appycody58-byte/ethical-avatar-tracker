/**
 * Ethical Avatar Tracker - Browser version
 * Optimized for WebRTC video calling applications.
 * Uses MediaPipe Tasks Face Landmarker.
 */

import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

export class WebFaceTracker {
  constructor() {
    this.faceLandmarker = null;
    this.lastVideoTime = -1;
    this.running = false;
    this.onResults = null;
  }

  async init(modelUrl = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task") {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelUrl,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
  }

  async start(videoElement, onResultsCallback) {
    if (!this.faceLandmarker) {
      await this.init();
    }
    this.onResults = onResultsCallback;
    this.running = true;

    const processFrame = () => {
      if (!this.running) return;

      if (videoElement.readyState >= 2 && videoElement.currentTime !== this.lastVideoTime) {
        this.lastVideoTime = videoElement.currentTime;
        const results = this.faceLandmarker.detectForVideo(videoElement, performance.now());

        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
          const payload = {
            landmarks: results.faceLandmarks[0],
            blendshapes: results.faceBlendshapes?.[0]?.categories || [],
            matrix: results.facialTransformationMatrixes?.[0] || null,
            timestamp: performance.now(),
          };
          if (this.onResults) this.onResults(payload);
        }
      }
      requestAnimationFrame(processFrame);
    };

    processFrame();
  }

  stop() {
    this.running = false;
  }
}

// Example usage (uncomment in your page):
/*
const video = document.getElementById("webcam");
navigator.mediaDevices.getUserMedia({ video: true, audio: false })
  .then((stream) => {
    video.srcObject = stream;
    video.play();

    const tracker = new WebFaceTracker();
    tracker.start(video, (data) => {
      console.log("Blendshapes:", data.blendshapes);
      // Send over WebRTC data channel or drive your 3D avatar here
    });
  });
*/
