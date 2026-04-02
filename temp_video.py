"""
Deepfake Video Detection API (FastAPI + TensorFlow)
--------------------------------------------------
This service analyzes uploaded videos for deepfake artifacts using a
CNN + LSTM temporal model. It is designed to be consumed by a Node.js
API gateway and returns responses in a gateway-compatible format.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------
# 1. Configuration & Constants
# ---------------------------------------------------------------------

MODEL_PATH = "deepfake_lstm_best.h5"

FRAME_COUNT = 15
IMG_SIZE = 112
CHANNELS = 3

CLASS_NAMES = ["REAL", "FAKE"]
CONFIDENCE_THRESHOLD = 0.85

# ---------------------------------------------------------------------
# 2. VideoDeepfakeDetector Class
# ---------------------------------------------------------------------

class VideoDeepfakeDetector:
    """
    Handles model loading, video preprocessing, inference,
    and forensic explanation generation.
    """

    def __init__(self) -> None:
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[INFO] Loaded model from {MODEL_PATH}")
            print(f"[INFO] Expected input shape: {self.model.input_shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    # ----------------------- Frame Extraction ------------------------

    @staticmethod
    def _extract_frames(video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened() and len(frames) < FRAME_COUNT:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame.astype("float32") / 255.0
            frames.append(frame)

        cap.release()

        if not frames:
            raise ValueError("No frames could be extracted from video")

        while len(frames) < FRAME_COUNT:
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, CHANNELS),
                                    dtype="float32"))

        return np.array(frames)

    # ----------------------- Explanation Logic -----------------------

    @staticmethod
    def _generate_message(pred_class: str, confidence: float) -> str:
        """
        Human-readable explanation aligned with Node.js gateway expectations.
        """
        base = (
            f"The system classified the video as **{pred_class}** "
            f"with confidence **{confidence*100:.2f}%**."
        )

        if pred_class == "REAL":
            if confidence >= CONFIDENCE_THRESHOLD:
                assessment = (
                    "Assessment: High temporal consistency detected. "
                    "Facial motion and frame-to-frame dynamics appear natural."
                )
            else:
                assessment = (
                    "Assessment: Likely authentic, but confidence is limited. "
                    "Minor inconsistencies may be caused by compression or motion blur."
                )
        else:
            if confidence >= CONFIDENCE_THRESHOLD:
                assessment = (
                    "Assessment: Strong deepfake indicators detected. "
                    "Temporal artifacts and unnatural transitions were observed."
                )
            else:
                assessment = (
                    "Assessment: Possible manipulation detected. "
                    "Artifacts are subtle and may indicate a refined deepfake."
                )

        temporal_note = (
            "Temporal analysis was performed using an LSTM to evaluate "
            "sequence-level facial consistency and motion coherence."
        )

        return f"{base}\n\n{assessment}\n\n{temporal_note}"

    # ----------------------- Prediction Pipeline ---------------------

    def predict(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video not found: {file_path}")

        frames = self._extract_frames(file_path)
        input_tensor = np.expand_dims(frames, axis=0)

        prediction = float(self.model.predict(input_tensor, verbose=0)[0][0])

        pred_class = "FAKE" if prediction > 0.5 else "REAL"
        confidence = prediction if pred_class == "FAKE" else 1 - prediction

        message = self._generate_message(pred_class, confidence)

        # 🔑 RESPONSE SHAPE MATCHES index.js EXPECTATIONS
        return {
            "prediction": pred_class,
            "confidence": round(float(confidence), 4),
            "message": message,
            "gradcam_image_url": None,  # videos don't generate Grad-CAM yet
            "class_probabilities": {
                "REAL": round(1 - prediction, 4),
                "FAKE": round(prediction, 4),
            }
        }

# ---------------------------------------------------------------------
# 3. FastAPI Application Setup
# ---------------------------------------------------------------------

app = FastAPI(
    title="Deepfake Video Detector API",
    description="CNN + LSTM based video deepfake detection service.",
    version="1.0.0"
)

try:
    detector = VideoDeepfakeDetector()
except Exception as e:
    print(f"[CRITICAL] {e}")
    detector = None

# ---------------------------------------------------------------------
# 4. API Schema
# ---------------------------------------------------------------------

class VideoRequest(BaseModel):
    file_path: str

# ---------------------------------------------------------------------
# 5. API Endpoint
# ---------------------------------------------------------------------

@app.post("/predict_video", response_class=JSONResponse)
async def predict_video(data: VideoRequest):
    """
    Called by Node.js API Gateway.
    Expects:
        { "file_path": "<absolute/path/to/video>" }
    """
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Video model unavailable."
        )

    try:
        return detector.predict(data.file_path)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))
