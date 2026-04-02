"""
Deepfake Detection API - Video Server (FastAPI + TensorFlow)
------------------------------------------------------------
This service loads a TensorFlow/Keras LSTM-based deepfake detector,
performs predictions on uploaded video files, generates Grad-CAM
heatmaps for interpretability, and serves results via FastAPI.

Run with:
    python video_server.py
    or
    uvicorn video_server:app --host 127.0.0.1 --port 8080 --reload
"""

import os
import cv2
import uuid
import shutil
import numpy as np
import tensorflow as tf
import uvicorn
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


# ---------------------------------------------------------------------
# 1. Configuration & Constants
# ---------------------------------------------------------------------

MODEL_PATH = r"D:\deeepVerify\model_video\deepfake_detection_model.h5"

RESULTS_DIR = os.path.join(os.getcwd(), "results_video")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 224
SEQUENCE_LENGTH = 20
CONFIDENCE_THRESHOLD = 0.85


# ---------------------------------------------------------------------
# 2. VideoDeepfakeDetector Class
# ---------------------------------------------------------------------

class VideoDeepfakeDetector:
    """
    Encapsulates the Keras video model, frame preprocessing pipeline,
    and Grad-CAM logic for deepfake video detection and explanation.
    """

    def __init__(self) -> None:
        """Load the model, identify the Grad-CAM target layer, and warm up."""
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print(f"[INFO] TensorFlow model loaded from: {MODEL_PATH}")

            self.target_layer_name = self._find_target_layer()
            print(f"[INFO] Grad-CAM target layer: {self.target_layer_name}")

            # Warm-up: forces Keras to initialize the functional graph
            dummy_input = np.zeros(
                (1, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32
            )
            self.model.predict(dummy_input, verbose=0)
            print("[INFO] Model warmed up successfully.")

        except Exception as e:
            print(f"[CRITICAL] Failed to load model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}") from e

    # ----------------------- Layer Discovery -------------------------

    def _find_target_layer(self) -> str:
        """
        Find the best layer for Grad-CAM.
        NOTE: For TimeDistributed(Sequential) models we use input-gradient
        saliency instead of activation-based Grad-CAM, so this method is
        kept only for informational logging purposes.
        """
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.TimeDistributed):
                return layer.name
            if "conv" in layer.name.lower():
                return layer.name
        return self.model.layers[-2].name

    # ----------------------- Preprocessing ---------------------------

    def _preprocess_video(self, video_path: str):
        """
        Sample SEQUENCE_LENGTH evenly-spaced frames from the video,
        resize to IMG_SIZE × IMG_SIZE, and normalize to [0, 1].

        Returns:
            input_tensor: shape (1, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)
            raw_frames:   list of uint8 RGB frames for visualization
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1  # Guard against corrupt/empty videos

        # Evenly-spaced frame indices
        indices = np.linspace(
            0, max(0, total_frames - 1), SEQUENCE_LENGTH, dtype=int
        )

        raw_frames, normalized_frames = [], []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            if not ret or frame is None:
                # Pad with blank frame if read fails
                frame_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))

            raw_frames.append(frame_rgb)
            normalized_frames.append(frame_rgb.astype(np.float32) / 255.0)

        cap.release()

        input_tensor = np.expand_dims(
            np.array(normalized_frames, dtype=np.float32), axis=0
        )  # Shape: (1, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)

        return input_tensor, raw_frames

    # ----------- Input-Gradient Saliency (replaces Grad-CAM) -----------
    # WHY: TimeDistributed(Sequential(...)) models don't expose intermediate
    # layer output tensors through model.get_layer(...).output because the
    # inner Sequential sub-model has no symbolic output until it's built as
    # a standalone functional graph. Input-gradient saliency avoids this
    # entirely — it computes gradients of the prediction score with respect
    # to the raw input pixels, which always works regardless of architecture.

    def _get_gradcam(self, input_tensor: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Generate an input-gradient saliency map for the chosen frame.

        Computes d(fake_score) / d(input_pixels) for the full video tensor,
        then extracts and post-processes the slice for `frame_idx`.

        Args:
            input_tensor: preprocessed video numpy array (1, T, H, W, C)
            frame_idx:    which temporal frame to visualize

        Returns:
            heatmap: 2-D float32 array in [0, 1], shape (IMG_SIZE, IMG_SIZE)
        """
        try:
            input_tf = tf.Variable(
                tf.cast(input_tensor, tf.float32), trainable=True
            )

            with tf.GradientTape() as tape:
                predictions = self.model(input_tf, training=False)
                # Score for the FAKE class (index 0 = fake, sigmoid output)
                loss = predictions[:, 0]

            grads = tape.gradient(loss, input_tf)  # shape: (1, T, H, W, C)

            if grads is None:
                print("[WARN] Gradients are None — returning blank heatmap.")
                return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

            # Extract gradients for the target frame: (H, W, C)
            frame_grads = grads[0, frame_idx].numpy()

            # Aggregate channels: take absolute value then mean across channels
            # This produces a (H, W) importance map
            heatmap = np.mean(np.abs(frame_grads), axis=-1)

            # Apply mild Gaussian blur to smooth the saliency map
            heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)

            # Normalize to [0, 1]
            min_val, max_val = heatmap.min(), heatmap.max()
            if max_val - min_val > 1e-8:
                heatmap = (heatmap - min_val) / (max_val - min_val)
            else:
                heatmap = np.zeros_like(heatmap)

            return heatmap.astype(np.float32)

        except Exception as e:
            print(f"[WARN] Saliency computation failed: {e}. Returning blank heatmap.")
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # ----------------------- Visualization ---------------------------

    def _save_gradcam_image(
        self,
        raw_frame: np.ndarray,
        heatmap: np.ndarray,
        label: str,
        confidence: float,
    ) -> str:
        """
        Overlay the Grad-CAM heatmap on the raw frame and save a
        side-by-side comparison image (original | overlay).

        Returns:
            URL string pointing to the saved result image.
        """
        # Resize heatmap to match frame
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend overlay: 60% original, 40% heatmap
        overlay = cv2.addWeighted(raw_frame, 0.6, heatmap_colored_rgb, 0.4, 0)

        # Combine side by side
        combined = np.hstack((raw_frame, overlay))  # (H, 2W, 3)

        timestamp = datetime.now().strftime("%H%M%S_%f")[:10]
        fname = f"vid_{label}_{int(confidence * 100)}_{timestamp}.jpg"
        save_path = os.path.join(RESULTS_DIR, fname)

        success = cv2.imwrite(
            save_path,
            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        if not success:
            raise IOError(f"Failed to write result image to: {save_path}")

        return f"http://127.0.0.1:8080/results/{fname}"

    # ---------------------- Explanation Builder ----------------------

    def _generate_explanation(
        self, label: str, confidence: float, mean_heatmap: float
    ) -> str:
        """Build a human-readable forensic rationale string."""
        base = (
            f"The DeepVerify Video model classified the footage as "
            f"**{label.upper()}** with confidence **{confidence * 100:.2f}%**."
        )

        if label == "REAL":
            if confidence >= CONFIDENCE_THRESHOLD:
                assessment = (
                    "Assessment: **High assurance of authenticity**. "
                    "Natural temporal patterns and facial consistency detected throughout the sequence."
                )
            else:
                assessment = (
                    "Assessment: **Probable authenticity (caution advised)**. "
                    "Minor compression or encoding artifacts may have reduced model certainty."
                )
        else:  # FAKE
            if confidence >= CONFIDENCE_THRESHOLD:
                assessment = (
                    "Assessment: **Clear synthetic manipulation detected**. "
                    "Strong anomalies found in facial structures and/or temporal flow."
                )
            else:
                assessment = (
                    "Assessment: **Subtle artifacts detected (inconclusive)**. "
                    "Possible refined deepfake or degraded authentic media — manual review recommended."
                )

        localization = (
            f"**Grad-CAM focus:** Forensic attention concentrated on high-heatmap regions "
            f"(average intensity: {mean_heatmap:.3f}), representing the features most influential to the decision."
        )

        return f"{base}\n\n{assessment}\n\n{localization}"

    # ----------------------- Main Prediction -------------------------

    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Full prediction pipeline:
          1. Preprocess video into a frame sequence tensor
          2. Run model inference
          3. Generate Grad-CAM on the middle frame
          4. Save visualization
          5. Return structured result dict

        Args:
            video_path: path to the video file on disk

        Returns:
            dict with keys: prediction, confidence, message, gradcam_image_url
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Step 1 – Preprocess
        input_tensor, raw_frames = self._preprocess_video(video_path)

        # Step 2 – Inference
        preds = self.model.predict(input_tensor, verbose=0)
        score = float(preds[0][0])  # Probability of FAKE class

        label = "FAKE" if score > 0.5 else "REAL"
        confidence = float(score if label == "FAKE" else 1.0 - score)

        # Step 3 – Grad-CAM on middle frame
        mid_idx = len(raw_frames) // 2
        heatmap = self._get_gradcam(input_tensor, frame_idx=mid_idx)
        mean_heatmap = float(np.mean(heatmap))

        # Step 4 – Save visualization
        gradcam_url = self._save_gradcam_image(
            raw_frames[mid_idx], heatmap, label, confidence
        )

        # Step 5 – Build and return result
        explanation = self._generate_explanation(label, confidence, mean_heatmap)

        return {
            "prediction": label,
            "confidence": confidence,
            "message": explanation,
            "gradcam_image_url": gradcam_url,
        }


# ---------------------------------------------------------------------
# 3. FastAPI Application Setup
# ---------------------------------------------------------------------

app = FastAPI(
    title="DeepVerify Video Deepfake Detection API",
    description=(
        "Detects deepfake artifacts in video files using a TensorFlow/Keras "
        "sequence model with Grad-CAM explainability."
    ),
    version="1.1.0",
)

# Serve the results directory as static files
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# Load the detector at startup (once, globally)
try:
    detector = VideoDeepfakeDetector()
except RuntimeError as e:
    print(f"[CRITICAL] Detector failed to initialize: {e}")
    detector = None  # API will respond with 503 if detector is unusable


# ---------------------------------------------------------------------
# 4. API Endpoints
# ---------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable due to a startup failure.",
        )
    return {"status": "ok", "model": "loaded"}


@app.post("/predict_video/", response_class=JSONResponse)
async def predict_video(file: UploadFile = File(...)):
    """
    Accept a video file upload, run deepfake detection, and return results.

    - **file**: video file (mp4, mov, avi, mkv, webm)

    Returns JSON with:
    - `prediction`: "FAKE" or "REAL"
    - `confidence`: float in [0, 1]
    - `message`: human-readable forensic explanation
    - `gradcam_image_url`: URL to the side-by-side Grad-CAM visualization
    """
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable due to a startup failure.",
        )

    # Validate MIME type
    if not (
        file.content_type and file.content_type.startswith("video/")
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Only video files are accepted.",
        )

    # Save uploaded file to a temp path
    temp_filename = f"temp_{uuid.uuid4().hex}.mp4"
    temp_path = os.path.join(os.getcwd(), temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = detector.predict(temp_path)
        return JSONResponse(content=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    except IOError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Video processing error: {e}",
        )

    except Exception as e:
        print(f"[ERROR] Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}",
        )

    finally:
        # Always clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"[INFO] Cleaned up temp file: {temp_path}")


# ---------------------------------------------------------------------
# 5. Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "video_server:app",
        host="127.0.0.1",
        port=8080,
        reload=False,   # Set True only during development
    )