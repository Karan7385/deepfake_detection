import os
import cv2
import uuid
import shutil
import numpy as np
import tensorflow as tf
import uvicorn  # <--- Added this missing import
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
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
    def __init__(self) -> None:
        try:
            # 1. Load the model
            self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print(f"[INFO] TensorFlow Model loaded from {MODEL_PATH}")
            
            # 2. Identify the target layer for Grad-CAM
            self.target_layer_name = self._find_target_layer()
            
            # 3. WARM UP: Forces Keras to initialize the functional graph
            dummy_input = np.zeros((1, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            self.model.predict(dummy_input, verbose=0)
            print("[INFO] Model warmed up and layers initialized.")
            
        except Exception as e:
            print(f"[CRITICAL] Failed to load model: {e}")
            raise e

    def _find_target_layer(self):
        """Finds the last convolutional or TimeDistributed layer for heatmaps."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.TimeDistributed):
                return layer.name
            if 'conv' in layer.name.lower():
                return layer.name
        return self.model.layers[-1].name

    def _preprocess_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate indices to get exactly SEQUENCE_LENGTH frames
        indices = np.linspace(0, max(0, total_frames - 1), SEQUENCE_LENGTH, dtype=int)
        raw_frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
            raw_frames.append(resized)
            frames.append(resized / 255.0)
        
        cap.release()
        return np.expand_dims(np.array(frames), axis=0).astype(np.float32), raw_frames

    def _get_gradcam(self, input_tensor, frame_idx):
        """Generates Grad-CAM heatmap for a specific frame."""
        grad_model = tf.keras.models.Model(
            [self.model.inputs], 
            [self.model.get_layer(self.target_layer_name).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(input_tensor)
            class_channel = preds[:, 0]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))

        frame_features = last_conv_layer_output[0][frame_idx]
        heatmap = frame_features @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()

    def _generate_explanation(self, label, confidence, heatmap_val):
        """Builds a human-readable forensic rationale."""
        base = f"The DeepVerify Video model classified the footage as **{label.upper()}** with confidence **{confidence*100:.2f}%**."
        
        if label == "REAL":
            assessment = "Assessment: **High assurance of authenticity**. Natural patterns and temporal consistency detected."
        else:
            assessment = "Assessment: **Synthetic Manipulation Detected**. Anomalies found in facial structures or temporal flow."

        localization = f"**Grad-CAM focus:** Forensic attention (intensity: {heatmap_val:.2f}) is focused on highlighted anomalies."
        return f"{base}\n\n{assessment}\n\n{localization}"

    def predict(self, video_path: str):
        input_tensor, raw_frames = self._preprocess_video(video_path)
        
        preds = self.model.predict(input_tensor, verbose=0)
        score = float(preds[0][0])
        
        label = "FAKE" if score > 0.5 else "REAL"
        confidence = float(score if label == "FAKE" else (1 - score))

        mid_idx = len(raw_frames) // 2
        heatmap = self._get_gradcam(input_tensor, frame_idx=mid_idx)
        
        # Create Side-by-Side result
        heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(raw_frames[mid_idx], 0.6, heatmap_img, 0.4, 0)

        timestamp = datetime.now().strftime("%H%M%S")
        fname = f"vid_analysis_{timestamp}.jpg"
        save_path = os.path.join(RESULTS_DIR, fname)
        
        combined = np.hstack((raw_frames[mid_idx], overlay))
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        return {
            "prediction": label,
            "confidence": confidence,
            "explanation": self._generate_explanation(label, confidence, float(np.mean(heatmap))),
            "gradcam_image_url": f"http://127.0.0.1:8080/results/{fname}"
        }

# ---------------------------------------------------------------------
# 3. FastAPI Setup
# ---------------------------------------------------------------------

app = FastAPI()
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

detector = VideoDeepfakeDetector()

@app.post("/predict_video/")
async def predict_video(file: UploadFile = File(...)):
    temp_name = f"temp_{uuid.uuid4()}.mp4"
    with open(temp_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        result = detector.predict(temp_name)
        return result
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)