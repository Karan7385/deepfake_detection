"""
Deepfake Detection API (FastAPI + PyTorch)
------------------------------------------
This service loads a ResNet50-based deepfake detector, performs predictions
on uploaded image paths, generates Grad-CAM heatmaps for interpretability,
and serves results via a FastAPI backend.
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from datetime import datetime
from typing import Dict, Any, Optional
from torchvision import transforms, models
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------
# 1. Configuration & Constants
# ---------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
MODEL_PATH = "best_resnet50.pth"
CLASS_NAMES = ["fake", "real"]
TARGET_LAYER = "layer4"               # ResNet target layer for Grad-CAM
ALPHA = 0.5                           # Heatmap blending factor
CONFIDENCE_THRESHOLD = 0.85           # Threshold for "high confidence" explanations

RESULTS_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# 2. DeepfakeDetector Class
# ---------------------------------------------------------------------

class DeepfakeDetector:
    """
    Encapsulates the ResNet model, preprocessing pipeline, and Grad-CAM logic
    for deepfake detection and explanation.
    """

    def __init__(self) -> None:
        """Initialize the model, device, and input transforms."""
        self.device = DEVICE
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.model = self._load_model()

    # ------------------------- Model Loading -------------------------

    def _load_model(self) -> nn.Module:
        """
        Load a ResNet50 model with custom classification head.
        Falls back to random initialization if weights are missing.
        """
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, NUM_CLASSES)
        )

        try:
            model.load_state_dict(torch.load(MODEL_PATH,
                                             map_location=self.device))
            print(f"[INFO] Loaded model weights from {MODEL_PATH}")
        except FileNotFoundError:
            print(f"[WARNING] No weights found at {MODEL_PATH}. Using random init.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        model.to(self.device).eval()
        return model

    # ----------------------- Utility Functions -----------------------

    @staticmethod
    def _denormalize(img_tensor: torch.Tensor) -> np.ndarray:
        """
        Convert normalized tensor back to numpy image for visualization.
        """
        img = img_tensor.cpu().numpy().transpose((1, 2, 0))
        img = img * np.array([0.229, 0.224, 0.225]) + \
              np.array([0.485, 0.456, 0.406])
        return np.clip(img, 0, 1)

    # ------------------------- Grad-CAM Logic ------------------------

    def _generate_gradcam(self,
                          input_tensor: torch.Tensor,
                          target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the predicted (or target) class.
        """
        gradients, activations = [], []

        def backward_hook(_, __, grad_out):  # Gradient hook
            gradients.append(grad_out[0].cpu().data.numpy())

        def forward_hook(_, __, out):       # Activation hook
            activations.append(out.cpu().data.numpy())

        # Attach hooks
        target_module = dict(self.model.named_modules()).get(TARGET_LAYER)
        if target_module is None:
            raise ValueError(f"Layer '{TARGET_LAYER}' not found in model.")

        handles = [
            target_module.register_forward_hook(forward_hook),
            target_module.register_backward_hook(backward_hook),
        ]

        try:
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            self.model.zero_grad()
            loss = output[0, target_class]
            loss.backward()

            # Process collected activations/gradients
            grads = gradients[0][0]
            acts = activations[0][0]
            weights = np.mean(grads, axis=(1, 2))

            cam = np.zeros(acts.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * acts[i]

            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (224, 224))
            cam = (cam - np.min(cam)) / (np.max(cam) + 1e-8)
            return cam

        finally:
            for h in handles:
                h.remove()

    def _overlay_gradcam(self,
                         img_tensor: torch.Tensor,
                         cam: np.ndarray,
                         alpha: float = ALPHA) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.
        """
        img = self._denormalize(img_tensor)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        return (1 - alpha) * img + alpha * heatmap

    def _save_gradcam_image(self,
                            overlay: np.ndarray,
                            orig_img: torch.Tensor,
                            pred_class: str,
                            confidence: float) -> str:
        """
        Save side-by-side original and Grad-CAM image; return file URL.
        """
        overlay_resized = cv2.resize(overlay, (512, 512))
        orig_resized = cv2.resize(self._denormalize(orig_img), (512, 512))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{pred_class}_{int(confidence*100)}_{timestamp}.jpg"
        save_path = os.path.join(RESULTS_DIR, fname)

        combined = np.hstack((orig_resized, overlay_resized))
        combined = (combined * 255).astype(np.uint8)

        success = cv2.imwrite(save_path,
                              cv2.cvtColor(combined, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise IOError(f"Failed to save image at {save_path}")

        return f"http://127.0.0.1:8000/results/{fname}"

    # ------------------------ Explanation Logic -----------------------

    def _generate_explanation(self,
                              pred_class: str,
                              confidence: float,
                              cam: np.ndarray) -> str:
        """
        Create a human-readable explanation including confidence,
        forensic assessment, and Grad-CAM focus region.
        """
        base = f"The DeepVerify model classified the asset as **{pred_class.upper()}** with confidence **{confidence*100:.2f}%**."

        if pred_class == "real":
            if confidence >= CONFIDENCE_THRESHOLD:
                assessment = "Assessment: **High assurance of authenticity**. Strong evidence of natural patterns and no significant anomalies."
            else:
                assessment = "Assessment: **Probable authenticity (caution)**. Minor artifacts suggest lower media quality, but overall structure is authentic."
        else:  # fake
            if confidence >= CONFIDENCE_THRESHOLD:
                assessment = "Assessment: **Clear synthetic detection**. Multiple anomalies strongly indicate manipulation or synthesis."
            else:
                assessment = "Assessment: **Subtle artifacts detected (inconclusive)**. Possible refined deepfake or degraded genuine media."

        localization = (f"**Grad-CAM focus:** Model attention concentrated on high-heatmap regions "
                        f"(avg focus strength: {np.mean(cam):.2f}), representing decisive features.")

        return f"{base}\n\n{assessment}\n\n{localization}"

    # ------------------------- Prediction API -------------------------

    def predict(self, file_path: str) -> Dict[str, Any]:
        """
        Full prediction pipeline:
        1. Load image
        2. Run inference
        3. Generate Grad-CAM heatmap
        4. Save visualization
        5. Build explanation and probability report
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            img = Image.open(file_path).convert("RGB")
        except Exception:
            raise IOError(f"Cannot open or process file: {file_path}")

        # Prepare input
        input_tensor = self.transform(img).unsqueeze(0)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor.to(self.device))
            probs = torch.softmax(outputs, dim=1)
            pred_idx = outputs.argmax(1).item()
            confidence = float(probs[0, pred_idx].item())
            pred_class = CLASS_NAMES[pred_idx]

        # Grad-CAM & Visualization
        cam = self._generate_gradcam(input_tensor, target_class=pred_idx)
        overlay = self._overlay_gradcam(input_tensor[0], cam)
        gradcam_url = self._save_gradcam_image(overlay,
                                               input_tensor[0],
                                               pred_class,
                                               confidence)

        # Explanation
        explanation = self._generate_explanation(pred_class, confidence, cam)

        # Class probabilities
        class_probs = {
            "fake": float(probs[0, CLASS_NAMES.index("fake")].item()),
            "real": float(probs[0, CLASS_NAMES.index("real")].item()),
        }

        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "explanation": explanation,
            "gradcam_image_url": gradcam_url,
            "class_probabilities": class_probs,
        }


# ---------------------------------------------------------------------
# 3. FastAPI Application Setup
# ---------------------------------------------------------------------

app = FastAPI(
    title="Deepfake Detector API",
    description="Detects deepfake artifacts in images using ResNet50 + Grad-CAM.",
    version="1.0.1"
)

# Load detector globally at startup
try:
    detector = DeepfakeDetector()
except RuntimeError as e:
    print(f"[CRITICAL] Model failed to load: {e}")
    detector = None  # Allow API to respond with 503 if unusable

# Serve results directory as static
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


# ---------------------------------------------------------------------
# 4. API Endpoint
# ---------------------------------------------------------------------

@app.post("/predict_image", response_class=JSONResponse)
async def predict_path(data: Dict[str, str]):
    """
    Predict deepfake authenticity from an image file path.
    Request body:
        { "file_path": "<path/to/image.jpg>" }
    """
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model unavailable due to startup failure."
        )

    file_path = data.get("file_path")
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'file_path' in request body."
        )

    try:
        return detector.predict(file_path)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except IOError as e:
        raise HTTPException(status_code=422,
                            detail=f"Image processing error: {e}")

    except Exception as e:
        print(f"[ERROR] Unexpected: {e}")
        raise HTTPException(status_code=500,
                            detail=f"Internal error: {e}")