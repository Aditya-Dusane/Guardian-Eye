"""
app.py — FastAPI backend for the 3D Autoencoder Anomaly Detection system.

Endpoints:
  GET  /health          — Liveness check
  GET  /model-status    — Model loading state + threshold info
  POST /predict         — Upload a video file, get anomaly prediction
"""

import io
import os
import sys
import tempfile
import uuid

import cv2  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
import torchvision.transforms as T  # type: ignore
from fastapi import FastAPI, File, HTTPException, UploadFile  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from PIL import Image  # type: ignore

# ---------------------------------------------------------------------------
# Ensure paths are set up before importing model_loader
# ---------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)

# Add both dirs so imports work regardless of invocation style
for _p in [PROJECT_DIR, BACKEND_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from . import model_loader  # type: ignore
except ImportError:
    import model_loader  # type: ignore

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="3D Autoencoder Anomaly Detection API",
    description="Video anomaly detection using a memory-augmented 3D autoencoder trained on ShanghaiTech.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLIP_LEN = 16
IMG_SIZE = 128
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])


# ---------------------------------------------------------------------------
# Startup: load model once
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    try:
        model_loader.get_model()
        print(f"[startup] Model ready. Threshold = {model_loader.get_threshold():.6f}")
    except Exception as e:
        print(f"[startup] WARNING: Could not load model: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe — returns OK."""
    return {"status": "ok", "message": "Anomaly detection API is running."}


@app.get("/model-status", tags=["System"])
def model_status():
    """Returns whether the model is loaded and the anomaly threshold."""
    loaded = model_loader.is_loaded()
    if not loaded:
        try:
            model_loader.get_model()
            loaded = True
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={"loaded": False, "error": str(e)},
            )

    errors = model_loader.get_train_errors()
    threshold = model_loader.get_threshold()
    device = str(model_loader.get_device())

    return {
        "loaded": True,
        "device": device,
        "threshold": round(threshold, 6),
        "train_error_count": int(len(errors)),
        "train_error_mean": round(float(np.mean(errors)), 6),  # type: ignore
        "train_error_std": round(float(np.std(errors)), 6),  # type: ignore
    }


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Upload a video file (mp4/avi/mov/mkv). Returns:
      - anomaly_score  : mean reconstruction error across clips
      - threshold      : the trained threshold
      - is_anomaly     : True if score > threshold
      - confidence     : 0–100% how far score is from threshold
      - frame_count    : total frames processed
      - clip_count     : number of 16-frame clips analysed
      - label          : "Anomaly" or "Normal"
    """
    # Validate file type
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Load model (already cached after startup)
    try:
        mdl = model_loader.get_model()
        threshold = model_loader.get_threshold()
        device = model_loader.get_device()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not ready: {e}")

    # Write uploaded file to a temp location
    video_bytes = await file.read()
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, f"upload_{uuid.uuid4().hex}{ext}")
    with open(tmp_path, "wb") as f_out:
        f_out.write(video_bytes)

    try:
        # Extract frames
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        cap.release()

        if len(frames) < CLIP_LEN:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Video too short: only {len(frames)} frames found. "
                    f"Need at least {CLIP_LEN} frames."
                ),
            )

        # Run inference over sliding window clips
        clip_scores = []
        num_clips = len(frames) - CLIP_LEN

        mdl.eval()
        with torch.no_grad():
            for i in range(num_clips):
                clip: list[Image.Image] = list(frames[i : i + CLIP_LEN])  # type: ignore
                clip_tensor = (
                    torch.stack([transform(fr) for fr in clip], dim=1)
                    .unsqueeze(0)
                    .to(device)
                )
                recon = mdl(clip_tensor)
                error = float(((recon - clip_tensor) ** 2).mean().item())
                clip_scores.append(error)

        scores_arr = np.array(clip_scores)
        anomaly_score = float(scores_arr.mean())
        max_score = float(scores_arr.max())
        is_anomaly = anomaly_score > threshold

        # Confidence: how far score is relative to threshold (clamped 0–100)
        if threshold > 0:
            confidence = min(100.0, round(float(abs(anomaly_score - threshold) / threshold * 100), 1))  # type: ignore
        else:
            confidence = 0.0

        return {
            "label": "Anomaly" if is_anomaly else "Normal",
            "is_anomaly": is_anomaly,
            "anomaly_score": round(float(anomaly_score), 6),  # type: ignore
            "max_clip_score": round(float(max_score), 6),  # type: ignore
            "threshold": round(threshold, 6),
            "confidence": confidence,
            "frame_count": len(frames),
            "clip_count": num_clips,
            "filename": file.filename,
        }

    finally:
        # Cleanup temp file
        try:
            os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn  # type: ignore
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
