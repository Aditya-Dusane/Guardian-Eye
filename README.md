# Guardian Eye — 3D Autoencoder Anomaly Detection

<div align="center">

**Memory-augmented 3D autoencoder for video anomaly detection, trained on the ShanghaiTech Campus dataset.**

</div>

---

## Overview

Guardian Eye is a deep learning system that detects anomalous events in surveillance video. It uses a **Memory-Augmented 3D Autoencoder (MemAE-3D)** that learns to reconstruct only normal video clips. Anomalies produce high reconstruction error and are flagged above a learned threshold.

**Key features:**
- 3D convolutional encoder–decoder with memory module
- Sliding-window clip inference (16 frames per clip)
- Pixel-level and frame-level anomaly localisation
- FastAPI backend with REST endpoints
- Modern dark-theme web frontend (no build step required)

---

## Folder Structure

```
major project/
├── Project/
│   ├── checkpoints/
│   │   ├── model.pth           # Trained model weights
│   │   └── train_errors.npy    # Training reconstruction errors
│   ├── dataset/
│   │   ├── training/videos/    # .avi training clips
│   │   └── testing/
│   │       ├── frames/         # Test video frames (per-video folders)
│   │       ├── test_frame_mask/
│   │       └── test_pixel_mask/
│   ├── outputs/                # Annotated inference output videos
│   ├── backend/
│   │   ├── app.py              # FastAPI server
│   │   ├── model_loader.py     # Singleton model loader
│   │   └── requirements.txt    # Backend dependencies
│   ├── frontend/
│   │   ├── index.html          # Main UI
│   │   ├── style.css           # Dark glassmorphism styles
│   │   └── app.js              # API integration logic
│   ├── model.py                # MemAE3D model definition
│   ├── dataset.py              # Training dataset loader
│   ├── train.py                # Training script
│   ├── inference.py            # Batch inference script (ShanghaiTech)
│   └── check_gpu.py            # GPU/CUDA diagnostic
├── .gitignore
└── README.md
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended (CPU inference is supported)
- `git`, `pip`

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install backend dependencies

```bash
cd "Project/backend"
pip install -r requirements.txt
```

> **Note:** PyTorch GPU builds require separate installation. Visit [pytorch.org](https://pytorch.org/get-started/locally/) and install the CUDA version matching your system before running the above command.

### 4. Verify GPU (optional)

```bash
cd Project
python check_gpu.py
```

---

## Running the Backend

From the **`Project/`** directory:

```bash
cd Project
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

The server starts at **http://localhost:8000**

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness probe |
| `GET`  | `/model-status` | Model info, device, threshold |
| `POST` | `/predict` | Upload video → get anomaly prediction |

**Quick test after starting the server:**

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-status
```

---

## Running the Frontend

No build step required. Open the file directly in a browser:

```
Project/frontend/index.html
```

Or serve it with any static server (to avoid CORS on file://) for a better experience:

```bash
# Using Python's built-in server
cd "Project/frontend"
python -m http.server 3000
# Then open http://localhost:3000
```

The frontend connects to the backend at `http://localhost:8000` by default.

---

## Running Batch Inference (ShanghaiTech)

Requires the ShanghaiTech dataset extracted into `Project/dataset/`:

```bash
cd Project
python inference.py
```

Outputs annotated MP4 videos to `Project/outputs/` and prints frame-level and pixel-level AUC scores.

---

## Training from Scratch

```bash
cd Project
python train.py
```

This reads `.avi` clips from `Project/dataset/training/videos/` and saves:
- `checkpoints/model.pth`
- `checkpoints/train_errors.npy`

---

## Model Architecture

```
Input clip  [B, 3, 16, 128, 128]
    │
    ▼
Encoder (Conv3D × 3 + MaxPool3D × 2)  →  [B, 256, 4, 32, 32]
    │
    ▼
Memory Module (attention over 200 prototype vectors)
    │
    ▼
Decoder (ConvTranspose3D × 2 + Conv3D)  →  [B, 3, 16, 128, 128]
    │
    ▼
MSE Reconstruction Error  →  Anomaly Score
```

**Threshold:** 98th percentile of training reconstruction errors.

---

## GitHub

### First push

```bash
cd "C:\Users\adity\OneDrive\Desktop\major project"
git init          # already done
git add .
git commit -m "Initial commit: MemAE-3D anomaly detection with backend & frontend"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin master
```

---

## Acknowledgements

- **Dataset:** [ShanghaiTech Campus Dataset](https://svip-lab.github.io/dataset/campus_dataset.html)
- **Model inspiration:** Gong et al., "Memorizing Normality to Detect Anomaly" (ICCV 2019)
