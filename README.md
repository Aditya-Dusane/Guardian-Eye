# 👁️ Guardian Eye — 3D Autoencoder Anomaly Detection

<div align="center">

**AI-powered video anomaly detection system using a Memory-Augmented 3D Autoencoder (MemAE-3D)**

</div>

---

## 🚀 Overview

Guardian Eye is a deep learning system that detects anomalous events in surveillance videos. It uses a **Memory-Augmented 3D Autoencoder (MemAE-3D)** trained on normal video patterns. Any deviation results in high reconstruction error and is flagged as an anomaly.

---

## ✨ Key Features

* 🎥 Video anomaly detection (clip-based)
* 🧠 3D CNN Autoencoder with memory module
* ⚡ FastAPI backend for inference
* 🌐 React (Vite) frontend for UI
* 📊 Hybrid scoring (mean + max error)
* 🎯 Clip voting for stable predictions
* 🖼️ Pixel-level anomaly localisation

---

## 📁 Project Structure

```bash
major project/
├── Project/
│   ├── backend/              # FastAPI backend
│   ├── frontend/             # React frontend (Vite)
│   ├── checkpoints/          # Model weights (NOT included in repo)
│   ├── dataset/              # Dataset (ignored)
│   ├── outputs/              # Output videos (ignored)
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── inference.py
│   ├── check_gpu.py
│   └── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup

### 1. Clone Repository

```bash
git clone https://github.com/Aditya-Dusane/Guardian-Eye.git
cd Guardian-Eye/Project
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ For GPU support, install PyTorch separately from: https://pytorch.org

---

## 🖥️ Running the Backend

```bash
cd Project
uvicorn backend.app:app --reload
```

👉 Backend runs at:

```
http://127.0.0.1:8000
```

👉 API Docs:

```
http://127.0.0.1:8000/docs
```

---

## 🌐 Running the Frontend

Open a new terminal:

```bash
cd Project/frontend
npm install
npm run dev
```

👉 Frontend runs at:

```
http://localhost:5173/
```

---

## 🔗 API Usage

### Endpoint:

```
POST /predict
```

### Input:

* Upload a video (`.mp4`, `.avi`, `.mov`, `.mkv`)

### Output:

```json
{
  "label": "Anomaly / Normal",
  "anomaly_score": 0.0021,
  "threshold": 0.0020,
  "confidence": 72.5,
  "frame_count": 300,
  "clip_count": 280
}
```

---

## 🧠 How It Works

1. Video is split into clips of 16 frames
2. Model reconstructs each clip
3. Reconstruction error is computed
4. Hybrid scoring (mean + max) is applied
5. Clip voting determines final video label

---

## ⚠️ Important Notes

* Model weights are NOT included due to size limits
* Dataset (ShanghaiTech / UCSD) is not included

👉 Required files:

```
Project/checkpoints/model.pth
Project/checkpoints/train_errors.npy
```

---

## 📊 Current Limitations

* Model sensitivity depends on dataset distribution
* Threshold requires tuning
* Performance varies across unseen environments

---

## 🔮 Future Improvements

* Dynamic thresholding
* Real-time video streaming
* Improved generalization
* Enhanced frontend UI

---

## 👨‍💻 Author

**Aditya Dusane**
AI & Data Science Engineer

---

## 🙏 Acknowledgements

* ShanghaiTech Campus Dataset
* UCSD Dataset
* PyTorch & FastAPI communities

---

## 📌 Status

🚧 Work in Progress — actively improving model accuracy
