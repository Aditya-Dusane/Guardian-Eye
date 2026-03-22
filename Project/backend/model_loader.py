"""
model_loader.py
Singleton loader for the MemAE3D anomaly detection model.
Loads model.pth and train_errors.npy once and exposes them globally.
"""

import os
import sys
import torch
import numpy as np

# Ensure Project/ is on path so we can import model.py
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model import MemAE3D  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model.pth")
ERRORS_PATH = os.path.join(CHECKPOINT_DIR, "train_errors.npy")

# Module-level singletons
_model: MemAE3D | None = None
_threshold: float | None = None
_train_errors: np.ndarray | None = None


def get_model() -> MemAE3D:
    """Return the loaded (cached) MemAE3D model."""
    global _model
    if _model is None:
        _load()
    return _model  # type: ignore[return-value]


def get_threshold() -> float:
    """Return the anomaly threshold."""
    global _threshold
    if _threshold is None:
        _load()
    return _threshold  # type: ignore[return-value]


def get_train_errors() -> np.ndarray:
    """Return the raw training reconstruction errors."""
    global _train_errors
    if _train_errors is None:
        _load()
    return _train_errors  # type: ignore[return-value]


def get_device() -> torch.device:
    return DEVICE


def is_loaded() -> bool:
    return _model is not None


def _load() -> None:
    """Internal: actually load the model and training errors."""
    global _model, _threshold, _train_errors

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"model.pth not found at {MODEL_PATH}. "
            "Please ensure checkpoints/model.pth exists."
        )
    if not os.path.exists(ERRORS_PATH):
        raise FileNotFoundError(
            f"train_errors.npy not found at {ERRORS_PATH}. "
            "Please ensure checkpoints/train_errors.npy exists."
        )

    # Load model
    model = MemAE3D().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    _model = model

    # Load training errors
    _train_errors = np.load(ERRORS_PATH)

    # ------------------- 🔥 IMPROVED THRESHOLD -------------------
    # Use higher percentile + safety margin to reduce false positives
    base_threshold = float(np.percentile(_train_errors, 99.5))
    _threshold = base_threshold * 1.1
    # ------------------------------------------------------------

    # Debug prints (VERY useful)
    print(f"[model_loader] Model loaded on {DEVICE}")
    print(f"[model_loader] Train error stats:")
    print(f"  Min: {np.min(_train_errors):.6f}")
    print(f"  Max: {np.max(_train_errors):.6f}")
    print(f"  Mean: {np.mean(_train_errors):.6f}")
    print(f"[model_loader] Threshold (99.5 pct + margin): {_threshold:.6f}")