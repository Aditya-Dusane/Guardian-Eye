# inference.py

import os
import sys
import glob
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from model import MemAE3D  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_LEN = 16
IMG_SIZE = 128
FPS = 25

_CKPT_DIR     = os.path.join(_HERE, "checkpoints")
_MODEL_PATH   = os.path.join(_CKPT_DIR, "model.pth")
_ERRORS_PATH  = os.path.join(_CKPT_DIR, "train_errors.npy")
_OUTPUTS_DIR  = os.path.join(_HERE, "outputs")
_DATASET_DIR  = os.path.join(_HERE, "dataset")

model = MemAE3D().to(DEVICE)
model.load_state_dict(torch.load(_MODEL_PATH, map_location=DEVICE))
model.eval()

train_errors = np.load(_ERRORS_PATH)
threshold = np.percentile(train_errors, 98)  # untouched

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

os.makedirs(_OUTPUTS_DIR, exist_ok=True)

all_frame_scores = []
all_frame_gt = []
all_pixel_scores = []
all_pixel_gt = []

video_folders = sorted(glob.glob(os.path.join(_DATASET_DIR, "testing", "frames", "*")))

for video_folder in video_folders:

    video_name = os.path.basename(video_folder)
    print("\nProcessing:", video_name)

    frame_paths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
    frames = [Image.open(p).convert("RGB") for p in frame_paths]

    frame_gt = np.load(os.path.join(_DATASET_DIR, "testing", "test_frame_mask", f"{video_name}.npy"))
    pixel_gt = np.load(os.path.join(_DATASET_DIR, "testing", "test_pixel_mask", f"{video_name}.npy"))

    if pixel_gt.shape[0] != len(frames):
        pixel_gt = np.transpose(pixel_gt, (2,0,1))

    gt_has_anomaly = np.sum(frame_gt) > 0
    print("Ground Truth Anomaly Present:", gt_has_anomaly)

    frame_scores = []
    pixel_scores_video = []
    pixel_gt_video = []
    output_frames = []

    predicted_anomaly = False
    anomaly_count = 0   # ✅ ADDED

    for i in range(len(frames) - CLIP_LEN):

        clip = frames[i:i+CLIP_LEN]
        clip_tensor = torch.stack(
            [transform(f) for f in clip], dim=1
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            recon = model(clip_tensor)
            diff = (recon - clip_tensor) ** 2

            # ✅ HYBRID SCORING
            mean_score = diff.mean().item()
            max_score = diff.max().item()
            frame_error = 0.6 * max_score + 0.4 * mean_score

        frame_scores.append(frame_error)

        pixel_map = diff.mean(dim=1)[0, -1].cpu().numpy()

        gt_mask = pixel_gt[i].astype(np.uint8)
        gt_mask_resized = cv2.resize(
            gt_mask,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        pixel_scores_video.append(pixel_map.flatten())
        pixel_gt_video.append(gt_mask_resized.flatten())

        frame_img = cv2.cvtColor(
            np.array(clip[-1].resize((IMG_SIZE, IMG_SIZE))),
            cv2.COLOR_RGB2BGR
        )

        # ✅ IMPROVED DECISION (frame-level)
        threshold = 0.00205
        margin = 0.0001

        if frame_error > (threshold + margin):
            anomaly_count += 1   # ✅ CHANGED

        # Pixel localization (UNCHANGED)
        if frame_error > threshold:

            mean = pixel_map.mean()
            std = pixel_map.std()
            adaptive_thresh = mean + 2 * std

            mask = (pixel_map > adaptive_thresh).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for c in contours:
                if cv2.contourArea(c) > 40:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        label_text = f"GT: {'Anomaly' if gt_has_anomaly else 'Normal'} | Pred: {'Anomaly' if predicted_anomaly else 'Normal'}"
        cv2.putText(
            frame_img,
            label_text,
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        output_frames.append(frame_img)

    # ✅ FINAL VIDEO DECISION (CLIP VOTING)
    total_frames = len(frame_scores)

    if anomaly_count > 0.1 * total_frames:
        predicted_anomaly = True
    else:
        predicted_anomaly = False

    # Save video
    out_path = os.path.join(_OUTPUTS_DIR, f"{video_name}_annotated.mp4")
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        FPS,
        (IMG_SIZE, IMG_SIZE)
    )

    for f in output_frames:
        writer.write(f)
    writer.release()

    # ✅ SMOOTHING ADDED
    frame_scores = np.array(frame_scores)
    frame_scores = np.convolve(frame_scores, np.ones(5)/5, mode='same')

    frame_scores = (frame_scores - frame_scores.min()) / (
        frame_scores.max() - frame_scores.min() + 1e-8
    )

    auc_frame = roc_auc_score(frame_gt[:len(frame_scores)], frame_scores)
    print("Frame AUC:", auc_frame)
    print("Model Predicted Anomaly:", predicted_anomaly)

    all_frame_scores.extend(frame_scores)
    all_frame_gt.extend(frame_gt[:len(frame_scores)])

    pixel_scores_video = np.concatenate(pixel_scores_video)
    pixel_gt_video = np.concatenate(pixel_gt_video)

    auc_pixel = roc_auc_score(pixel_gt_video, pixel_scores_video)
    print("Pixel AUC:", auc_pixel)

    all_pixel_scores.extend(pixel_scores_video)
    all_pixel_gt.extend(pixel_gt_video)

print("\nOVERALL FRAME AUC:", roc_auc_score(all_frame_gt, all_frame_scores))
print("OVERALL PIXEL AUC:", roc_auc_score(all_pixel_gt, all_pixel_scores))