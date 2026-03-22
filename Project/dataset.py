# dataset.py

import os
import glob
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class TrainDataset(Dataset):
    def __init__(self, root, clip_len=16):
        self.clip_len = clip_len
        self.samples = []

        self.transform = T.Compose([
            T.Resize((128,128)),
            T.ToTensor()
        ])

        video_files = glob.glob(os.path.join(root, "*.avi"))

        for video_path in video_files:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            for i in range(0, total_frames - clip_len, 5):
                self.samples.append((video_path, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, start_idx = self.samples[idx]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        frames = []

        for _ in range(self.clip_len):
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        clip_tensor = torch.stack(frames, dim=1)  # C,T,H,W
        return clip_tensor