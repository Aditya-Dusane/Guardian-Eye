# train.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import TrainDataset
from model import MemAE3D
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_LEN = 16
BATCH_SIZE = 3
EPOCHS = 10

dataset = TrainDataset("dataset/training/videos", clip_len=CLIP_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = MemAE3D().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

scaler = torch.amp.GradScaler("cuda")

os.makedirs("checkpoints", exist_ok=True)

train_errors = []

print("Starting Training...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for clips in tqdm(loader):
        clips = clips.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            recon = model(clips)
            loss = criterion(recon, clips)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        train_errors.append(loss.item())

    print(f"Epoch {epoch+1} Loss: {epoch_loss/len(loader):.6f}")

torch.save(model.state_dict(), "checkpoints/model.pth")
np.save("checkpoints/train_errors.npy", np.array(train_errors))

print("Training Complete.")