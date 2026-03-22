# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryModule(nn.Module):
    def __init__(self, mem_dim=200, fea_dim=256):
        super().__init__()
        self.mem = nn.Parameter(torch.randn(mem_dim, fea_dim))

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        att_weight = F.softmax(torch.matmul(x_flat, self.mem.t()), dim=-1)
        mem_out = torch.matmul(att_weight, self.mem)
        mem_out = mem_out.permute(0, 2, 1).view(B, C, T, H, W)
        return mem_out


class MemAE3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),

            nn.Conv3d(128, 256, 3, padding=1),
            nn.ReLU()
        )

        self.memory = MemoryModule(mem_dim=200, fea_dim=256)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose3d(128, 64, 2, stride=2),
            nn.ReLU(),

            nn.Conv3d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.memory(z)
        out = self.decoder(z)
        return out