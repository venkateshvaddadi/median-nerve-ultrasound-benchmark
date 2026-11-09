# -*- coding: utf-8 -*-
"""
Med-SegDiff Conditional Denoising Diffusion Model for Medical Ultrasound Segmentation:
Implements continuous noise schedule, sinusoidal timestep embeddings,
and conditional ultrasound feature modulation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlockWithTime(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.act(self.bn1(self.conv1(x)))
        t_emb = self.time_mlp(t)[:, :, None, None]
        h = h + t_emb
        h = self.act(self.bn2(self.conv2(h)))
        return h + self.shortcut(x)

class MedSegDiff(nn.Module):
    """
    Conditional Diffusion Model for Ultrasound Segmentation.
    Input:
        x_t: Noisy mask at step t (B, 1, 448, 320)
        img: Raw ultrasound conditioning image (B, 3, 448, 320)
        t: Diffusion timestep integer [0, T-1]
    Output:
        predicted noise epsilon_theta (B, 1, 448, 320)
    """
    def __init__(self, in_channels=4, out_channels=1, time_dim=128, base_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Concatenates noisy mask (1ch) + ultrasound image (3ch) = 4 channels
        self.inc = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        self.down1 = ResidualBlockWithTime(base_dim, base_dim * 2, time_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = ResidualBlockWithTime(base_dim * 2, base_dim * 4, time_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = ResidualBlockWithTime(base_dim * 4, base_dim * 8, time_dim)
        self.pool3 = nn.MaxPool2d(2)

        self.mid = ResidualBlockWithTime(base_dim * 8, base_dim * 8, time_dim)

        self.up1 = nn.ConvTranspose2d(base_dim * 8, base_dim * 4, 2, stride=2)
        # u1 (base_dim * 4) + d3 (base_dim * 8) = base_dim * 12 (768 channels)
        self.dec1 = ResidualBlockWithTime(base_dim * 12, base_dim * 4, time_dim)

        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 2, stride=2)
        # u2 (base_dim * 2) + d2 (base_dim * 4) = base_dim * 6 (384 channels)
        self.dec2 = ResidualBlockWithTime(base_dim * 6, base_dim * 2, time_dim)

        self.up3 = nn.ConvTranspose2d(base_dim * 2, base_dim, 2, stride=2)
        # u3 (base_dim) + d1 (base_dim * 2) = base_dim * 3 (192 channels)
        self.dec3 = ResidualBlockWithTime(base_dim * 3, base_dim, time_dim)

        self.out_conv = nn.Conv2d(base_dim, out_channels, 1)

    def forward(self, x_t, img, t):
        t_emb = self.time_mlp(t)
        x = torch.cat([x_t, img], dim=1)

        x0 = self.inc(x)
        d1 = self.down1(x0, t_emb)
        d2 = self.down2(self.pool1(d1), t_emb)
        d3 = self.down3(self.pool2(d2), t_emb)

        mid = self.mid(self.pool3(d3), t_emb)

        u1 = self.up1(mid)
        u1 = self.dec1(torch.cat([u1, d3], dim=1), t_emb)

        u2 = self.up2(u1)
        u2 = self.dec2(torch.cat([u2, d2], dim=1), t_emb)

        u3 = self.up3(u2)
        u3 = self.dec3(torch.cat([u3, d1], dim=1), t_emb)

        eps_pred = self.out_conv(u3)
        return eps_pred
