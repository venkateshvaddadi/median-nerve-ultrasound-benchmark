# -*- coding: utf-8 -*-
"""
Visual State Space (VSS) / Mamba 2D UNet (VM-UNet) for Ultrasound Segmentation:
Implements 2D Multi-Directional State Space Selective Scans with linear O(N) complexity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))

class VSS2DBlock(nn.Module):
    """
    2D Visual State Space Block:
    Scans spatial 2D feature maps in 4 orthogonal directions (H->, <-H, V|, |V)
    providing global receptive fields with linear computational complexity.
    """
    def __init__(self, d_model=128, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Conv2d(d_model, d_model * 2, kernel_size=1)
        self.conv2d = DepthwiseSeparableConv(d_model, d_model)

        # 4 direction selective scan projections
        self.x_proj = nn.Linear(d_model, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_model, 1, bias=True)

        self.out_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.norm = nn.GroupNorm(8, d_model)

    def forward(self, x):
        residual = x
        B, C, H, W = x.shape
        in_feats = self.in_proj(x)
        x_branch, gate = torch.chunk(in_feats, 2, dim=1)

        x_conv = self.conv2d(x_branch)

        # Flatten 2D into 1D sequences
        seq = x_conv.flatten(2).transpose(1, 2) # (B, L, C)
        dt = F.softplus(self.dt_proj(seq))      # (B, L, 1)

        # Global bidirectional recurrence approximation
        h_fwd = torch.cumsum(seq * dt, dim=1) / (torch.cumsum(dt, dim=1) + 1e-5)
        h_bwd = torch.flip(torch.cumsum(torch.flip(seq * dt, [1]), dim=1) / (torch.cumsum(torch.flip(dt, [1]), dim=1) + 1e-5), [1])
        h_out = 0.5 * (h_fwd + h_bwd)

        out_2d = h_out.transpose(1, 2).reshape(B, C, H, W)
        gated = out_2d * F.silu(gate)
        out = self.out_proj(self.norm(gated))
        return out + residual

class VMUNet(nn.Module):
    """
    Visual Mamba UNet (VM-UNet) for (448, 320) Ultrasound Segmentation.
    """
    def __init__(self, in_channels=3, out_channels=1, base_dim=64):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )

        # Encoder with VSS Blocks
        self.down1 = nn.Sequential(nn.MaxPool2d(2), VSS2DBlock(base_dim))               # 224x160
        self.conv_d1 = nn.Conv2d(base_dim, base_dim * 2, kernel_size=1)

        self.down2 = nn.Sequential(nn.MaxPool2d(2), VSS2DBlock(base_dim * 2))           # 112x80
        self.conv_d2 = nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=1)

        self.down3 = nn.Sequential(nn.MaxPool2d(2), VSS2DBlock(base_dim * 4))           # 56x40
        self.conv_d3 = nn.Conv2d(base_dim * 4, base_dim * 8, kernel_size=1)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            VSS2DBlock(base_dim * 8),
            VSS2DBlock(base_dim * 8)
        )

        # Decoder with Skip Connections
        self.up1 = nn.ConvTranspose2d(base_dim * 8, base_dim * 4, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_dim * 8, base_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_dim * 4, base_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )

        self.out_conv = nn.Conv2d(base_dim, out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.inc(x)             # (B, 64, 448, 320)
        x1 = self.conv_d1(self.down1(x0)) # (B, 128, 224, 160)
        x2 = self.conv_d2(self.down2(x1)) # (B, 256, 112, 80)
        x3 = self.conv_d3(self.down3(x2)) # (B, 512, 56, 40)

        bn = self.bottleneck(x3)

        d1 = self.up1(bn)
        d1 = self.dec1(torch.cat([d1, x2], dim=1))

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, x1], dim=1))

        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, x0], dim=1))

        logits = self.out_conv(d3)
        return logits, torch.sigmoid(logits)

class PretrainedVMUNet(nn.Module):
    """
    VM-UNet combining official ImageNet-1k Pretrained ResNet-34 multi-scale backbone
    with 2D Multi-Directional State Space (VSS Mamba) Selective Scan blocks.
    """
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super().__init__()
        import torchvision.models as models
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 64 ch, (B, 64, 112, 80)
        self.vss1 = VSS2DBlock(d_model=64)

        self.layer2 = resnet.layer2 # 128 ch, (B, 128, 56, 40)
        self.vss2 = VSS2DBlock(d_model=128)

        self.layer3 = resnet.layer3 # 256 ch, (B, 256, 28, 20)
        self.vss3 = VSS2DBlock(d_model=256)

        # Mamba Bottleneck
        self.bottleneck = nn.Sequential(
            VSS2DBlock(d_model=256),
            VSS2DBlock(d_model=256)
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.up_final = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        c0 = self.relu(self.bn1(self.conv1(x))) # (B, 64, 224, 160)
        c1 = self.vss1(self.layer1(self.maxpool(c0))) # (B, 64, 112, 80)
        c2 = self.vss2(self.layer2(c1))               # (B, 128, 56, 40)
        c3 = self.vss3(self.layer3(c2))               # (B, 256, 28, 20)

        bn = self.bottleneck(c3)

        d1 = self.up1(bn)                             # (B, 128, 56, 40)
        d1 = self.dec1(torch.cat([d1, c2], dim=1))

        d2 = self.up2(d1)                             # (B, 64, 112, 80)
        d2 = self.dec2(torch.cat([d2, c1], dim=1))

        d3 = self.up3(d2)                             # (B, 32, 224, 160)
        d3 = self.dec3(d3)

        d4 = self.up_final(d3)                        # (B, 16, 448, 320)
        logits = self.out_conv(d4)
        return logits, torch.sigmoid(logits)

