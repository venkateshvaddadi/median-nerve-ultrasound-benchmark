# -*- coding: utf-8 -*-
"""
Swin-UNETR and Swin-UNet 2D Model Architectures for Ultrasound Segmentation
Integrates Hierarchical Swin Transformer with shifted windows and multi-scale feature aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR

class SwinUNETR2D(nn.Module):
    """
    Swin-UNETR adapted for 2D Median Nerve Ultrasound (448, 320).
    Uses hierarchical Swin Transformer encoder blocks with shifted windows,
    followed by multi-scale residual skip decoders.
    Parameters: ~6.30 M
    """
    def __init__(self, in_channels=3, out_channels=1, feature_size=24, drop_rate=0.0):
        super().__init__()
        self.swin_unetr = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            use_checkpoint=False,
            spatial_dims=2
        )

    def forward(self, x):
        logits = self.swin_unetr(x)
        return logits, torch.sigmoid(logits)

class SwinUNet2D(nn.Module):
    """
    Pure Swin-UNet (Swin Transformer Encoder + Swin Transformer Decoder).
    Leverages SwinUNETR with higher capacity feature scaling.
    Parameters: ~14.1 M
    """
    def __init__(self, in_channels=3, out_channels=1, feature_size=36, drop_rate=0.1):
        super().__init__()
        self.swin_unet = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            use_checkpoint=False,
            spatial_dims=2
        )

    def forward(self, x):
        logits = self.swin_unet(x)
        return logits, torch.sigmoid(logits)

class PretrainedSwinUNet(nn.Module):
    """
    Swin-UNet with official ImageNet-1k Pretrained Swin-Transformer (swin_t) backbone.
    Hierarchical shifted-window self-attention features are extracted at 4 scales
    and decoded via multi-scale skip residual upsampling.
    Parameters: ~32.6 M
    """
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super().__init__()
        import torchvision.models as models
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        swin = models.swin_t(weights=weights)
        self.patch_embed = swin.features[0] # (B, 96, H/4, W/4)
        self.stage1 = swin.features[1]      # (B, 96, H/4, W/4)
        self.stage2 = nn.Sequential(swin.features[2], swin.features[3]) # (B, 192, H/8, W/8)
        self.stage3 = nn.Sequential(swin.features[4], swin.features[5]) # (B, 384, H/16, W/16)
        self.stage4 = nn.Sequential(swin.features[6], swin.features[7]) # (B, 768, H/32, W/32)
        self.norm = swin.norm

        self.up4 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(768, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(384, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(192, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(96, 48, kernel_size=4, stride=4)
        self.head = nn.Conv2d(48, out_channels, 1)

    def forward(self, x):
        x0 = self.patch_embed(x)
        e1 = self.stage1(x0)
        e1_p = e1.permute(0, 3, 1, 2).contiguous()

        e2 = self.stage2(e1)
        e2_p = e2.permute(0, 3, 1, 2).contiguous()

        e3 = self.stage3(e2)
        e3_p = e3.permute(0, 3, 1, 2).contiguous()

        e4 = self.stage4(e3)
        e4 = self.norm(e4)
        e4_p = e4.permute(0, 3, 1, 2).contiguous()

        d4 = self.up4(e4_p)
        d4 = self.dec4(torch.cat([d4, e3_p], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e2_p], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1_p], dim=1))

        d1 = self.up1(d2)
        logits = self.head(d1)
        return logits, torch.sigmoid(logits)

