# -*- coding: utf-8 -*-
"""
TransUNet Architecture implementation for 2D Medical Ultrasound Segmentation:
Combines ResNet CNN feature extractor with Vision Transformer (ViT) encoder
and Cascaded Upsampler (CUP) decoder.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class TransUNet(nn.Module):
    """
    TransUNet adapted for (448, 320) ultrasound inputs.
    CNN Stem downsamples by 16x -> (B, 512, 28, 20).
    Transformer processes N = 28 * 20 = 560 tokens with self-attention.
    CUP Decoder restores full (448, 320) with skip connections.
    """
    def __init__(self, in_channels=3, out_channels=1, embed_dim=512, num_layers=6, num_heads=8):
        super().__init__()
        # CNN Encoder / Stem
        self.inc = ConvBlock(in_channels, 64)        # (B, 64, 448, 320)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))   # (B, 128, 224, 160)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 256))  # (B, 256, 112, 80)
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(256, 512))  # (B, 512, 56, 40)
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(512, embed_dim)) # (B, 512, 28, 20)

        # ViT Bottleneck
        self.token_h, self.token_w = 28, 20
        num_patches = self.token_h * self.token_w
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder (Cascaded Upsampler with Skips)
        self.up1 = nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2) # 28x20 -> 56x40
        self.conv_up1 = ConvBlock(256 + 512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # 56x40 -> 112x80
        self.conv_up2 = ConvBlock(128 + 256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 112x80 -> 224x160
        self.conv_up3 = ConvBlock(64 + 128, 64)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)   # 224x160 -> 448x320
        self.conv_up4 = ConvBlock(32 + 64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)         # (B, 64, 448, 320)
        x2 = self.down1(x1)      # (B, 128, 224, 160)
        x3 = self.down2(x2)      # (B, 256, 112, 80)
        x4 = self.down3(x3)      # (B, 512, 56, 40)
        x5 = self.down4(x4)      # (B, 512, 28, 20)

        B, C, H, W = x5.shape
        # Flatten spatial tokens for ViT: (B, C, H, W) -> (B, H*W, C)
        tokens = x5.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed

        for blk in self.transformer_blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        # Reshape tokens back to spatial map
        vit_out = tokens.transpose(1, 2).reshape(B, C, H, W)

        # Decoder with skip connections
        d1 = self.up1(vit_out)
        d1 = self.conv_up1(torch.cat([d1, x4], dim=1))

        d2 = self.up2(d1)
        d2 = self.conv_up2(torch.cat([d2, x3], dim=1))

        d3 = self.up3(d2)
        d3 = self.conv_up3(torch.cat([d3, x2], dim=1))

        d4 = self.up4(d3)
        d4 = self.conv_up4(torch.cat([d4, x1], dim=1))

        logits = self.out_conv(d4)
        return logits, torch.sigmoid(logits)

class PretrainedTransUNet(nn.Module):
    """
    TransUNet with official ImageNet-1k Pretrained ResNet-50 CNN Stem
    and Vision Transformer bottleneck with Cascaded Upsampler (CUP) decoder.
    Parameters: ~25.2 M
    """
    def __init__(self, in_channels=3, out_channels=1, embed_dim=512, num_layers=4, num_heads=8, pretrained=True):
        super().__init__()
        import torchvision.models as models
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1 # 256 ch, (112, 80)
        self.layer2 = resnet.layer2 # 512 ch, (56, 40)
        self.layer3 = resnet.layer3 # 1024 ch, (28, 20)

        # Transformer bottleneck
        self.proj_in = nn.Conv2d(1024, embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, 28 * 20, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Conv2d(embed_dim, 512, kernel_size=1)

        # Cascaded Upsampler (CUP)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.head = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        c0 = self.relu(self.bn1(self.conv1(x))) # (B, 64, 224, 160)
        c1 = self.layer1(self.maxpool(c0))     # (B, 256, 112, 80)
        c2 = self.layer2(c1)                    # (B, 512, 56, 40)
        c3 = self.layer3(c2)                    # (B, 1024, 28, 20)

        feat = self.proj_in(c3)                 # (B, 512, 28, 20)
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        t_feat = tokens.transpose(1, 2).view(B, C, H, W)
        t_feat = self.proj_out(t_feat)          # (B, 512, 28, 20)

        d1 = self.up1(t_feat)                   # (B, 256, 56, 40)
        d1 = self.dec1(torch.cat([d1, c2], dim=1))

        d2 = self.up2(d1)                       # (B, 128, 112, 80)
        d2 = self.dec2(torch.cat([d2, c1], dim=1))

        d3 = self.up3(d2)                       # (B, 64, 224, 160)
        d3 = self.dec3(torch.cat([d3, c0], dim=1))

        d4 = self.up4(d3)                       # (B, 32, 448, 320)
        logits = self.head(d4)
        return logits, torch.sigmoid(logits)

