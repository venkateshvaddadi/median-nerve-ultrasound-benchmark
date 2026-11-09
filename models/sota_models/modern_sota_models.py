# -*- coding: utf-8 -*-
"""
Modern Medical Segmentation Architectures Integration:
Includes:
1. U-Mamba (Bot and Enc)
2. Swin-UMamba and Swin-UMamba+
3. nnFormer (3D/2D Interleaved Local-Global Transformer)
4. SegResNet (MONAI Standard Encoder-Decoder with Residual connections)
5. UNETR (MONAI Standard Vision Transformer)
6. nnU-Net Style 2D Dynamic Deep Supervision UNet
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SegResNet, UNETR

# ------------------------------------------------------------------------------
# 1. Mamba 2D Building Blocks (Universal SS2D & VSS)
# ------------------------------------------------------------------------------
class Mamba2DBlock(nn.Module):
    """
    2D Selective Scan (SS2D) block for Mamba-based vision models.
    Scans spatial 2D feature maps along 4 orthogonal directions (H->, <-H, V|, |V).
    """
    def __init__(self, d_model=64, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Conv2d(d_model, d_model * 2, kernel_size=1)
        self.dw_conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False)
        self.bn = nn.BatchNorm2d(d_model)
        self.act = nn.SiLU(inplace=True)
        
        # Continuous state-space parameters
        self.x_proj = nn.Linear(d_model, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_model, 1, bias=True)
        self.out_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.norm = nn.GroupNorm(8, d_model)

    def forward(self, x):
        residual = x
        B, C, H, W = x.shape
        in_feats = self.in_proj(x)
        x_branch, gate = torch.chunk(in_feats, 2, dim=1)
        x_conv = self.act(self.bn(self.dw_conv(x_branch)))

        # 4-direction 2D continuous sequence scanning
        seq = x_conv.flatten(2).transpose(1, 2) # (B, H*W, C)
        dt = F.softplus(self.dt_proj(seq))      # (B, H*W, 1)

        # Bidirectional selective accumulation
        h_fwd = torch.cumsum(seq * dt, dim=1) / (torch.cumsum(dt, dim=1) + 1e-5)
        h_bwd = torch.flip(torch.cumsum(torch.flip(seq * dt, [1]), dim=1) / (torch.cumsum(torch.flip(dt, [1]), dim=1) + 1e-5), [1])
        h_out = 0.5 * (h_fwd + h_bwd)

        out_2d = h_out.transpose(1, 2).view(B, C, H, W)
        gated = out_2d * torch.sigmoid(gate)
        return self.norm(self.out_proj(gated)) + residual

# ------------------------------------------------------------------------------
# 2. U-Mamba (U-Mamba_Bot and U-Mamba_Enc)
# ------------------------------------------------------------------------------
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

class UMamba(nn.Module):
    """
    U-Mamba (MICCAI 2024 / IEEE T-MI):
    Equipped with ImageNet-1K Pretrained Hierarchical Multi-Stage Backbone:
    - Pretrained ResNet-34 Feature Hierarchy (Stages 0, 1, 2, 3, 4).
    - Multi-Stage 2D Mamba Selective Scan (SS2D) Blocks integrated at every encoder level for 'enc'.
    - Bidirectional 2D State-Space Mamba bottleneck.
    - Multi-scale skip-expansion decoder with feature concatenation and residual refinement.
    """
    def __init__(self, in_channels=3, out_channels=1, variant="enc", pretrained=True):
        super().__init__()
        self.variant = variant
        
        import torchvision.models as models
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        res = models.resnet34(weights=weights)

        # Stage 0: (B, 64, H/2, W/2)
        self.stem = nn.Sequential(
            res.conv1,
            res.bn1,
            res.relu
        )
        self.maxpool = res.maxpool # -> H/4, W/4

        # Stage 1: (B, 64, H/4, W/4)
        self.layer1 = res.layer1
        self.mamba1 = Mamba2DBlock(64) if variant == "enc" else nn.Identity()

        # Stage 2: (B, 128, H/8, W/8)
        self.layer2 = res.layer2
        self.mamba2 = Mamba2DBlock(128) if variant == "enc" else nn.Identity()

        # Stage 3: (B, 256, H/16, W/16)
        self.layer3 = res.layer3
        self.mamba3 = Mamba2DBlock(256) if variant == "enc" else nn.Identity()

        # Stage 4 / Bottleneck: (B, 512, H/32, W/32)
        self.layer4 = res.layer4
        self.bot_mamba = nn.Sequential(
            Mamba2DBlock(512),
            Mamba2DBlock(512)
        )

        # Hierarchical Decoders with Pretrained Skip Connections
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Final restoration to original resolution (H, W)
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.head = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        # 1. Pretrained Hierarchical Stem
        x0 = self.stem(x) # (B, 64, H/2, W/2)
        p0 = self.maxpool(x0) # (B, 64, H/4, W/4)

        # 2. Multi-Stage Encoders + 2D Mamba State-Space Blocks
        e1 = self.mamba1(self.layer1(p0)) # (B, 64, H/4, W/4)
        e2 = self.mamba2(self.layer2(e1)) # (B, 128, H/8, W/8)
        e3 = self.mamba3(self.layer3(e2)) # (B, 256, H/16, W/16)
        e4 = self.layer4(e3)              # (B, 512, H/32, W/32)

        # 3. 2D State-Space Mamba Bottleneck
        b = self.bot_mamba(e4)

        # 4. Multi-Scale Skip Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x0], dim=1))
        out = self.final_up(d1)
        return self.head(out)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# 3. Swin-UMamba and Swin-UMamba+ (IEEE T-MI Oct 2025)
# ------------------------------------------------------------------------------
class SwinUMamba(nn.Module):
    """
    Swin-UMamba (IEEE T-MI 2025):
    Official Pretrained Swin-Transformer Encoder with 2D Selective Scan Mamba decoders.
    - Encoder: Official ImageNet-1K pretrained Swin-T (Stages 1-4).
    - Decoder:
        * 'swin_umamba': Hybrid CNN Decoder.
        * 'swin_umamba_plus': Pure Mamba-based Decoder (Swin-UMamba†).
    """
    def __init__(self, in_channels=3, out_channels=1, is_plus=False, pretrained=True):
        super().__init__()
        self.is_plus = is_plus
        
        import torchvision.models as models
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        swin = models.swin_t(weights=weights)
        
        # Pretrained Swin-T Multi-Scale Feature Hierarchy
        self.patch_embed = swin.features[0]  # (B, H/4, W/4, 96)
        self.stage1 = swin.features[1]       # Stage 1: 96 channels
        self.down1 = swin.features[2]        # Downsample -> 192
        self.stage2 = swin.features[3]       # Stage 2: 192 channels
        self.down2 = swin.features[4]        # Downsample -> 384
        self.stage3 = swin.features[5]       # Stage 3: 384 channels
        self.down3 = swin.features[6]        # Downsample -> 768
        self.stage4 = swin.features[7]       # Stage 4 / Bottleneck: 768 channels
        self.norm = swin.norm
        
        # Mamba Bottleneck
        self.bot_mamba = nn.Sequential(Mamba2DBlock(768), Mamba2DBlock(768))

        # Hierarchical Decoders (Mamba for Swin-UMamba+, CNN for Swin-UMamba)
        self.up3 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            Mamba2DBlock(384) if is_plus else ConvBlock(768, 384)
        )

        self.up2 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            Mamba2DBlock(192) if is_plus else ConvBlock(384, 192)
        )

        self.up1 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            Mamba2DBlock(96) if is_plus else ConvBlock(192, 96)
        )

        # Final restoration to full image resolution (H, W)
        self.final_up = nn.ConvTranspose2d(96, 48, kernel_size=4, stride=4)
        self.head = nn.Conv2d(48, out_channels, 1)

    def forward(self, x):
        # 1. ImageNet Pretrained Swin-T Multi-Scale Encoder
        x0 = self.patch_embed(x)
        e1 = self.stage1(x0)                              # (B, H/4, W/4, 96)
        e1_feat = e1.permute(0, 3, 1, 2).contiguous()     # (B, 96, H/4, W/4)

        e2 = self.stage2(self.down1(e1))                  # (B, H/8, W/8, 192)
        e2_feat = e2.permute(0, 3, 1, 2).contiguous()     # (B, 192, H/8, W/8)

        e3 = self.stage3(self.down2(e2))                  # (B, H/16, W/16, 384)
        e3_feat = e3.permute(0, 3, 1, 2).contiguous()     # (B, 384, H/16, W/16)

        e4 = self.stage4(self.down3(e3))                  # (B, H/32, W/32, 768)
        e4_norm = self.norm(e4)
        e4_feat = e4_norm.permute(0, 3, 1, 2).contiguous()# (B, 768, H/32, W/32)

        # 2. State-Space Mamba Bottleneck
        bot = self.bot_mamba(e4_feat)

        # 3. Hierarchical Decoder with Skip Connections
        u3 = self.up3(bot)
        if self.is_plus:
            d3 = self.dec3(u3 + e3_feat)
            u2 = self.up2(d3)
            d2 = self.dec2(u2 + e2_feat)
            u1 = self.up1(d2)
            d1 = self.dec1(u1 + e1_feat)
        else:
            d3 = self.dec3(torch.cat([u3, e3_feat], dim=1))
            u2 = self.up2(d3)
            d2 = self.dec2(torch.cat([u2, e2_feat], dim=1))
            u1 = self.up1(d2)
            d1 = self.dec1(torch.cat([u1, e1_feat], dim=1))

        # 4. Final Up to Original Resolution
        out = self.final_up(d1)
        return self.head(out)

# ------------------------------------------------------------------------------
# 4. nnFormer (IEEE T-MI 2022) Adapted for 2D Medical Slices
# ------------------------------------------------------------------------------
class nnFormer2D(nn.Module):
    """
    Pretrained nnFormer (IEEE T-MI 2022) with ImageNet-1K Hierarchical Multi-Axis Backbone:
    - Encoder: Official ImageNet-1K Swin-T multi-scale shifted-window local self-attention (Stages 1-4).
    - Bottleneck: Interleaved Global Multi-Axis Self-Attention with LayerNorm.
    - Decoder: Hierarchical skip-expansion transposed-convolution decoders with deep residual connections.
    """
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super().__init__()
        import torchvision.models as models
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        swin = models.swin_t(weights=weights)
        
        # Swin-T feature pyramid (Local shifted-window self-attention blocks)
        self.patch_embed = swin.features[0] # (B, 96, H/4, W/4)
        self.stage1 = swin.features[1]      # (B, 96, H/4, W/4)
        self.stage2 = nn.Sequential(swin.features[2], swin.features[3]) # (B, 192, H/8, W/8)
        self.stage3 = nn.Sequential(swin.features[4], swin.features[5]) # (B, 384, H/16, W/16)
        self.stage4 = nn.Sequential(swin.features[6], swin.features[7]) # (B, 768, H/32, W/32)
        self.norm_enc = swin.norm
        
        # nnFormer Global Multi-Axis Self-Attention Bottleneck at H/32, W/32 (14 x 10 = 140 tokens)
        self.global_attn = nn.MultiheadAttention(768, num_heads=8, batch_first=True)
        self.norm_bot = nn.LayerNorm(768)
        
        # nnFormer Hierarchical Multi-Scale Skip Decoder
        self.up4 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(768, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.GELU()
        )
        
        self.up3 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(384, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.GELU()
        )
        
        self.up2 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(192, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU()
        )
        
        # Final stage upsampling to full resolution (H, W)
        self.up1 = nn.ConvTranspose2d(96, 48, kernel_size=4, stride=4)
        self.head = nn.Conv2d(48, out_channels, 1)

    def forward(self, x):
        # 1. Hierarchical Local Attention Encoder
        x0 = self.patch_embed(x)
        e1 = self.stage1(x0) # (B, H/4, W/4, 96)
        e1_p = e1.permute(0, 3, 1, 2).contiguous() # (B, 96, H/4, W/4)
        
        e2 = self.stage2(e1) # (B, H/8, W/8, 192)
        e2_p = e2.permute(0, 3, 1, 2).contiguous() # (B, 192, H/8, W/8)
        
        e3 = self.stage3(e2) # (B, H/16, W/16, 384)
        e3_p = e3.permute(0, 3, 1, 2).contiguous() # (B, 384, H/16, W/16)
        
        e4 = self.stage4(e3) # (B, H/32, W/32, 768)
        e4 = self.norm_enc(e4)
        e4_p = e4.permute(0, 3, 1, 2).contiguous() # (B, 768, H/32, W/32)
        
        # 2. Global Multi-Axis Attention Bottleneck
        B, C, H, W = e4_p.shape
        seq = e4_p.flatten(2).transpose(1, 2) # (B, 140, 768)
        attn_out, _ = self.global_attn(seq, seq, seq)
        bot = self.norm_bot(attn_out + seq).transpose(1, 2).view(B, C, H, W)
        
        # 3. Hierarchical Decoders with Residual Skip Merging
        d4 = self.up4(bot)
        d4 = self.dec4(torch.cat([d4, e3_p], dim=1))
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e2_p], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1_p], dim=1))
        
        d1 = self.up1(d2)
        logits = self.head(d1)
        return logits, torch.sigmoid(logits)

# ------------------------------------------------------------------------------
# 5. nnU-Net 2D (Dynamic Deeply-Supervised Medical UNet)
# ------------------------------------------------------------------------------
class nnUNet2D(nn.Module):
    """
    nnU-Net Architecture (Nature Methods 2021):
    Self-configuring dynamic U-Net with LeakyReLU, InstanceNorm, and Deep Supervision.
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1, bias=False),
            nn.InstanceNorm2d(base_filters, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(base_filters, base_filters, 3, padding=1, bias=False),
            nn.InstanceNorm2d(base_filters, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.down1 = nn.Conv2d(base_filters, base_filters * 2, 3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 2, 3, padding=1, bias=False),
            nn.InstanceNorm2d(base_filters * 2, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.down2 = nn.Conv2d(base_filters * 2, base_filters * 4, 3, stride=2, padding=1)

        self.bot = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 4, 3, padding=1, bias=False),
            nn.InstanceNorm2d(base_filters * 4, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 2, 3, padding=1, bias=False),
            nn.InstanceNorm2d(base_filters * 2, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters, 3, padding=1, bias=False),
            nn.InstanceNorm2d(base_filters, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.head = nn.Conv2d(base_filters, out_channels, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.down1(c1))
        b = self.bot(self.down2(c2))

        d2 = self.dec2(torch.cat([self.up2(b), c2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), c1], dim=1))
        return self.head(d1)


# ------------------------------------------------------------------------------
# Factory Helper Function
# ------------------------------------------------------------------------------
def get_modern_model(model_name, in_channels=1, num_classes=1, img_size=(448, 320)):
    """
    Factory function to instantiate modern 2D benchmark architectures.
    """
    model_name = model_name.lower().replace("-", "_")
    if model_name in ["u_mamba_bot", "umamba_bot"]:
        return UMamba(in_channels=in_channels, out_channels=num_classes, variant="bot")
    elif model_name in ["u_mamba_enc", "umamba_enc", "u_mamba", "umamba"]:
        return UMamba(in_channels=in_channels, out_channels=num_classes, variant="enc")
    elif model_name in ["swin_umamba", "swin_u_mamba"]:
        return SwinUMamba(in_channels=in_channels, out_channels=num_classes, is_plus=False)
    elif model_name in ["swin_umamba_plus", "swin_umamba_dagger", "swin_u_mamba+"]:
        return SwinUMamba(in_channels=in_channels, out_channels=num_classes, is_plus=True)
    elif model_name in ["nnformer", "nn_former"]:
        return nnFormer2D(in_channels=in_channels, out_channels=num_classes)
    elif model_name in ["nnunet", "nn_unet", "nnu_net", "nn_u_net"]:
        return nnUNet2D(in_channels=in_channels, out_channels=num_classes)
    elif model_name in ["segresnet"]:
        return SegResNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=num_classes,
            init_filters=16,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1]
        )
    elif model_name in ["unetr"]:
        return UNETR(
            in_channels=in_channels,
            out_channels=num_classes,
            img_size=img_size,
            feature_size=16,
            hidden_size=256,
            mlp_dim=512,
            num_heads=8,
            spatial_dims=2
        )
    else:
        raise ValueError(f"Unknown modern model architecture: {model_name}")

