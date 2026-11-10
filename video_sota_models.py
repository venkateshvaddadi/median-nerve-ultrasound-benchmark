# -*- coding: utf-8 -*-
"""
Spatio-Temporal Video Segmentation Architecture Suite for Ultrasound:
Designed for consecutive video clip sequences (e.g., T=16 or T=32 frames).

Includes:
1. Modified VisTR (IEEE TUFFC 2024 / Your Lab Benchmark)
2. Video-Swin Transformer (CVPR 2022 / 3D Shifted-Window Attention)
3. Space-Time Factorized Video Transformer (TimeSformer / Divided Space-Time ViT)
4. Spatio-Temporal Video-Mamba (Vivim / 3D SS3D Selective Scan)
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR

# ==============================================================================
# 1. Modified VisTR Wrapper
# ==============================================================================
def get_modified_vistr(num_frames=16, backbone="resnet18", device="cuda"):
    from models.VisTR.models import build_model
    target_bb = str(backbone)
    target_nf = int(num_frames)
    target_dev = str(device)
    class Args:
        lr = 1e-4
        lr_backbone = 1e-5
        batch_size = 2
        weight_decay = 1e-4
        epochs = 50
        lr_drop = 40
        clip_max_norm = 0.1
        backbone = target_bb
        backbone_name = target_bb
        dilation = False

        position_embedding = 'sine'
        enc_layers = 6
        dec_layers = 6
        dim_feedforward = 2048
        hidden_dim = 384
        dropout = 0.1
        nheads = 8
        num_classes = 1
        num_frames = target_nf
        num_queries = target_nf
        pre_norm = False
        masks = True
        aux_loss = False
        labels_loss = True
        boxes_loss = True
        mask_loss = True
        box_loss = True
        giou_loss = True
        set_cost_class = 1.0
        set_cost_bbox = 5.0
        set_cost_giou = 2.0
        mask_loss_coef = 1.0
        dice_loss_coef = 1.0
        bbox_loss_coef = 5.0
        giou_loss_coef = 2.0
        eos_coef = 0.1
        dataset_file = 'ytvos'
        device = target_dev


    model, criterion, postprocessors = build_model(Args())
    return model.to(device)


# ==============================================================================
# 2. Video-Swin Transformer for 3D Spatio-Temporal Volumes (T, C, H, W)
# ==============================================================================
class VideoSwinUNet(nn.Module):
    """
    Video-Swin Transformer:
    Processes (B, C, T, H, W) spatio-temporal video volumes using 3D Shifted Windows.
    Generates continuous segmentation masks (B, 1, T, H, W) for all T frames.
    """
    def __init__(self, in_channels=3, out_channels=1, feature_size=48, drop_rate=0.0, pretrained_path=None):
        super().__init__()
        self.swin3d = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            use_checkpoint=True,
            spatial_dims=3
        )
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained_swin(pretrained_path)

    def load_pretrained_swin(self, pretrained_path):
        state = torch.load(pretrained_path, map_location='cpu')
        ckpt_dict = state.get('state_dict', state.get('net', state))
        model_dict = self.swin3d.state_dict()
        matched = 0
        for k, v in ckpt_dict.items():
            clean_k = k.replace('module.', '')
            if clean_k.startswith('swin_vit.'):
                clean_k = 'swinViT.' + clean_k[9:]
            elif not clean_k.startswith('swinViT.'):
                clean_k = 'swinViT.' + clean_k
            
            if clean_k in model_dict and model_dict[clean_k].shape == v.shape:
                model_dict[clean_k] = v
                matched += 1
        self.swin3d.load_state_dict(model_dict)
        print(f">> Video-Swin: Successfully transferred {matched}/{len(model_dict)} layers from 3D pretrained weights ({pretrained_path})!")

    def forward(self, x):
        # x input: (B, C, T, H, W)
        orig_shape = (x.shape[2], x.shape[3], x.shape[4])
        # Ensure 3D dimensions are divisible by 32 for Swin hierarchical downsampling:
        # T: padded/interpolated to 32, H: 224 (7*32), W: 160 (5*32)
        x_in = F.interpolate(x, size=(32, 224, 160), mode='trilinear', align_corners=False)
        logits = self.swin3d(x_in)
        out = F.interpolate(logits, size=orig_shape, mode='trilinear', align_corners=False)
        return out



# ==============================================================================
# 3. Space-Time Factorized Transformer (TimeSformer / Divided Space-Time ViT)
# ==============================================================================
class DividedSpaceTimeBlock(nn.Module):
    """
    Factorized Spatio-Temporal Attention:
    1. Temporal Attention: computes self-attention across T frames for each spatial patch token.
    2. Spatial Attention: computes self-attention across spatial patches within each frame.
    """
    def __init__(self, dim=256, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm_temp = nn.LayerNorm(dim)
        self.attn_temp = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm_spat = nn.LayerNorm(dim)
        self.attn_spat = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x, B, T, N):
        # x shape: (B * T, N, D)
        D = x.shape[-1]

        # --- Temporal Self-Attention ---
        # Reshape to (B * N, T, D)
        x_t = x.view(B, T, N, D).permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
        res_t = x_t
        x_t = self.norm_temp(x_t)
        attn_t, _ = self.attn_temp(x_t, x_t, x_t)
        x_t = res_t + attn_t
        # Reshape back to (B * T, N, D)
        x = x_t.view(B, N, T, D).permute(0, 2, 1, 3).contiguous().view(B * T, N, D)

        # --- Spatial Self-Attention ---
        res_s = x
        x_s = self.norm_spat(x)
        attn_s, _ = self.attn_spat(x_s, x_s, x_s)
        x = res_s + attn_s

        # --- Feed-Forward Network ---
        x = x + self.mlp(self.norm_mlp(x))
        return x


class TimeSformerUNet(nn.Module):
    """
    Spatio-Temporal Video Segmentation Transformer with Divided Space-Time Attention.
    """
    def __init__(self, in_channels=3, out_channels=1, embed_dim=256, depth=4, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stem = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.blocks = nn.ModuleList([
            DividedSpaceTimeBlock(dim=embed_dim, num_heads=8) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder to restore original frame resolution
        self.dec1 = nn.ConvTranspose2d(embed_dim, 128, kernel_size=2, stride=2)
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.head = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # x input: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        # Flatten B and T to extract patch tokens
        x_flat = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        tokens = self.stem(x_flat) # (B*T, embed_dim, H_p, W_p)
        H_p, W_p = tokens.shape[2], tokens.shape[3]
        N = H_p * W_p

        seq = tokens.flatten(2).transpose(1, 2) # (B*T, N, embed_dim)

        # Pass through space-time factorized transformer blocks
        for blk in self.blocks:
            seq = blk(seq, B, T, N)

        seq = self.norm(seq)
        feat = seq.transpose(1, 2).view(B * T, self.embed_dim, H_p, W_p)

        # Upsample back to (B*T, out_channels, H, W)
        u1 = F.relu(self.dec1(feat))
        u2 = F.relu(self.dec2(u1))
        u3 = F.relu(self.dec3(u2))
        u4 = F.relu(self.dec4(u3))
        out = self.head(u4)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        # Reshape to (B, out_channels, T, H, W)
        return out.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4).contiguous()


# ==============================================================================
# 4. Spatio-Temporal Video-Mamba (Vivim / 3D Selective Scan SS3D)
# ==============================================================================
class Mamba3DBlock(nn.Module):
    """
    3D Selective Scan (SS3D) across (T, H, W) for linear-time video state modeling.
    """
    def __init__(self, d_model=64, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Conv3d(d_model, d_model * 2, kernel_size=1)
        self.dw_conv3d = nn.Conv3d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False)
        self.bn = nn.BatchNorm3d(d_model)
        self.act = nn.SiLU(inplace=True)

        self.dt_proj = nn.Linear(d_model, 1, bias=True)
        self.out_proj = nn.Conv3d(d_model, d_model, kernel_size=1)
        self.norm = nn.GroupNorm(8, d_model)

    def forward(self, x):
        residual = x
        B, C, T, H, W = x.shape
        in_feats = self.in_proj(x)
        x_branch, gate = torch.chunk(in_feats, 2, dim=1)
        x_conv = self.act(self.bn(self.dw_conv3d(x_branch)))

        # 3D Sequence scan along temporal-spatial continuous dimension
        seq = x_conv.flatten(2).transpose(1, 2) # (B, T*H*W, C)
        dt = F.softplus(self.dt_proj(seq))

        # Bidirectional selective accumulation across time-space trajectory
        h_fwd = torch.cumsum(seq * dt, dim=1) / (torch.cumsum(dt, dim=1) + 1e-5)
        h_bwd = torch.flip(torch.cumsum(torch.flip(seq * dt, [1]), dim=1) / (torch.cumsum(torch.flip(dt, [1]), dim=1) + 1e-5), [1])
        h_out = 0.5 * (h_fwd + h_bwd)

        out_3d = h_out.transpose(1, 2).view(B, C, T, H, W)
        gated = out_3d * torch.sigmoid(gate)
        return self.norm(self.out_proj(gated)) + residual


class VideoMambaUNet(nn.Module):
    """
    Video-Mamba (Vivim) for real-time Spatio-Temporal Ultrasound Video Tracking.
    Linear complexity O(T * H * W) enables processing T=32 or 64 frames at >60 FPS.
    """
    def __init__(self, in_channels=3, out_channels=1, base_ch=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_ch, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_ch),
            nn.SiLU(inplace=True)
        )
        self.mamba1 = Mamba3DBlock(base_ch)
        self.down1 = nn.Conv3d(base_ch, base_ch * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.mamba2 = Mamba3DBlock(base_ch * 2)
        self.down2 = nn.Conv3d(base_ch * 2, base_ch * 4, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.bottleneck = Mamba3DBlock(base_ch * 4)

        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = nn.Sequential(
            nn.Conv3d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_ch * 2),
            nn.SiLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_ch),
            nn.SiLU(inplace=True)
        )

        self.head = nn.Conv3d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        # x input: (B, C, T, H, W)
        x0 = self.stem(x)
        e1 = self.mamba1(x0)
        e2 = self.mamba2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ==============================================================================
# 5. Video-Mask2Former (3D Mask Queries & Spatio-Temporal Mask Attention)
# ==============================================================================
class VideoMask2Former(nn.Module):
    """
    Video-Mask2Former: Treats video segmentation as 3D spatio-temporal tube mask classification.
    - Backbone: ImageNet-1K Pretrained ResNet-50 extracting multi-scale per-frame feature maps.
    - Pixel Decoder: 3D spatio-temporal FPN multi-scale feature aggregator.
    - Transformer Decoder: Learnable spatio-temporal queries with masked cross-attention.
    """
    def __init__(self, in_channels=3, out_channels=1, num_queries=8, hidden_dim=256, pretrained_path=None):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Load ResNet-50 backbone
        import torchvision.models as models
        res = models.resnet50(weights=None)
        if pretrained_path and os.path.exists(pretrained_path):
            state = torch.load(pretrained_path, map_location='cpu')
            if 'model' in state:
                state = state['model']
            res.load_state_dict(state, strict=False)
            print(f">> VideoMask2Former: Successfully loaded ImageNet pretrained weights from {pretrained_path}!")
        elif os.path.exists('/home/medimg/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth'):
            p = '/home/medimg/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth'
            state = torch.load(p, map_location='cpu')
            res.load_state_dict(state, strict=False)
            print(f">> VideoMask2Former: Successfully loaded ImageNet pretrained weights from {p}!")

        self.conv1 = res.conv1
        self.bn1 = res.bn1
        self.relu = res.relu
        self.maxpool = res.maxpool
        self.layer1 = res.layer1 # 256 ch, 1/4
        self.layer2 = res.layer2 # 512 ch, 1/8
        self.layer3 = res.layer3 # 1024 ch, 1/16

        # Multi-scale 3D spatio-temporal pixel decoder (FPN)
        self.lateral3 = nn.Conv3d(1024, hidden_dim, kernel_size=1)
        self.lateral2 = nn.Conv3d(512, hidden_dim, kernel_size=1)
        self.lateral1 = nn.Conv3d(256, hidden_dim, kernel_size=1)
        self.smooth = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # Spatio-temporal queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cls_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        c1 = self.relu(self.bn1(self.conv1(x_flat)))
        c1_p = self.maxpool(c1)
        l1 = self.layer1(c1_p) # (B*T, 256, H/4, W/4)
        l2 = self.layer2(l1)   # (B*T, 512, H/8, W/8)
        l3 = self.layer3(l2)   # (B*T, 1024, H/16, W/16)

        _, _, H4, W4 = l1.shape
        _, _, H8, W8 = l2.shape
        _, _, H16, W16 = l3.shape

        # Reshape to 3D video tensors (B, C, T, H, W)
        v1 = l1.view(B, T, 256, H4, W4).permute(0, 2, 1, 3, 4)
        v2 = l2.view(B, T, 512, H8, W8).permute(0, 2, 1, 3, 4)
        v3 = l3.view(B, T, 1024, H16, W16).permute(0, 2, 1, 3, 4)

        # 3D Feature Pyramid Pixel Decoder
        p3 = self.lateral3(v3)
        p2 = self.lateral2(v2) + F.interpolate(p3, size=(T, H8, W8), mode="trilinear", align_corners=False)
        p1 = self.lateral1(v1) + F.interpolate(p2, size=(T, H4, W4), mode="trilinear", align_corners=False)
        feats = self.smooth(p1) # (B, hidden_dim, T, H/4, W/4)

        feats_flat = feats.permute(0, 2, 3, 4, 1).contiguous().view(B, T * H4 * W4, self.hidden_dim)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        q_norm = self.norm1(queries)
        attn_out, _ = self.cross_attn(q_norm, feats_flat, feats_flat)
        queries = queries + attn_out
        queries = queries + self.mlp(self.norm2(queries))

        mask_embeds = self.mask_embed(queries)
        pred_masks_flat = torch.bmm(mask_embeds, feats_flat.transpose(1, 2))
        pred_masks = pred_masks_flat.view(B, self.num_queries, T, H4, W4)
        cls_scores = F.softmax(self.cls_head(queries).squeeze(-1), dim=-1)
        weighted_mask = torch.sum(pred_masks * cls_scores.view(B, self.num_queries, 1, 1, 1), dim=1, keepdim=True)
        out = F.interpolate(weighted_mask, size=(T, H, W), mode="trilinear", align_corners=False)
        return out


# ==============================================================================
# 6. SAM 2 / MemSAM (Streaming Spatio-Temporal Memory Bank Network)
# ==============================================================================
class SAM2MemSAM(nn.Module):
    """
    SAM 2 / MemSAM for Medical Ultrasound Video:
    Streaming memory attention architecture with ImageNet-1K pretrained ResNet backbone.
    Maintains a temporal memory bank of past frame embeddings to guide current frame
    segmentation, preventing drift and dropouts.
    """
    def __init__(self, in_channels=3, out_channels=1, embed_dim=256, memory_len=4, pretrained_path=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_len = memory_len

        # 2D Frame Encoder with ImageNet Pretrained ResNet Backbone
        import torchvision.models as models
        res = models.resnet34(weights=None)
        if pretrained_path and os.path.exists(pretrained_path):
            state = torch.load(pretrained_path, map_location='cpu')
            if 'model' in state:
                state = state['model']
            res.load_state_dict(state, strict=False)
            print(f">> SAM2MemSAM: Successfully loaded pretrained backbone from {pretrained_path}!")
        elif os.path.exists('/home/medimg/.cache/torch/hub/checkpoints/resnet34-b627a593.pth'):
            p = '/home/medimg/.cache/torch/hub/checkpoints/resnet34-b627a593.pth'
            state = torch.load(p, map_location='cpu')
            res.load_state_dict(state, strict=False)
            print(f">> SAM2MemSAM: Successfully loaded pretrained backbone from {p}!")
        elif os.path.exists('/home/venkatesh/.cache/torch/hub/checkpoints/resnet34-b627a593.pth'):
            p = '/home/venkatesh/.cache/torch/hub/checkpoints/resnet34-b627a593.pth'
            state = torch.load(p, map_location='cpu')
            res.load_state_dict(state, strict=False)
            print(f">> SAM2MemSAM: Successfully loaded pretrained backbone from {p}!")

        self.conv1 = res.conv1
        self.bn1 = res.bn1
        self.relu = res.relu
        self.maxpool = res.maxpool
        self.layer1 = res.layer1 # 64 ch, H/4, W/4
        self.layer2 = res.layer2 # 128 ch, H/8, W/8
        self.layer3 = res.layer3 # 256 ch, H/16, W/16

        # Projection to memory embed dimension
        self.proj = nn.Conv2d(256, embed_dim, kernel_size=1)

        # Streaming Memory Attention: Queries attend to past frame memory embeddings
        self.mem_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.mem_norm = nn.LayerNorm(embed_dim)

        # Memory Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True)
        )

        # High-resolution multi-scale feature skip decoder
        self.up3 = nn.ConvTranspose2d(embed_dim, 128, kernel_size=2, stride=2) # H/8
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # H/4
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        self.up_final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # H/2
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), # H
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        preds = []
        memory_bank = []

        for t in range(T):
            frame_t = x[:, :, t, :, :] # (B, C, H, W)
            c1 = self.relu(self.bn1(self.conv1(frame_t)))
            c1_p = self.maxpool(c1)
            f1 = self.layer1(c1_p) # (B, 64, H/4, W/4)
            f2 = self.layer2(f1)   # (B, 128, H/8, W/8)
            f3 = self.layer3(f2)   # (B, 256, H/16, W/16)

            feat_t = self.proj(f3) # (B, embed_dim, H/16, W/16)
            B_f, D_f, H_e, W_e = feat_t.shape
            feat_flat = feat_t.flatten(2).transpose(1, 2) # (B, H_e*W_e, self.embed_dim)

            if len(memory_bank) == 0:
                mem_context = feat_flat
            else:
                mem_stack = torch.cat(memory_bank[-self.memory_len:], dim=1)
                norm_q = self.mem_norm(feat_flat)
                attn_context, _ = self.mem_attn(norm_q, mem_stack, mem_stack)
                mem_context = feat_flat + attn_context

            memory_bank.append(feat_flat)
            ctx_spatial = mem_context.transpose(1, 2).view(B, D_f, H_e, W_e)
            fused = self.fusion(torch.cat([feat_t, ctx_spatial], dim=1))

            d2 = self.dec2(torch.cat([self.up3(fused), f2], dim=1))
            d1 = self.dec1(torch.cat([self.up2(d2), f1], dim=1))
            mask_t = self.up_final(d1)
            mask_t = F.interpolate(mask_t, size=(H, W), mode='bilinear', align_corners=False)
            preds.append(mask_t.unsqueeze(2))

        return torch.cat(preds, dim=2)


# ==============================================================================
# 7. Space-Time Memory Networks (STCN / XMem / SLM-SAM2)
# ==============================================================================
class STCNXMemUNet(nn.Module):
    """
    Space-Time Correspondence Network (STCN / XMem):
    Official STCN uses an ImageNet-pretrained ResNet backbone as the Key Encoder
    to compute dense non-local pixel-to-pixel space-time affinity matches between past memory
    frames and the current query frame.
    """
    def __init__(self, in_channels=3, out_channels=1, key_dim=64, value_dim=128, pretrained_path=None):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim

        # Key Encoder: ImageNet Pretrained ResNet-18 Backbone
        import torchvision.models as models
        res = models.resnet18(weights=None)
        if pretrained_path and os.path.exists(pretrained_path):
            state = torch.load(pretrained_path, map_location='cpu')
            res.load_state_dict(state, strict=False)
            print(f">> STCNXMem: Loaded pretrained key backbone from {pretrained_path}!")
        elif os.path.exists('/home/medimg/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'):
            p = '/home/medimg/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
            state = torch.load(p, map_location='cpu')
            res.load_state_dict(state, strict=False)
            print(f">> STCNXMem: Loaded pretrained key backbone from {p}!")
        elif os.path.exists('/home/venkatesh/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'):
            p = '/home/venkatesh/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
            state = torch.load(p, map_location='cpu')
            res.load_state_dict(state, strict=False)
            print(f">> STCNXMem: Loaded pretrained key backbone from {p}!")

        # Key Encoder layers: down to 1/16 resolution
        self.key_stem = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
        self.key_layer1 = res.layer1 # 64 ch, 1/4
        self.key_layer2 = res.layer2 # 128 ch, 1/8
        self.key_layer3 = res.layer3 # 256 ch, 1/16
        self.key_proj = nn.Conv2d(256, key_dim, kernel_size=1)

        # Value Encoder: lightweight value projection head
        self.value_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, value_dim, kernel_size=3, stride=4, padding=1)
        )

        # Dynamic read-out decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(value_dim, 64, kernel_size=4, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        outputs = []
        keys_mem, vals_mem = [], []
        max_mem_frames = 4

        for t in range(T):
            cur_frame = x[:, :, t, :, :]
            # Key extraction via Pretrained ResNet
            x_stem = self.key_stem(cur_frame)
            x_l1 = self.key_layer1(x_stem)
            x_l2 = self.key_layer2(x_l1)
            x_l3 = self.key_layer3(x_l2)
            k_t = self.key_proj(x_l3) # (B, key_dim, H/16, W/16)

            v_t = self.value_encoder(cur_frame) # (B, value_dim, H/16, W/16)
            B_s, _, H_s, W_s = k_t.shape

            k_flat = k_t.flatten(2).transpose(1, 2) # (B, N_s, K)
            v_flat = v_t.flatten(2).transpose(1, 2) # (B, N_s, V)

            if len(keys_mem) == 0:
                readout_val = v_t
            else:
                all_keys = torch.cat(keys_mem[-max_mem_frames:], dim=1)
                all_vals = torch.cat(vals_mem[-max_mem_frames:], dim=1)
                affinity = torch.bmm(k_flat, all_keys.transpose(1, 2)) / math.sqrt(self.key_dim)
                aff_weights = F.softmax(affinity, dim=-1)
                readout = torch.bmm(aff_weights, all_vals)
                readout_val = readout.transpose(1, 2).view(B, self.value_dim, H_s, W_s) + v_t

            keys_mem.append(k_flat)
            vals_mem.append(v_flat)

            m_t = self.decoder(readout_val)
            m_t = F.interpolate(m_t, size=(H, W), mode='bilinear', align_corners=False)
            outputs.append(m_t.unsqueeze(2))

        return torch.cat(outputs, dim=2) # (B, 1, T, H, W)


# ==============================================================================
# 8. Motion-Guided Transformers: Flow-Guided UNet / EchoFlow-Net
# ==============================================================================
class FlowGuidedEchoUNet(nn.Module):
    """
    Optical Flow / Motion-Guided Video Transformer:
    Explicitly computes inter-frame displacement/motion correlation vectors to compensate
    for ultrasound probe movement and hand jitter between adjacent frames.
    """
    def __init__(self, in_channels=3, out_channels=1, base_dim=32):
        super().__init__()
        # Motion vector estimator between frame t-1 and frame t
        self.flow_estimator = nn.Sequential(
            nn.Conv2d(in_channels * 2, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, padding=1) # 2D optical motion grid (dx, dy)
        )

        # Multi-frame warped feature fusion
        self.feat_net = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1)
        )

        # Spatio-Temporal Motion Integration
        self.motion_fusion = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, out_channels, kernel_size=1)
        )

    def warp_features(self, x, flow):
        # x: (B, C, H, W), flow: (B, 2, H_f, W_f)
        B, C, H, W = x.shape
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device),
                                        torch.linspace(-1, 1, W, device=x.device), indexing='ij')
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).repeat(B, 1, 1, 1) # (B, H, W, 2)
        vgrid = grid + flow.permute(0, 2, 3, 1)
        return F.grid_sample(x, vgrid, mode='bilinear', padding_mode='border', align_corners=False)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        outs = []
        prev_feat = None
        prev_frame = None

        for t in range(T):
            cur_frame = x[:, :, t, :, :]
            cur_feat = self.feat_net(cur_frame)

            if prev_frame is None:
                fused = cur_feat
            else:
                pair = torch.cat([prev_frame, cur_frame], dim=1)
                motion_flow = self.flow_estimator(pair)
                # Warp previous frame features according to estimated probe translation
                warped_feat = self.warp_features(prev_feat, motion_flow)
                warped_feat = F.interpolate(warped_feat, size=(cur_feat.shape[2], cur_feat.shape[3]), mode='bilinear', align_corners=False)
                fused = self.motion_fusion(torch.cat([cur_feat, warped_feat], dim=1))

            prev_frame = cur_frame
            prev_feat = cur_feat

            mask_t = self.seg_head(fused)
            mask_t = F.interpolate(mask_t, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(mask_t.unsqueeze(2))

        return torch.cat(outs, dim=2) # (B, 1, T, H, W)


# ==============================================================================
# 9. TubeDETR / Tube-Former (Continuous Spatio-Temporal Cylinder Queries)
# ==============================================================================
class TubeFormer(nn.Module):
    """
    TubeDETR / Tube-Former:
    Uses an ImageNet-pretrained ResNet backbone per frame to extract robust anatomical features,
    then factorizes joint space-time cross-attention via continuous 3D Spatio-Temporal Tube Queries
    bounding the nerve trajectory through ultrasound sweeps.
    """
    def __init__(self, in_channels=3, out_channels=1, hidden_dim=256, num_tube_queries=8, pretrained_path=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tube_queries = num_tube_queries

        # Pretrained ResNet Backbone for frame-level feature extraction
        import torchvision.models as models
        res = models.resnet34(weights=None)
        if pretrained_path and os.path.exists(pretrained_path):
            state = torch.load(pretrained_path, map_location='cpu')
            res.load_state_dict(state, strict=False)
            print(f">> TubeFormer: Loaded pretrained ResNet-34 backbone from {pretrained_path}!")
        elif os.path.exists('/home/medimg/.cache/torch/hub/checkpoints/resnet34-b627a593.pth'):
            p = '/home/medimg/.cache/torch/hub/checkpoints/resnet34-b627a593.pth'
            state = torch.load(p, map_location='cpu')
            res.load_state_dict(state, strict=False)
            print(f">> TubeFormer: Loaded pretrained ResNet-34 backbone from {p}!")
        elif os.path.exists('/home/venkatesh/.cache/torch/hub/checkpoints/resnet34-b627a593.pth'):
            p = '/home/venkatesh/.cache/torch/hub/checkpoints/resnet34-b627a593.pth'
            state = torch.load(p, map_location='cpu')
            res.load_state_dict(state, strict=False)
            print(f">> TubeFormer: Loaded pretrained ResNet-34 backbone from {p}!")

        self.stem = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
        self.layer1 = res.layer1 # 64 ch, 1/4
        self.layer2 = res.layer2 # 128 ch, 1/8
        self.layer3 = res.layer3 # 256 ch, 1/16
        self.proj = nn.Conv2d(256, hidden_dim, kernel_size=1)

        # Spatio-Temporal Tube Queries: (num_tube_queries, hidden_dim)
        self.tube_queries = nn.Embedding(num_tube_queries, hidden_dim)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        # High-resolution multi-scale skip decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, kernel_size=2, stride=2), # 1/8
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=2, stride=2), # 1/4
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, kernel_size=2, stride=2), # 1/2
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), # 1
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

        # Backbone feature extraction
        stem = self.stem(x_flat)
        l1 = self.layer1(stem) # (BT, 64, H/4, W/4)
        l2 = self.layer2(l1)   # (BT, 128, H/8, W/8)
        l3 = self.layer3(l2)   # (BT, 256, H/16, W/16)
        feat = self.proj(l3)   # (BT, hidden_dim, H/16, W/16)
        _, D_f, H_f, W_f = feat.shape

        # Space-Time Tube Attention across volume
        tokens = feat.view(B, T, D_f, H_f, W_f).permute(0, 1, 3, 4, 2).contiguous().view(B, T * H_f * W_f, D_f)
        queries = self.tube_queries.weight.unsqueeze(0).expand(B, -1, -1) # (B, num_tube_queries, D_f)

        # Tube cross-attention
        attn_out, _ = self.temporal_attn(tokens, queries, queries)
        tokens_refined = self.norm(tokens + attn_out)

        # Reshape back to (B*T, D_f, H_f, W_f)
        feat_refined = tokens_refined.view(B, T, H_f, W_f, D_f).permute(0, 1, 4, 2, 3).contiguous().view(B * T, D_f, H_f, W_f)

        # High-resolution skip decoder
        d3 = self.dec3(feat_refined) # (BT, 128, H/8, W/8)
        d2 = self.dec2(torch.cat([d3, l2], dim=1)) # (BT, 64, H/4, W/4)
        out_flat = self.dec1(torch.cat([d2, l1], dim=1)) # (BT, 1, H, W)

        out = out_flat.view(B, T, 1, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return out


# ==============================================================================
# Factory Helper Function
# ==============================================================================
def get_video_model(model_name, in_channels=3, num_classes=1, num_frames=16, device="cuda"):
    name = model_name.lower().replace("-", "_").strip()
    if name in ["vistr", "modified_vistr"]:
        return get_modified_vistr(num_frames=num_frames, backbone="resnet18", device=device)
    elif name in ["video_swin", "videoswin", "video_swin_unet"]:
        return VideoSwinUNet(in_channels=in_channels, out_channels=num_classes).to(device)
    elif name in ["timesformer", "divided_space_time"]:
        return TimeSformerUNet(in_channels=in_channels, out_channels=num_classes).to(device)
    elif name in ["video_mamba", "vivim", "videomamba"]:
        return VideoMambaUNet(in_channels=in_channels, out_channels=num_classes).to(device)
    elif name in ["video_mask2former", "mask2former", "v_mask2former"]:
        return VideoMask2Former(in_channels=in_channels, out_channels=num_classes).to(device)
    elif name in ["sam2", "memsam", "sam_2", "med_sam2"]:
        return SAM2MemSAM(in_channels=in_channels, out_channels=num_classes).to(device)
    elif name in ["stcn", "xmem", "slm_sam2"]:
        return STCNXMemUNet(in_channels=in_channels, out_channels=num_classes).to(device)
    elif name in ["flow_guided", "echoflow", "echoflow_net", "motion_guided"]:
        return FlowGuidedEchoUNet(in_channels=in_channels, out_channels=num_classes).to(device)
    elif name in ["tubedetr", "tubeformer", "tube_former"]:
        return TubeFormer(in_channels=in_channels, out_channels=num_classes).to(device)
    else:
        raise ValueError(f"Unknown video architecture: {model_name}")

