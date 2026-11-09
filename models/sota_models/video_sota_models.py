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
    Learns continuous 3D mask queries that cross-attend to multi-scale spatio-temporal features.
    """
    def __init__(self, in_channels=3, out_channels=1, num_queries=8, hidden_dim=128):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Multi-scale 3D spatio-temporal encoder backbone
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True)
        )
        self.down1 = nn.Conv3d(32, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.SiLU(inplace=True)
        )
        self.down2 = nn.Conv3d(64, hidden_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # 3D Pixel Decoder: project features to hidden_dim
        self.pixel_decoder = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.SiLU(inplace=True)
        )

        # Learnable 3D spatio-temporal queries (num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Cross-attention: queries attend to spatio-temporal pixel decoder features
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Query mask projector and binary classification score head
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cls_head = nn.Linear(hidden_dim, 1)

        # Final upsampling head back to original (T, H, W)
        self.up_final = nn.Sequential(
            nn.ConvTranspose3d(1, 1, kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            nn.Conv3d(1, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.pixel_decoder(self.down2(e2)) # (B, hidden_dim, T, H/4, W/4)

        B_sz, C_hid, T_sz, H_sub, W_sub = e3.shape
        # Flatten spatio-temporal features: (B, T * H_sub * W_sub, hidden_dim)
        feats_flat = e3.permute(0, 2, 3, 4, 1).contiguous().view(B_sz, T_sz * H_sub * W_sub, C_hid)

        # Expand queries: (B, num_queries, hidden_dim)
        queries = self.query_embed.weight.unsqueeze(0).expand(B_sz, -1, -1)

        # Masked Spatio-Temporal Cross Attention
        q_norm = self.norm1(queries)
        attn_out, _ = self.cross_attn(q_norm, feats_flat, feats_flat)
        queries = queries + attn_out
        queries = queries + self.mlp(self.norm2(queries))

        # Compute query mask embeddings: (B, num_queries, hidden_dim)
        mask_embeddings = self.mask_embed(queries)
        # Dot product with spatio-temporal feature map gives 3D query masks: (B, num_queries, T * H_sub * W_sub)
        pred_masks_flat = torch.bmm(mask_embeddings, feats_flat.transpose(1, 2))
        pred_masks = pred_masks_flat.view(B_sz, self.num_queries, T_sz, H_sub, W_sub)

        # Attention weights over queries via cls_head
        cls_scores = F.softmax(self.cls_head(queries).squeeze(-1), dim=-1) # (B, num_queries)
        weighted_mask = torch.sum(pred_masks * cls_scores.view(B_sz, self.num_queries, 1, 1, 1), dim=1, keepdim=True) # (B, 1, T, H_sub, W_sub)

        # Restore original (T, H, W) resolution
        out = F.interpolate(weighted_mask, size=(T, H, W), mode='trilinear', align_corners=False)
        return out


# ==============================================================================
# 6. SAM 2 / MemSAM (Streaming Spatio-Temporal Memory Bank Network)
# ==============================================================================
class SAM2MemSAM(nn.Module):
    """
    SAM 2 / MemSAM for Medical Ultrasound Video:
    Streaming memory attention architecture. Maintains a temporal memory bank of past
    frame embeddings to guide current frame segmentation, preventing drift and dropouts.
    """
    def __init__(self, in_channels=3, out_channels=1, embed_dim=128, memory_len=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_len = memory_len

        # 2D Frame Encoder (Hierarchical lightweight ConvNet / ViT style)
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True)
        )

        # Memory Attention: Current frame queries attend to past memory bank key/values
        self.mem_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.mem_norm = nn.LayerNorm(embed_dim)

        # Memory Fusion & Mask Decoder
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True)
        )

        self.mask_decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        preds = []
        memory_bank = [] # list of past frame feature tensors

        for t in range(T):
            frame_t = x[:, :, t, :, :] # (B, C, H, W)
            feat_t = self.frame_encoder(frame_t) # (B, D, H_e, W_e)
            B_f, D_f, H_e, W_e = feat_t.shape
            feat_flat = feat_t.flatten(2).transpose(1, 2) # (B, H_e*W_e, D)

            if len(memory_bank) == 0:
                # First frame initialization: Self-memory condition
                mem_context = feat_flat
            else:
                # Memory Bank condition: concatenate past stored frames
                mem_stack = torch.cat(memory_bank[-self.memory_len:], dim=1) # (B, k * tokens, D)
                norm_q = self.mem_norm(feat_flat)
                attn_context, _ = self.mem_attn(norm_q, mem_stack, mem_stack)
                mem_context = feat_flat + attn_context

            # Update memory bank with current processed features
            memory_bank.append(feat_flat)

            # Fuse current frame spatial features with streaming memory context
            ctx_spatial = mem_context.transpose(1, 2).view(B, D_f, H_e, W_e)
            fused = self.fusion(torch.cat([feat_t, ctx_spatial], dim=1))
            mask_t = self.mask_decoder(fused) # (B, 1, H, W)
            mask_t = F.interpolate(mask_t, size=(H, W), mode='bilinear', align_corners=False)
            preds.append(mask_t.unsqueeze(2)) # (B, 1, 1, H, W)

        return torch.cat(preds, dim=2) # (B, 1, T, H, W)


# ==============================================================================
# 7. Space-Time Memory Networks (STCN / XMem / SLM-SAM2)
# ==============================================================================
class STCNXMemUNet(nn.Module):
    """
    Space-Time Correspondence Network (STCN / XMem):
    Computes dense non-local pixel-to-pixel space-time affinity matches between past memory
    frames and the current query frame to propagate median nerve boundaries seamlessly.
    """
    def __init__(self, in_channels=3, out_channels=1, key_dim=64, value_dim=128):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim

        # Key and Value encoders for space-time memory matching (1/16 resolution for memory efficiency)
        self.key_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, key_dim, kernel_size=3, stride=4, padding=1)
        )
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
            k_t = self.key_encoder(cur_frame) # (B, K, H/16, W/16)
            v_t = self.value_encoder(cur_frame) # (B, V, H/16, W/16)
            B_s, _, H_s, W_s = k_t.shape

            k_flat = k_t.flatten(2).transpose(1, 2) # (B, N_s, K)
            v_flat = v_t.flatten(2).transpose(1, 2) # (B, N_s, V)

            if len(keys_mem) == 0:
                readout_val = v_t
            else:
                # Compute Space-Time Memory affinity across recent memory frames
                all_keys = torch.cat(keys_mem[-max_mem_frames:], dim=1)
                all_vals = torch.cat(vals_mem[-max_mem_frames:], dim=1)
                affinity = torch.bmm(k_flat, all_keys.transpose(1, 2)) / math.sqrt(self.key_dim) # (B, N_s, M_total)
                aff_weights = F.softmax(affinity, dim=-1)
                # Weighted readout from memory values: (B, N_s, V)
                readout = torch.bmm(aff_weights, all_vals)
                readout_val = readout.transpose(1, 2).view(B, self.value_dim, H_s, W_s) + v_t

            # Store in short-term/working memory
            keys_mem.append(k_flat)
            vals_mem.append(v_flat)

            # Decode to mask
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
    Predicts continuous 3D cylindrical tubes across video frames.
    Factorizes joint space-time cross-attention into temporal tube queries that bound the nerve path.
    """
    def __init__(self, in_channels=3, out_channels=1, hidden_dim=128, num_tube_queries=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tube_queries = num_tube_queries

        # 3D Feature Extractor
        self.backbone3d = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, hidden_dim, kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU()
        )

        # Spatio-Temporal Tube Queries: (num_tube_queries, hidden_dim)
        self.tube_queries = nn.Embedding(num_tube_queries, hidden_dim)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        # 3D Tube Head to project back into volumetric segmentation mask
        self.tube_head = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        feat3d = self.backbone3d(x) # (B, hidden_dim, T, H/4, W/4)
        B_f, D_f, T_f, H_f, W_f = feat3d.shape

        # Flatten spatio-temporal tokens: (B, T_f * H_f * W_f, D_f)
        tokens = feat3d.permute(0, 2, 3, 4, 1).contiguous().view(B_f, T_f * H_f * W_f, D_f)
        queries = self.tube_queries.weight.unsqueeze(0).expand(B_f, -1, -1) # (B, num_tube_queries, D_f)

        # Tube cross-attention
        attn_out, _ = self.temporal_attn(tokens, queries, queries)
        tokens_refined = self.norm(tokens + attn_out)

        # Reshape back to 3D volume and decode
        vol_refined = tokens_refined.view(B_f, T_f, H_f, W_f, D_f).permute(0, 4, 1, 2, 3).contiguous()
        out = self.tube_head(vol_refined)
        out = F.interpolate(out, size=(T, H, W), mode='trilinear', align_corners=False)
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

