#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Spatio-Temporal Video Segmentation Benchmark Runner
Supports:
    1. 'vistr': Modified VisTR (IEEE TUFFC 2024 / DETR Spatio-Temporal Transformer)
    2. 'video_mamba': Spatio-Temporal Video-Mamba (SS3D Selective Scan)
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sota_models.video_dataset import VideoAsterDataset
from sota_models.video_sota_models import (
    VideoMambaUNet, get_modified_vistr, VideoSwinUNet, TimeSformerUNet,
    VideoMask2Former, SAM2MemSAM, STCNXMemUNet, FlowGuidedEchoUNet, TubeFormer, OriginalTubeDETR, get_video_model
)
from models.VisTR.util.misc import NestedTensor

CALIB_CSA_MM2 = 0.004328254

def compute_detailed_metrics(pred_bin, target_bin):
    TP = np.count_nonzero((pred_bin == 1) & (target_bin == 1))
    FP = np.count_nonzero((pred_bin == 1) & (target_bin == 0))
    FN = np.count_nonzero((pred_bin == 0) & (target_bin == 1))
    TN = np.count_nonzero((pred_bin == 0) & (target_bin == 0))

    prec = TP / max(1, (TP + FP))
    rec = TP / max(1, (TP + FN))
    iou = TP / max(1, (TP + FP + FN))
    dice = (2.0 * TP) / max(1, (2 * TP + FP + FN))

    csa_pred = float(np.sum(pred_bin)) * CALIB_CSA_MM2
    csa_gt = float(np.sum(target_bin)) * CALIB_CSA_MM2
    csa_err = abs(csa_pred - csa_gt)

    return {
        "dice": dice,
        "iou": iou,
        "precision": prec,
        "recall": rec,
        "csa_pred": csa_pred,
        "csa_gt": csa_gt,
        "csa_err": csa_err
    }

def dice_loss_fn(pred, target, eps=1e-4):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1.0 - (2.0 * intersection + eps) / (union + eps)

def main():
    parser = argparse.ArgumentParser(description="Train Video Segmentation Models")
    parser.add_argument("--model", type=str, required=True, choices=[
        "vistr", "vistr_resnet50", "vistr_resnet101", "video_mamba", "video_swin",
        "timesformer", "video_mask2former", "tubeformer", "tubedetr", "original_tubedetr", "sam2", "stcn", "flow_guided"
    ])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"VIDEO BENCHMARK RUNNER: {args.model.upper()}")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size} | Frames: {args.num_frames} | LR: {args.lr}")
    print("=" * 80)

    # Locate dataset
    candidate_roots = [
        "../data_making/aster_updated_data_nov_09_2022_with_flip/",
        "/media/medimg/Data/venkatesh_data/Aster/data_making/aster_updated_data_nov_09_2022_with_flip/",
        "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/Aster_hosptial_median_nerve/CTS/data_making/aster_updated_data_nov_09_2022_with_flip/"
    ]
    data_root = None
    for cand in candidate_roots:
        if os.path.exists(cand):
            data_root = cand
            break
    if not data_root:
        raise FileNotFoundError(f"Could not locate aster data root in: {candidate_roots}")

    patients_csv = os.path.join(data_root, "csv_files/full_data_csv/patients_list_1_99.csv")
    train_csv = os.path.join(data_root, "csv_files/full_data_csv/data_with_v_h_1_79_train.csv")
    val_csv = os.path.join(data_root, "csv_files/full_data_csv/data_with_v_h_80_89_validation.csv")
    test_csv = os.path.join(data_root, "csv_files/full_data_csv/data_with_v_h_90_99.csv")

    # Datasets
    train_ds = VideoAsterDataset(
        data_path=data_root,
        patients_csv=patients_csv,
        frames_csv=train_csv,
        num_frames=args.num_frames,
        stride=args.num_frames // 2,
        is_training=True
    )
    val_ds = VideoAsterDataset(
        data_path=data_root,
        patients_csv=patients_csv,
        frames_csv=val_csv,
        num_frames=args.num_frames,
        stride=args.num_frames,
        is_training=False
    )
    test_ds = VideoAsterDataset(
        data_path=data_root,
        patients_csv=patients_csv,
        frames_csv=test_csv,
        num_frames=args.num_frames,
        stride=args.num_frames,
        is_training=False
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train clips: {len(train_ds)} | Val clips: {len(val_ds)} | Test clips: {len(test_ds)}")

    # Create model
    save_dir = os.path.join("video_benchmark_results", f"{args.model}_benchmark")
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "training_log.txt")

    if args.model == "vistr":
        # Modified VisTR with ResNet-18 backbone (ImageNet-1K pretrained)
        model = get_modified_vistr(num_frames=args.num_frames, backbone="resnet18", device=str(device))
        print(">> VisTR initialized with ImageNet-1K pretrained ResNet-18 backbone!")
    elif args.model == "vistr_resnet50":
        # Modified VisTR with ResNet-50 backbone (ImageNet-1K pretrained)
        model = get_modified_vistr(num_frames=args.num_frames, backbone="resnet50", device=str(device))
        print(">> VisTR initialized with ImageNet-1K pretrained ResNet-50 backbone!")
    elif args.model == "vistr_resnet101":
        # Modified VisTR with ResNet-101 backbone (ImageNet-1K pretrained matching IEEE TUFFC paper)
        model = get_modified_vistr(num_frames=args.num_frames, backbone="resnet101", device=str(device))
        print(">> VisTR initialized with ImageNet-1K pretrained ResNet-101 backbone (Paper Spec)!")
    elif args.model == "video_swin":
        # VideoSwinUNet initialized with official 3D pretrained weights
        pretrained_path = "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/pretrained_weights/model_swinvit.pt"
        model = VideoSwinUNet(in_channels=3, out_channels=1, feature_size=48, pretrained_path=pretrained_path).to(device)
        print(">> Video-Swin initialized with official 3D pre-trained Swin Transformer weights!")
    elif args.model == "video_mamba":
        # VideoMambaUNet initialized with transfer weights from U-Mamba
        model = VideoMambaUNet(in_channels=3, out_channels=1, base_ch=64).to(device)
        pretrained_path = "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/pretrained_weights/video_mamba_pretrained_umamba.pth"
        if os.path.exists(pretrained_path):
            state = torch.load(pretrained_path, map_location=device, weights_only=False)
            model.load_state_dict(state, strict=False)
            print(f">> Video-Mamba loaded pretrained weights from: {pretrained_path}!")
        else:
            print(">> Pretrained weights not found, initializing randomly.")
    elif args.model == "timesformer":
        vit_pretrained = "/home/medimg/.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth"
        model = TimeSformerUNet(in_channels=3, out_channels=1, pretrained_path=vit_pretrained).to(device)
        print(f">> TimeSformer (Divided Space-Time Transformer) initialized with official ViT-B/16 pretrained weights ({vit_pretrained})!")
    elif args.model == "video_mask2former":
        resnet_pretrained = "/home/medimg/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"
        model = VideoMask2Former(in_channels=3, out_channels=1, pretrained_path=resnet_pretrained).to(device)
        print(f">> Video Mask2Former initialized with ImageNet-1K pretrained ResNet-50 backbone ({resnet_pretrained})!")
    elif args.model in ["tubedetr", "original_tubedetr", "tubedetr_original"]:
        mdetr_pretrained = "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/pretrained_weights/pretrained_resnet101_checkpoint.pth"
        res101_hub = "/home/medimg/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth"
        chosen_ckpt = mdetr_pretrained if os.path.exists(mdetr_pretrained) else res101_hub
        model = OriginalTubeDETR(in_channels=3, out_channels=1, hidden_dim=256, num_tube_queries=16, backbone_pretrained=chosen_ckpt).to(device)
        print(f">> Original TubeDETR initialized with pretrained ResNet-101 ({chosen_ckpt}) and Spatio-Temporal DETR Transformer!")
    elif args.model == "tubeformer":
        model = TubeFormer(in_channels=3, out_channels=1).to(device)
        print(">> TubeFormer initialized with 3D space-time tube cross-attention!")
    elif args.model == "sam2":
        res34_pretrained = "/home/medimg/.cache/torch/hub/checkpoints/resnet34-b627a593.pth"
        model = SAM2MemSAM(in_channels=3, out_channels=1, pretrained_path=res34_pretrained).to(device)
        print(f">> SAM2 / MemSAM initialized with ImageNet-1K pretrained ResNet-34 backbone ({res34_pretrained})!")
    elif args.model == "stcn":
        model = STCNXMemUNet(in_channels=3, out_channels=1).to(device)
        print(">> STCN / XMem initialized with space-time memory matching!")
    elif args.model == "flow_guided":
        model = FlowGuidedEchoUNet(in_channels=3, out_channels=1).to(device)
        print(">> Flow-Guided EchoUNet initialized with temporal warp attention!")
    else:
        model = get_video_model(args.model, in_channels=3, num_classes=1, num_frames=args.num_frames, device=str(device))

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model parameters: {param_count:.2f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Use bfloat16 on Ampere RTX 3090 GPUs to eliminate numerical overflows and reduce memory footprint
    use_bfloat16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(not use_bfloat16))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_dice = -1.0
    best_ckpt_path = os.path.join(save_dir, f"best_{args.model}.pth")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for x_clip, mask_clip, targets in train_loader:
            optimizer.zero_grad()
            B, C, T, H, W = x_clip.shape

            if args.model.startswith("vistr"):
                # For VisTR, interpolate frames to 224x160 with bfloat16/float16 AMP acceleration
                x_down = F.interpolate(x_clip, size=(T, 224, 160), mode='trilinear', align_corners=False)
                mask_down = F.interpolate(mask_clip, size=(T, 224, 160), mode='nearest')
                x_flat = x_down.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, 224, 160).to(device)
                flat_mask = torch.zeros((B * T, 224, 160), dtype=torch.bool, device=device)
                samples = NestedTensor(x_flat, flat_mask)

                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=True):
                    out = model(samples)
                    pred_masks = out["pred_masks"] # (B, num_queries, H_m, W_m)
                    pred_masks_up = F.interpolate(pred_masks, size=(224, 160), mode='bilinear', align_corners=False)
                    # Clamp raw logits to prevent exponential overflow in BCE / sigmoid
                    pred_masks_up = torch.clamp(pred_masks_up, min=-30.0, max=30.0)
                    tgt_masks = mask_down.squeeze(1).to(device) # (B, T, 224, 160)
                    loss = bce_loss(pred_masks_up, tgt_masks) + dice_loss_fn(pred_masks_up, tgt_masks)
            else:
                # Dense spatio-temporal video models (video_swin, video_mamba, timesformer, mask2former, tubeformer, etc.)
                x_clip_down = F.interpolate(x_clip, size=(T, 224, 160), mode='trilinear', align_corners=False).to(device)
                mask_clip_down = F.interpolate(mask_clip, size=(T, 224, 160), mode='nearest').to(device)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=True):
                    preds = model(x_clip_down) # (B, 1, T, 224, 160)
                    preds = torch.clamp(preds, min=-30.0, max=30.0)
                    loss = bce_loss(preds, mask_clip_down) + dice_loss_fn(preds, mask_clip_down)

            if use_bfloat16:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # Validation
        model.eval()
        val_dice_list = []
        with torch.no_grad():
            for x_clip, mask_clip, targets in val_loader:
                B, C, T, H, W = x_clip.shape
                if args.model.startswith("vistr"):
                    x_down = F.interpolate(x_clip, size=(T, 224, 160), mode='trilinear', align_corners=False)
                    x_flat = x_down.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, 224, 160).to(device)
                    flat_mask = torch.zeros((B * T, 224, 160), dtype=torch.bool, device=device)
                    samples = NestedTensor(x_flat, flat_mask)
                    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=True):
                        out = model(samples)
                    pred_masks = torch.sigmoid(torch.clamp(out["pred_masks"].float(), min=-30.0, max=30.0))
                if args.model.startswith("vistr"):
                    pred_masks_up = F.interpolate(pred_masks, size=(H, W), mode='bilinear', align_corners=False)
                    # Vectorized GPU Dice calculation
                    tgt_masks = mask_clip.squeeze(1).to(device)
                    pred_bin = (pred_masks_up > 0.5).float()
                    gt_bin = (tgt_masks > 0.5).float()
                    inter = (pred_bin * gt_bin).sum(dim=(1, 2, 3))
                    union = pred_bin.sum(dim=(1, 2, 3)) + gt_bin.sum(dim=(1, 2, 3))
                    batch_dice = (2.0 * inter + 1e-5) / (union + 1e-5)
                    val_dice_list.extend(batch_dice.cpu().numpy().tolist())
                else:
                    x_clip_down = F.interpolate(x_clip, size=(T, 224, 160), mode='trilinear', align_corners=False).to(device)
                    mask_down = F.interpolate(mask_clip, size=(T, 224, 160), mode='nearest').to(device)
                    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=True):
                        preds_down = torch.sigmoid(torch.clamp(model(x_clip_down), min=-30.0, max=30.0))
                    # Vectorized GPU Dice calculation
                    pred_bin = (preds_down > 0.5).float()
                    gt_bin = (mask_down > 0.5).float()
                    inter = (pred_bin * gt_bin).sum(dim=(1, 2, 3, 4))
                    union = pred_bin.sum(dim=(1, 2, 3, 4)) + gt_bin.sum(dim=(1, 2, 3, 4))
                    batch_dice = (2.0 * inter + 1e-5) / (union + 1e-5)
                    val_dice_list.extend(batch_dice.cpu().numpy().tolist())

        val_dice = np.mean(val_dice_list) * 100.0 if val_dice_list else 0.0
        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        log_str = f"Epoch [{epoch:02d}/{args.epochs:02d}] | Loss: {train_loss:.4f} | Val Dice: {val_dice:.2f}% | LR: {cur_lr:.2e} | Time: {elapsed:.1f}s"
        print(log_str)
        with open(log_file, "a") as f:
            f.write(log_str + "\n")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_dice": val_dice,
            "lr": cur_lr,
            "time": elapsed
        })
        pd.DataFrame(history).to_csv(os.path.join(save_dir, "epoch_training_history.csv"), index=False)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_ckpt_path)
            torch.save(model.state_dict(), os.path.join(save_dir, f"{args.model}_best_ep{epoch}.pth"))
            print(f"  >>> New Best Val Dice: {best_val_dice:.2f}% (Saved to {best_ckpt_path})")

    print(f"\nTraining completed! Evaluating best checkpoint on held-out test dataset (Patients 90-99)...")
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    model.eval()
    test_records = []
    with torch.no_grad():
        for x_clip, mask_clip, targets in test_loader:
            B, C, T, H, W = x_clip.shape
            pids = targets["patient_id"].numpy()
            f_indices = targets["frame_indices"].numpy()

            if args.model.startswith("vistr"):
                x_down = F.interpolate(x_clip, size=(T, 224, 160), mode='trilinear', align_corners=False)
                x_flat = x_down.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, 224, 160).to(device)
                flat_mask = torch.zeros((B * T, 224, 160), dtype=torch.bool, device=device)
                samples = NestedTensor(x_flat, flat_mask)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=True):
                    out = model(samples)
                pred_masks = torch.sigmoid(torch.clamp(out["pred_masks"].float(), min=-30.0, max=30.0))
                pred_masks_up = F.interpolate(pred_masks, size=(H, W), mode='bilinear', align_corners=False)
                pred_bin = (pred_masks_up > 0.5).cpu().numpy().astype(np.uint8)
                gt_bin = (mask_clip.squeeze(1) > 0.5).numpy().astype(np.uint8)
            else:
                x_clip_down = F.interpolate(x_clip, size=(T, 224, 160), mode='trilinear', align_corners=False).to(device)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=True):
                    preds_down = torch.sigmoid(torch.clamp(model(x_clip_down), min=-30.0, max=30.0))
                preds = F.interpolate(preds_down, size=(T, H, W), mode='trilinear', align_corners=False)
                pred_bin = (preds > 0.5).cpu().numpy().astype(np.uint8)
                gt_bin = (mask_clip > 0.5).numpy().astype(np.uint8)

            for b in range(B):
                for t in range(T):
                    pb = pred_bin[b, t] if args.model.startswith("vistr") else pred_bin[b, 0, t]
                    gb = gt_bin[b, t] if args.model.startswith("vistr") else gt_bin[b, 0, t]
                    rec = compute_detailed_metrics(pb, gb)
                    rec["patient_id"] = int(pids[b])
                    rec["frame_no"] = int(f_indices[b, t])
                    test_records.append(rec)

    test_df = pd.DataFrame(test_records)
    out_csv = os.path.join(save_dir, f"test_benchmark_{args.model}.csv")
    test_df.to_csv(out_csv, index=False)
    print("=" * 80)
    print(f"FINAL TEST BENCHMARK RESULTS FOR {args.model.upper()}:")
    print(f"Dice: {test_df['dice'].mean()*100:.2f}% +/- {test_df['dice'].std()*100:.2f}%")
    print(f"IoU: {test_df['iou'].mean()*100:.2f}% +/- {test_df['iou'].std()*100:.2f}%")
    print(f"Precision: {test_df['precision'].mean()*100:.2f}% +/- {test_df['precision'].std()*100:.2f}%")
    print(f"Recall: {test_df['recall'].mean()*100:.2f}% +/- {test_df['recall'].std()*100:.2f}%")
    print(f"CSA Error: {test_df['csa_err'].mean():.3f} +/- {test_df['csa_err'].std():.3f} mm^2")
    print(f"Saved results to: {out_csv}")
    print("=" * 80)

if __name__ == "__main__":
    main()
