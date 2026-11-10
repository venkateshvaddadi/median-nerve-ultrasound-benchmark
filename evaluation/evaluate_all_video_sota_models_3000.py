#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Spatio-Temporal Video SOTA Benchmark Evaluation Script
Evaluates all benchmarked spatio-temporal video architectures on the held-out test cohort
(Patients 90-99, all 3,000 independent continuous ultrasound frames).

Supports:
  1. 'tubedetr': Original TubeDETR (ResNet-101 + Spatio-Temporal DETR Transformer)
  2. 'memsam' / 'sam2': SAM 2 / MemSAM (Streaming Video Memory ViT)
  3. 'tubeformer': Modified TubeFormer (ResNet-34 + 3D Tube Attention Queries)
  4. 'vistr_resnet101': Modified VisTR (ResNet-101)
  5. 'vistr_resnet50': Modified VisTR (ResNet-50)
  6. 'video_mask2former': Video-Mask2Former (3D Mask Query Transformer)
  7. 'video_swin': VideoSwinUNet (3D Swin Transformer)
  8. 'timesformer': TimeSformer (Divided Space-Time Attention ViT)
  9. 'video_mamba': Spatio-Temporal Video-Mamba (MambaOut / Vivim SSM)

Computes:
  - Full-sweep Dice, IoU, Precision, Recall
  - Clinical Cross-Sectional Area (CSA in mm²) and absolute CSA error (|pred - gt|)
  - Distal Carpal Tunnel Wrist segment metrics (first 40 frames of each patient)
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent directory for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../spatio_temporal_video")))

from video_sota_models import (
    OriginalTubeDETR, TubeFormer, SAM2MemSAM, VideoMask2Former,
    VideoSwinUNet, TimeSformerUNet, VideoMambaUNet, get_modified_vistr
)
from spatio_temporal_video.video_dataset import VideoAsterDataset

CALIB_CSA_MM2 = 0.004328254 # Physical calibration factor (mm² per pixel)

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Video SOTA 3000-Frame Evaluation")
    parser.add_argument("--model", type=str, required=True,
                        choices=["tubedetr", "memsam", "sam2", "tubeformer",
                                 "vistr_resnet101", "vistr_resnet50", "video_mask2former",
                                 "video_swin", "timesformer", "video_mamba"],
                        help="Model architecture name to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint (.pth). If None, searches standard paths.")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to Aster ultrasound dataset directory.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size of 16-frame clips")
    parser.add_argument("--num_frames", type=int, default=16, help="Frames per clip (default: 16)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Destination CSV to save per-frame predictions and clinical metrics.")
    return parser.parse_args()

def locate_data_root(user_path=None):
    if user_path and os.path.exists(user_path):
        return user_path
    candidates = [
        "/media/medimg/Data/venkatesh_data/Aster/data_making/aster_updated_data_nov_09_2022_with_flip/",
        "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/Aster_hosptial_median_nerve/CTS/data_making/aster_updated_data_nov_09_2022_with_flip/",
        "../data_making/aster_updated_data_nov_09_2022_with_flip/"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not locate Aster dataset root. Please specify --data_root.")

def locate_checkpoint(model_name, user_ckpt=None):
    if user_ckpt and os.path.exists(user_ckpt):
        return user_ckpt
    
    ckpt_map = {
        "tubedetr": [
            "/home/medimg/video_benchmark_results/tubedetr_benchmark/best_tubedetr.pth",
            "video_benchmark_results/tubedetr_benchmark/best_tubedetr.pth"
        ],
        "memsam": [
            "/home/medimg/video_benchmark_results/sam2_benchmark/best_sam2.pth",
            "video_benchmark_results/sam2_benchmark/best_sam2.pth"
        ],
        "sam2": [
            "/home/medimg/video_benchmark_results/sam2_benchmark/best_sam2.pth",
            "video_benchmark_results/sam2_benchmark/best_sam2.pth"
        ],
        "tubeformer": [
            "/home/medimg/video_benchmark_results/tubeformer_benchmark/best_tubeformer.pth",
            "video_benchmark_results/tubeformer_benchmark/best_tubeformer.pth"
        ],
        "vistr_resnet101": [
            "/home/medimg/video_benchmark_results/vistr_resnet101_benchmark/best_vistr_resnet101.pth",
            "video_benchmark_results/vistr_resnet101_benchmark/best_vistr_resnet101.pth"
        ],
        "vistr_resnet50": [
            "/home/medimg/video_benchmark_results/vistr_resnet50_benchmark/best_vistr_resnet50.pth",
            "video_benchmark_results/vistr_resnet50_benchmark/best_vistr_resnet50.pth"
        ],
        "video_mask2former": [
            "/home/medimg/video_benchmark_results/video_mask2former_benchmark/best_video_mask2former.pth",
            "video_benchmark_results/video_mask2former_benchmark/best_video_mask2former.pth"
        ],
        "video_swin": [
            "/home/medimg/video_benchmark_results/video_swin_benchmark/best_video_swin.pth",
            "video_benchmark_results/video_swin_benchmark/best_video_swin.pth"
        ],
        "timesformer": [
            "/home/medimg/video_benchmark_results/timesformer_benchmark/best_timesformer.pth",
            "video_benchmark_results/timesformer_benchmark/best_timesformer.pth"
        ],
        "video_mamba": [
            "/home/medimg/video_benchmark_results/video_mamba_benchmark/best_video_mamba.pth",
            "video_benchmark_results/video_mamba_benchmark/best_video_mamba.pth"
        ]
    }
    
    for path in ckpt_map.get(model_name, []):
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Checkpoint for {model_name} not found. Please specify --checkpoint <path>.")

def load_model(model_name, ckpt_path, device):
    print(f">> Initializing architecture: {model_name}...")
    if model_name == "tubedetr":
        model = OriginalTubeDETR(num_frames=16, in_channels=3, num_classes=1, pretrained=False)
    elif model_name in ["memsam", "sam2"]:
        model = SAM2MemSAM(num_frames=16, in_channels=3, num_classes=1)
    elif model_name == "tubeformer":
        model = TubeFormer(num_frames=16, in_channels=3, num_classes=1)
    elif model_name == "vistr_resnet101":
        model = get_modified_vistr(backbone="resnet101", num_frames=16)
    elif model_name == "vistr_resnet50":
        model = get_modified_vistr(backbone="resnet50", num_frames=16)
    elif model_name == "video_mask2former":
        model = VideoMask2Former(in_channels=3, out_channels=1)
    elif model_name == "video_swin":
        model = VideoSwinUNet(in_channels=3, out_channels=1, feature_size=48)
    elif model_name == "timesformer":
        model = TimeSformerUNet(in_channels=3, out_channels=1, pretrained=False)
    elif model_name == "video_mamba":
        model = VideoMambaUNet(in_channels=3, out_channels=1, pretrained=False)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    print(f">> Loading state_dict from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f">> Model successfully loaded ({params / 1e6:.2f} M parameters)!")
    return model

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    data_root = locate_data_root(args.data_root)
    ckpt_path = locate_checkpoint(args.model, args.checkpoint)

    test_csv = os.path.join(data_root, "csv_files/full_data_csv/data_with_v_h_90_99.csv")
    patients_csv = os.path.join(data_root, "csv_files/full_data_csv/patients_list_1_99.csv")

    test_ds = VideoAsterDataset(
        data_path=data_root,
        patients_csv=patients_csv,
        frames_csv=test_csv,
        num_frames=args.num_frames,
        stride=args.num_frames,
        is_training=False
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f">> Dataset loaded: {len(test_ds)} clips ({len(test_ds)*args.num_frames} frames across Patients 90-99).")

    model = load_model(args.model, ckpt_path, device)

    rows = []
    frame_idx = 0
    t0 = time.time()

    print(f">> Commencing evaluation on {device}...")
    with torch.no_grad():
        for x_clip, mask_clip, targets in test_loader:
            x_clip = x_clip.to(device)
            mask_clip = mask_clip.to(device)
            pids = targets["patient_id"].numpy()

            if args.model.startswith("vistr"):
                # NestedTensor forward for VisTR
                from models.VisTR.util.misc import NestedTensor
                B, C, T, H, W = x_clip.shape
                x_down = F.interpolate(x_clip, size=(T, 224, 160), mode='trilinear', align_corners=False)
                x_flat = x_down.permute(0, 2, 1, 3, 4).reshape(B * T, C, 224, 160)
                mask_t = torch.zeros((B * T, 224, 160), dtype=torch.bool, device=device)
                nested = NestedTensor(x_flat, mask_t)
                preds_masks = model(nested)
                preds = F.interpolate(preds_masks, size=(H, W), mode='bilinear', align_corners=False).view(B, T, 1, H, W).permute(0, 2, 1, 3, 4)
            else:
                B, C, T, H, W = x_clip.shape
                # Dense spatio-temporal video models accept (B, C, T, 224, 160)
                x_down = F.interpolate(x_clip, size=(T, 224, 160), mode='trilinear', align_corners=False).to(device)
                preds_down = model(x_down)
                preds = F.interpolate(preds_down, size=(T, H, W), mode='trilinear', align_corners=False)

            preds_sig = torch.sigmoid(preds)
            preds_bin = (preds_sig > 0.5).float()

            B, _, T, H, W = preds.shape
            for b in range(B):
                pid = str(pids[b])
                for t in range(T):
                    p_flat = preds_bin[b, 0, t].view(-1)
                    g_flat = mask_clip[b, 0, t].view(-1)

                    intersection = (p_flat * g_flat).sum().item()
                    total = p_flat.sum().item() + g_flat.sum().item()
                    dice = (2.0 * intersection / (total + 1e-6)) * 100.0
                    iou = (intersection / (total - intersection + 1e-6)) * 100.0

                    pred_csa = p_flat.sum().item() * CALIB_CSA_MM2
                    gt_csa = g_flat.sum().item() * CALIB_CSA_MM2
                    csa_err = abs(pred_csa - gt_csa)

                    prec = (intersection / (p_flat.sum().item() + 1e-6)) * 100.0
                    rec = (intersection / (g_flat.sum().item() + 1e-6)) * 100.0

                    rows.append({
                        "global_frame_idx": frame_idx,
                        "patient_id": pid,
                        "clip_frame": t,
                        "dice": dice,
                        "iou": iou,
                        "precision": prec,
                        "recall": rec,
                        "pred_csa": pred_csa,
                        "gt_csa": gt_csa,
                        "csa_error": csa_err
                    })
                    frame_idx += 1

    total_time = time.time() - t0
    fps = frame_idx / total_time
    df = pd.DataFrame(rows)

    out_csv = args.output_csv or f"test_benchmark_{args.model}_all3000.csv"
    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print(f"BENCHMARK EVALUATION SUMMARY: {args.model.upper()}")
    print("=" * 80)
    print(f"Total Frames Evaluated : {frame_idx} (Patients 90-99)")
    print(f"Mean Test Dice Score   : {df['dice'].mean():.2f}% +/- {df['dice'].std():.2f}%")
    print(f"Mean Test IoU Score    : {df['iou'].mean():.2f}% +/- {df['iou'].std():.2f}%")
    print(f"Mean Precision         : {df['precision'].mean():.2f}%")
    print(f"Mean Recall            : {df['recall'].mean():.2f}%")
    print(f"Mean Clinical CSA Error: {df['csa_error'].mean():.3f} +/- {df['csa_error'].std():.3f} mm2")
    print(f"Inference Throughput   : {fps:.1f} FPS")

    # Distal Carpal Tunnel Wrist Segment (first 40 frames of each patient)
    wrist_df = df[df['clip_frame'] < 40]
    print("-" * 80)
    print(f"Distal Wrist Mean Dice : {wrist_df['dice'].mean():.2f}% +/- {wrist_df['dice'].std():.2f}%")
    print(f"Distal Wrist CSA Error : {wrist_df['csa_error'].mean():.3f} +/- {wrist_df['csa_error'].std():.3f} mm2")
    print(f"Results CSV saved to   : {out_csv}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
