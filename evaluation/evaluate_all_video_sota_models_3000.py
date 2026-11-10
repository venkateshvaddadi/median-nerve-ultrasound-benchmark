#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Official 3,000-Frame Benchmark Evaluator for All Spatio-Temporal Video Models.
Exact identical script used to evaluate VisTR, TubeDETR, SAM2/MemSAM, Video-Mask2Former, and VideoSwin.

Evaluates across all 10 independent test patients (Patients 90-99), 300 continuous frames each.
Handles non-overlapping 16-frame clips + tail-anchored boundary slices [284..299].
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

# Add path for models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../spatio_temporal_video")))
sys.path.append("/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet")
sys.path.append("/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/sota_models")

try:
    from video_sota_models import (
        VideoSwinUNet, get_modified_vistr, VideoMask2Former, SAM2MemSAM,
        TubeFormer, STCNXMemUNet, TimeSformerUNet, VideoMambaUNet
    )
except ImportError:
    from sota_models.video_sota_models import (
        VideoSwinUNet, get_modified_vistr, VideoMask2Former, SAM2MemSAM,
        TubeFormer, STCNXMemUNet, TimeSformerUNet, VideoMambaUNet
    )

try:
    from video_sota_models import OriginalTubeDETR
except ImportError:
    try:
        from sota_models.video_sota_models import OriginalTubeDETR
    except ImportError:
        OriginalTubeDETR = None

try:
    from models.VisTR.util.misc import NestedTensor
except ImportError:
    NestedTensor = None

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

def evaluate_model(model_type, ckpt_path, device_id=0, data_root=None, output_csv=None):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f">> Starting Full 3,000-Frame Benchmark for: {model_type.upper()}")
    print(f">> Checkpoint: {ckpt_path}")
    print(f">> Device:     {device}")
    print(f"========================================================")

    if data_root is None:
        candidates = [
            "/media/medimg/Data/venkatesh_data/Aster/data_making/aster_updated_data_nov_09_2022_with_flip/",
            "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/Aster_hosptial_median_nerve/CTS/data_making/aster_updated_data_nov_09_2022_with_flip/",
            "../data_making/aster_updated_data_nov_09_2022_with_flip/"
        ]
        for c in candidates:
            if os.path.exists(c):
                data_root = c
                break

    patients_csv = os.path.join(data_root, "csv_files/full_data_csv/patients_list_1_99.csv")
    test_csv = os.path.join(data_root, "csv_files/full_data_csv/data_with_v_h_90_99.csv")
    patients_df = pd.read_csv(patients_csv)
    test_df = pd.read_csv(test_csv)

    # Initialize model
    if model_type == "video_swin":
        model = VideoSwinUNet(in_channels=3, out_channels=1, feature_size=48).to(device)
    elif model_type == "video_mask2former":
        model = VideoMask2Former(in_channels=3, out_channels=1).to(device)
    elif model_type in ["sam2", "memsam"]:
        model = SAM2MemSAM(in_channels=3, out_channels=1).to(device)
    elif model_type == "tubeformer":
        model = TubeFormer(in_channels=3, out_channels=1).to(device)
    elif model_type in ["tubedetr", "original_tubedetr"]:
        if OriginalTubeDETR is not None:
            model = OriginalTubeDETR(in_channels=3, out_channels=1, hidden_dim=256, num_tube_queries=16).to(device)
        else:
            model = TubeFormer(in_channels=3, out_channels=1).to(device)
    elif model_type == "stcn":
        model = STCNXMemUNet(in_channels=3, out_channels=1).to(device)
    elif model_type == "timesformer":
        model = TimeSformerUNet(in_channels=3, out_channels=1, pretrained=False).to(device)
    elif model_type == "video_mamba":
        model = VideoMambaUNet(in_channels=3, out_channels=1, pretrained=False).to(device)
    elif model_type == "vistr_resnet50":
        model = get_modified_vistr(num_frames=16, backbone="resnet50", device=str(device)).to(device)
    elif model_type == "vistr_resnet101":
        model = get_modified_vistr(num_frames=16, backbone="resnet101", device=str(device)).to(device)
    elif model_type == "vistr_resnet18":
        model = get_modified_vistr(num_frames=16, backbone="resnet18", device=str(device)).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()

    # Exact training normalization
    mean = torch.tensor((0.2316, 0.2316, 0.2316)).view(3, 1, 1).to(device)
    std = torch.tensor((0.2038, 0.2038, 0.2038)).view(3, 1, 1).to(device)

    all_metrics = []
    unique_local_pids = sorted(test_df['patient_id'].unique())
    t0 = time.time()

    with torch.no_grad():
        for local_pid in unique_local_pids:
            actual_pid = local_pid + 90 if local_pid < 10 else local_pid
            if actual_pid >= len(patients_df):
                continue
            pdir = str(patients_df.iloc[actual_pid]['path']).strip()

            patient_rows = test_df[test_df['patient_id'] == local_pid]
            available_frames = sorted(patient_rows['image_no'].unique())
            if len(available_frames) == 0:
                available_frames = list(range(300))

            evaluated_frames = set()

            # 1. Non-overlapping 16-frame chunks
            for start_idx in range(0, 288, 16):
                clip_frame_nums = available_frames[start_idx : start_idx + 16]
                if len(clip_frame_nums) < 16:
                    break

                frames_t, gts, valid_indices = [], [], []
                for fnum in clip_frame_nums:
                    ip = os.path.join(data_root, pdir, "images", f"{fnum}.jpg")
                    mp = os.path.join(data_root, pdir, "masks", f"{fnum}.tif")
                    if not os.path.exists(mp):
                        mp = os.path.join(data_root, pdir, "masks", f"{fnum}.png")

                    if os.path.exists(ip) and os.path.exists(mp):
                        im = np.array(Image.open(ip).convert('RGB'))[4:452, 11:331, :]
                        gt = (np.array(Image.open(mp))[4:452, 11:331] > 127).astype(np.uint8)
                    else:
                        im = np.zeros((448, 320, 3), dtype=np.uint8)
                        gt = np.zeros((448, 320), dtype=np.uint8)

                    it = torch.from_numpy(im).permute(2, 0, 1).float().to(device) / 255.0
                    it = (it - mean) / std
                    frames_t.append(it)
                    gts.append(gt)
                    valid_indices.append(fnum)

                x_clip = torch.stack(frames_t, dim=1).unsqueeze(0) # (1, 3, 16, 448, 320)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    if model_type in ["video_swin", "video_mask2former", "sam2", "memsam", "tubeformer", "tubedetr", "original_tubedetr", "stcn", "timesformer", "video_mamba"]:
                        x_down = F.interpolate(x_clip, size=(16, 224, 160), mode='trilinear', align_corners=False)
                        pred = model(x_down)
                        pred_up = F.interpolate(pred, size=(16, 448, 320), mode='trilinear', align_corners=False)
                        prob = torch.sigmoid(torch.clamp(pred_up, min=-30.0, max=30.0)).squeeze(0).squeeze(0)
                        pred_bin = (prob > 0.5).cpu().numpy().astype(np.uint8)
                    else:
                        # Modified VisTR
                        x_down = F.interpolate(x_clip, size=(16, 224, 160), mode='trilinear', align_corners=False)
                        x_flat = x_down.permute(0, 2, 1, 3, 4).contiguous().view(16, 3, 224, 160)
                        flat_mask = torch.zeros((16, 224, 160), dtype=torch.bool, device=device)
                        samples = NestedTensor(x_flat, flat_mask)
                        out = model(samples)
                        pred_masks = torch.sigmoid(torch.clamp(out["pred_masks"].float(), min=-30.0, max=30.0))
                        pred_masks_up = F.interpolate(pred_masks, size=(448, 320), mode='bilinear', align_corners=False)
                        pred_bin = (pred_masks_up > 0.5).squeeze(0).cpu().numpy().astype(np.uint8)

                for t_idx, fnum in enumerate(valid_indices):
                    if fnum not in evaluated_frames:
                        m = compute_detailed_metrics(pred_bin[t_idx], gts[t_idx])
                        m["patient_id"] = actual_pid
                        m["image_no"] = fnum
                        all_metrics.append(m)
                        evaluated_frames.add(fnum)

            # 2. Tail-anchored clip [284..299]
            tail_indices = available_frames[284:300]
            if len(tail_indices) == 16:
                frames_t, gts = [], []
                for fnum in tail_indices:
                    ip = os.path.join(data_root, pdir, "images", f"{fnum}.jpg")
                    mp = os.path.join(data_root, pdir, "masks", f"{fnum}.tif")
                    if not os.path.exists(mp):
                        mp = os.path.join(data_root, pdir, "masks", f"{fnum}.png")

                    if os.path.exists(ip) and os.path.exists(mp):
                        im = np.array(Image.open(ip).convert('RGB'))[4:452, 11:331, :]
                        gt = (np.array(Image.open(mp))[4:452, 11:331] > 127).astype(np.uint8)
                    else:
                        im = np.zeros((448, 320, 3), dtype=np.uint8)
                        gt = np.zeros((448, 320), dtype=np.uint8)

                    it = torch.from_numpy(im).permute(2, 0, 1).float().to(device) / 255.0
                    it = (it - mean) / std
                    frames_t.append(it)
                    gts.append(gt)

                x_clip = torch.stack(frames_t, dim=1).unsqueeze(0)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    if model_type in ["video_swin", "video_mask2former", "sam2", "memsam", "tubeformer", "tubedetr", "original_tubedetr", "stcn", "timesformer", "video_mamba"]:
                        x_down = F.interpolate(x_clip, size=(16, 224, 160), mode='trilinear', align_corners=False)
                        pred = model(x_down)
                        pred_up = F.interpolate(pred, size=(16, 448, 320), mode='trilinear', align_corners=False)
                        prob = torch.sigmoid(torch.clamp(pred_up, min=-30.0, max=30.0)).squeeze(0).squeeze(0)
                        pred_bin = (prob > 0.5).cpu().numpy().astype(np.uint8)
                    else:
                        x_down = F.interpolate(x_clip, size=(16, 224, 160), mode='trilinear', align_corners=False)
                        x_flat = x_down.permute(0, 2, 1, 3, 4).contiguous().view(16, 3, 224, 160)
                        flat_mask = torch.zeros((16, 224, 160), dtype=torch.bool, device=device)
                        samples = NestedTensor(x_flat, flat_mask)
                        out = model(samples)
                        pred_masks = torch.sigmoid(torch.clamp(out["pred_masks"].float(), min=-30.0, max=30.0))
                        pred_masks_up = F.interpolate(pred_masks, size=(448, 320), mode='bilinear', align_corners=False)
                        pred_bin = (pred_masks_up > 0.5).squeeze(0).cpu().numpy().astype(np.uint8)

                for t_idx, fnum in enumerate(tail_indices):
                    if fnum not in evaluated_frames:
                        m = compute_detailed_metrics(pred_bin[t_idx], gts[t_idx])
                        m["patient_id"] = actual_pid
                        m["image_no"] = fnum
                        all_metrics.append(m)
                        evaluated_frames.add(fnum)

            print(f">> Patient {actual_pid} (Local {local_pid}) evaluated: {len(evaluated_frames)} / 300 frames.")

    elapsed = time.time() - t0
    df_out = pd.DataFrame(all_metrics)
    csv_out = output_csv or f"/home/medimg/video_benchmark_results/test_benchmark_{model_type}_all3000.csv"
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    df_out.to_csv(csv_out, index=False)

    print("\n" + "=" * 80)
    print(f"FINAL 3,000-FRAME TEST RESULTS FOR: {model_type.upper()}")
    print("=" * 80)
    print(f"Total Evaluated Frames: {len(df_out)} / 3000")
    print(f"Test Dice:      {df_out['dice'].mean()*100:.2f}% ± {df_out['dice'].std()*100:.2f}%")
    print(f"Test IoU:       {df_out['iou'].mean()*100:.2f}% ± {df_out['iou'].std()*100:.2f}%")
    print(f"Test Precision: {df_out['precision'].mean()*100:.2f}% ± {df_out['precision'].std()*100:.2f}%")
    print(f"Test Recall:    {df_out['recall'].mean()*100:.2f}% ± {df_out['recall'].std()*100:.2f}%")
    print(f"Test CSA Error: {df_out['csa_err'].mean():.3f} mm² ± {df_out['csa_err'].std():.3f} mm²")
    print(f"Throughput:     {len(df_out)/elapsed:.1f} FPS")

    # Distal Carpal Tunnel Wrist Segment (image_no < 40)
    wrist_df = df_out[df_out["image_no"] < 40]
    print("-" * 80)
    print(f"Distal Wrist Mean Dice: {wrist_df['dice'].mean()*100:.2f}% ± {wrist_df['dice'].std()*100:.2f}%")
    print(f"Distal Wrist CSA Error: {wrist_df['csa_err'].mean():.3f} ± {wrist_df['csa_err'].std():.3f} mm²")
    print(f"Saved CSV report to: {csv_out}")
    print("=" * 80 + "\n")
    return df_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified 3000-Frame Evaluation for Video SOTA Models")
    parser.add_argument("--model", type=str, required=True,
                        choices=["video_swin", "video_mask2former", "sam2", "memsam", "tubeformer",
                                 "tubedetr", "original_tubedetr", "stcn", "timesformer", "video_mamba",
                                 "vistr_resnet50", "vistr_resnet101", "vistr_resnet18"])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    # Default checkpoint locations
    if args.ckpt is None:
        ckpt_map = {
            "vistr_resnet101": "/home/medimg/video_benchmark_results/vistr_resnet101_benchmark/best_vistr_resnet101.pth",
            "vistr_resnet50": "/home/medimg/video_benchmark_results/vistr_resnet50_benchmark/best_vistr_resnet50.pth",
            "vistr_resnet18": "/home/medimg/video_benchmark_results/vistr_resnet18_benchmark/best_vistr_resnet18.pth",
            "video_mask2former": "/home/medimg/video_benchmark_results/video_mask2former_benchmark/best_video_mask2former.pth",
            "sam2": "/home/medimg/video_benchmark_results/sam2_benchmark/best_sam2.pth",
            "memsam": "/home/medimg/video_benchmark_results/sam2_benchmark/best_sam2.pth",
            "tubeformer": "/home/medimg/video_benchmark_results/tubeformer_benchmark/best_tubeformer.pth",
            "tubedetr": "/home/medimg/video_benchmark_results/tubedetr_benchmark/best_tubedetr.pth",
            "original_tubedetr": "/home/medimg/video_benchmark_results/tubedetr_benchmark/best_tubedetr.pth",
            "video_swin": "/home/medimg/video_benchmark_results/video_swin_benchmark/best_video_swin.pth",
            "video_mamba": "/home/medimg/video_benchmark_results/video_mamba_benchmark/best_video_mamba.pth",
            "timesformer": "/home/medimg/video_benchmark_results/timesformer_benchmark/best_timesformer.pth"
        }
        args.ckpt = ckpt_map[args.model]

    evaluate_model(args.model, args.ckpt, args.device, args.data_root, args.output_csv)
