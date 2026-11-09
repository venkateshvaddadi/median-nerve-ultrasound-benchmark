#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate VideoSwinUNet on all 3,000 test frames (Patients 90-99)
Handles stride 16 non-overlapping clips + tail-anchored boundary slices [284..299].
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F

from sota_models.video_sota_models import VideoSwinUNet

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

def main():
    device_id = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f">> Evaluating VideoSwinUNet on device: {device}")

    # Dataset paths
    candidate_roots = [
        "/media/medimg/Data/venkatesh_data/Aster/data_making/aster_updated_data_nov_09_2022_with_flip/",
        "../data_making/aster_updated_data_nov_09_2022_with_flip/",
        "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/Aster_hosptial_median_nerve/CTS/data_making/aster_updated_data_nov_09_2022_with_flip/"
    ]
    data_root = None
    for cand in candidate_roots:
        if os.path.exists(cand):
            data_root = cand
            break
    if not data_root:
        raise FileNotFoundError("Could not find aster data root")

    patients_csv = os.path.join(data_root, "csv_files/full_data_csv/patients_list_1_99.csv")
    test_csv = os.path.join(data_root, "csv_files/full_data_csv/data_with_v_h_90_99.csv")
    patients_df = pd.read_csv(patients_csv)
    test_df = pd.read_csv(test_csv)

    # Checkpoint
    ckpt_path = "video_benchmark_results/video_swin_benchmark/best_video_swin.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/video_benchmark_results/video_swin_benchmark/best_video_swin.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "./best_video_swin.pth"

    print(f">> Loading checkpoint: {ckpt_path}")
    model = VideoSwinUNet(in_channels=3, out_channels=1, feature_size=48).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    mean = torch.tensor((0.2316, 0.2316, 0.2316)).view(3, 1, 1).to(device)
    std = torch.tensor((0.2038, 0.2038, 0.2038)).view(3, 1, 1).to(device)

    all_metrics = []

    # Patients 90 to 99
    unique_local_pids = sorted(test_df['patient_id'].unique())
    print(f">> Found {len(unique_local_pids)} test patients: {unique_local_pids}")

    with torch.no_grad():
        for local_pid in unique_local_pids:
            actual_pid = local_pid + 90 if local_pid < 10 else local_pid
            if actual_pid >= len(patients_df):
                continue
            pdir = str(patients_df.iloc[actual_pid]['path']).strip()

            # Sequence of 300 frames
            patient_rows = test_df[test_df['patient_id'] == local_pid]
            available_frames = sorted(patient_rows['image_no'].unique())
            if len(available_frames) == 0:
                available_frames = list(range(300))

            evaluated_frames = set()

            # 1. Non-overlapping 16-frame chunks (0..15, 16..31, ..., 272..287)
            for start_idx in range(0, 288, 16):
                clip_frame_nums = available_frames[start_idx : start_idx + 16]
                if len(clip_frame_nums) < 16:
                    break

                frames_t = []
                gts = []
                valid_indices = []

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

                # Volume: (1, 3, 16, 448, 320) -> resample to (1, 3, 16, 224, 160)
                x_clip = torch.stack(frames_t, dim=1).unsqueeze(0)
                x_down = F.interpolate(x_clip, size=(16, 224, 160), mode='trilinear', align_corners=False)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(x_down) # (1, 1, 16, 224, 160)
                    pred_up = F.interpolate(pred, size=(16, 448, 320), mode='trilinear', align_corners=False)
                    prob = torch.sigmoid(pred_up).squeeze(0).squeeze(0) # (16, 448, 320)
                    pred_bin = (prob > 0.5).cpu().numpy().astype(np.uint8)

                for t_idx, fnum in enumerate(valid_indices):
                    if fnum not in evaluated_frames:
                        m = compute_detailed_metrics(pred_bin[t_idx], gts[t_idx])
                        m["patient_id"] = actual_pid
                        m["image_no"] = fnum
                        all_metrics.append(m)
                        evaluated_frames.add(fnum)

            # 2. Tail-anchored clip [284..299] (16 frames)
            tail_indices = available_frames[284:300]
            if len(tail_indices) == 16:
                frames_t = []
                gts = []
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
                x_down = F.interpolate(x_clip, size=(16, 224, 160), mode='trilinear', align_corners=False)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(x_down)
                    pred_up = F.interpolate(pred, size=(16, 448, 320), mode='trilinear', align_corners=False)
                    prob = torch.sigmoid(pred_up).squeeze(0).squeeze(0)
                    pred_bin = (prob > 0.5).cpu().numpy().astype(np.uint8)

                # Record remaining frames (288..299)
                for t_idx, fnum in enumerate(tail_indices):
                    if fnum not in evaluated_frames:
                        m = compute_detailed_metrics(pred_bin[t_idx], gts[t_idx])
                        m["patient_id"] = actual_pid
                        m["image_no"] = fnum
                        all_metrics.append(m)
                        evaluated_frames.add(fnum)

            print(f"Patient {actual_pid} (Local {local_pid}) finished: {len(evaluated_frames)} / 300 frames evaluated.")

    df_out = pd.DataFrame(all_metrics)
    csv_save_path = "video_benchmark_results/video_swin_benchmark/test_benchmark_video_swin_all3000.csv"
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    df_out.to_csv(csv_save_path, index=False)

    print("\n" + "=" * 80)
    print("FINAL 3,000-FRAME TEST RESULTS: VIDEO_SWIN_UNET")
    print("=" * 80)
    print(f"Total Evaluated Frames: {len(df_out)} / 3000")
    print(f"Test Dice:      {df_out['dice'].mean()*100:.2f}% ± {df_out['dice'].std()*100:.2f}%")
    print(f"Test IoU:       {df_out['iou'].mean()*100:.2f}% ± {df_out['iou'].std()*100:.2f}%")
    print(f"Test Precision: {df_out['precision'].mean()*100:.2f}% ± {df_out['precision'].std()*100:.2f}%")
    print(f"Test Recall:    {df_out['recall'].mean()*100:.2f}% ± {df_out['recall'].std()*100:.2f}%")
    print(f"Test CSA Error: {df_out['csa_err'].mean():.3f} mm² ± {df_out['csa_err'].std():.3f} mm²")
    print("=" * 80)

if __name__ == "__main__":
    main()
