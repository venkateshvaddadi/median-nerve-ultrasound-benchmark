#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Official 3,000-Frame Benchmark Evaluator for 2D Single-Frame Segmentation Models.
Direct counterpart to `evaluate_all_video_sota_models_3000.py`.

Supported Architectures (Table I Benchmark):
  1. segformer-b2    (NVIDIA MiT-B2)
  2. segformer-b0    (NVIDIA MiT-B0)
  3. segformer-b1    (NVIDIA MiT-B1)
  4. transunet       (ResNet-50 + ViT-12)
  5. swin-unet       (Microsoft Swin-T)
  6. nnformer        (3D/2D Interlaced Transformer)
  7. swin-unetr      (Hierarchical Swin)
  8. unetr           (ViT-B 16x16)
  9. vmunet          (Visual State Space / VSS)
 10. swin-umamba     (Swin + Mamba Hybrid)
 11. swin-umamba-plus (Hybrid Swin-VSS Block)
 12. mnsegnet        (RSU + Half-UNet Decoder)

Evaluates across all 10 independent test patients (Patients 90-99), 3,000 frames total,
under the Unified Golden Synergy Recipe:
 - Single-Channel CLAHE Input Normalization
 - Test-Time Augmentation (TTA: Original + Horizontal Flip)
 - Largest Connected Component Morphological Post-Cleaning
 - Calibrated CSA Calculation (0.004328254 mm^2/pixel)
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import label
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Workspace paths
sys.path.append("/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet")
sys.path.append("/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/sota_models")
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "sota_models")))

from train_golden_synergy_benchmark import (
    get_model,
    postprocess_largest_component,
    CALIB_CSA_MM2
)
from train_ablation_segformer import SegFormerWrapper, UltrasoundAblationDataset

# ==============================================================================
# Model Registry & Checkpoints
# ==============================================================================
MODEL_REGISTRY = {
    "segformer-b2": {
        "paradigm": "Vision Transformers",
        "type": "segformer",
        "spec": "nvidia/mit-b2",
        "checkpoints": [
            "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/ablation_results/EXP-20_segformer_b0/best_model.pth",
            "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/Aster_hosptial_median_nerve/CTS/experiement_10_ResUnet/ablation_results/EXP-20_segformer_b0/best_model.pth",
            "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/Aster_hosptial_median_nerve/CTS/experiement_10_ResUnet/EXP-20_best_model.pth"
        ]
    },
    "segformer-b0": {
        "paradigm": "Vision Transformers",
        "type": "segformer",
        "spec": "nvidia/mit-b0",
        "checkpoints": [
            "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/ablation_results/EXP-17_segformer_b0/best_model.pth",
            "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/Aster_hosptial_median_nerve/CTS/experiement_10_ResUnet/ablation_results/EXP-17_segformer_b0/best_model.pth"
        ]
    },
    "segformer-b1": {
        "paradigm": "Vision Transformers",
        "type": "segformer",
        "spec": "nvidia/mit-b1",
        "checkpoints": [
            "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/ablation_results/EXP-18_segformer_b0/best_model.pth",
            "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/Aster_hosptial_median_nerve/CTS/experiement_10_ResUnet/ablation_results/EXP-18_segformer_b0/best_model.pth"
        ]
    },
    "transunet": {
        "paradigm": "Hybrid CNN-Transformer",
        "type": "golden",
        "spec": "transunet",
        "checkpoints": [
            "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/golden_benchmark_results/transunet_golden/best_model.pth",
            "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/Aster_hosptial_median_nerve/CTS/experiement_10_ResUnet/golden_benchmark_results/transunet_golden/best_model.pth"
        ]
    },
    "swin-unet": {
        "paradigm": "Shifted-Window ViT",
        "type": "golden",
        "spec": "swin-unet",
        "checkpoints": [
            "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/golden_benchmark_results/swin-unet_golden/best_model.pth",
            "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/Aster_hosptial_median_nerve/CTS/experiement_10_ResUnet/golden_benchmark_results/swin-unet_golden/best_model.pth"
        ]
    },
    "nnformer": {
        "paradigm": "Medical Hierarchical ViT",
        "type": "golden",
        "spec": "nnformer",
        "checkpoints": [
            "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/local_nnformer_pretrained_test/nnformer_golden/best_model.pth",
            "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/Aster_hosptial_median_nerve/CTS/experiement_10_ResUnet/local_nnformer_pretrained_test/nnformer_golden/best_model.pth"
        ]
    },
    "swin-unetr": {
        "paradigm": "Medical Hierarchical ViT",
        "type": "golden",
        "spec": "swin-unetr",
        "checkpoints": [
            "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/golden_benchmark_results/swin-unetr_golden/best_model.pth",
            "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/Aster_hosptial_median_nerve/CTS/experiement_10_ResUnet/golden_benchmark_results/swin-unetr_golden/best_model.pth"
        ]
    },
    "unetr": {
        "paradigm": "Pure Vision Transformer",
        "type": "golden",
        "spec": "unetr",
        "checkpoints": [
            "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/golden_benchmark_results/unetr_golden/best_model.pth",
            "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/Aster_hosptial_median_nerve/CTS/experiement_10_ResUnet/golden_benchmark_results/unetr_golden/best_model.pth"
        ]
    },
    "vmunet": {
        "paradigm": "State-Space Models (Mamba)",
        "type": "golden",
        "spec": "vmunet",
        "checkpoints": [
            "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/golden_benchmark_results/vmunet_golden/best_model.pth",
            "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/Aster_hosptial_median_nerve/CTS/experiement_10_ResUnet/golden_benchmark_results/vmunet_golden/best_model.pth"
        ]
    },
    "swin-umamba": {
        "paradigm": "State-Space Models (Mamba)",
        "type": "golden",
        "spec": "swin-umamba",
        "checkpoints": [
            "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/golden_benchmark_results/swin-umamba_golden/best_model.pth",
            "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/Aster_hosptial_median_nerve/CTS/experiement_10_ResUnet/golden_benchmark_results/swin-umamba_golden/best_model.pth"
        ]
    },
    "swin-umamba-plus": {
        "paradigm": "State-Space Models (Mamba)",
        "type": "golden",
        "spec": "swin-umamba-plus",
        "checkpoints": [
            "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/golden_benchmark_results/swin-umamba-plus_golden/best_model.pth",
            "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/Aster_hosptial_median_nerve/CTS/experiement_10_ResUnet/golden_benchmark_results/swin-umamba-plus_golden/best_model.pth"
        ]
    },
    "mnsegnet": {
        "paradigm": "Proposed Architecture",
        "type": "golden",
        "spec": "mnseg-net",
        "checkpoints": [
            "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/classical_benchmark_results/mnsegnet_benchmark/mnsegnet_best_ep9.pth",
            "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/Aster_hosptial_median_nerve/CTS/experiement_10_ResUnet/classical_benchmark_results/mnsegnet_benchmark/mnsegnet_best_ep9.pth"
        ]
    }
}

MODEL_ALIASES = {
    "segformer_b0": "segformer-b0",
    "segformer_b1": "segformer-b1",
    "segformer_b2": "segformer-b2",
    "swin_unet": "swin-unet",
    "swin_unetr": "swin-unetr",
    "nn_former": "nnformer",
    "nn-former": "nnformer",
    "vm_unet": "vmunet",
    "vm-unet": "vmunet",
    "swin_umamba": "swin-umamba",
    "swin_umamba_plus": "swin-umamba-plus",
    "swin-umamba+": "swin-umamba-plus",
    "mnseg-net": "mnsegnet"
}

# ==============================================================================
# Metric Computation
# ==============================================================================
def compute_detailed_metrics(pred_bin, target_bin):
    TP = np.count_nonzero((pred_bin == 1) & (target_bin == 1))
    FP = np.count_nonzero((pred_bin == 1) & (target_bin == 0))
    FN = np.count_nonzero((pred_bin == 0) & (target_bin == 1))
    TN = np.count_nonzero((pred_bin == 0) & (target_bin == 0))

    prec = TP / max(1, (TP + FP))
    rec = TP / max(1, (TP + FN))
    dice = (2.0 * TP) / max(1, (2 * TP + FP + FN))
    iou = TP / max(1, (TP + FP + FN))
    acc = (TP + TN) / max(1, (TP + TN + FP + FN))

    # Real-world CSA computation solely on predicted mask
    csa_pred_mm2 = float(np.sum(pred_bin)) * CALIB_CSA_MM2
    csa_gt_mm2 = float(np.sum(target_bin)) * CALIB_CSA_MM2
    csa_abs_err_mm2 = abs(csa_pred_mm2 - csa_gt_mm2)

    return {
        "Dice": dice,
        "IoU": iou,
        "Precision": prec,
        "Recall": rec,
        "Accuracy": acc,
        "CSA_Pred_mm2": csa_pred_mm2,
        "CSA_GT_mm2": csa_gt_mm2,
        "CSA_Abs_Err_mm2": csa_abs_err_mm2
    }

# ==============================================================================
# Single Model Evaluator
# ==============================================================================
def evaluate_model(model_name, ckpt_path=None, batch_size=16, device_id=0, data_root=None, output_csv=None):
    norm_name = MODEL_ALIASES.get(model_name.lower().strip(), model_name.lower().strip())
    if norm_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model architecture '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}")

    info = MODEL_REGISTRY[norm_name]
    mtype = info["type"]
    spec = info["spec"]
    paradigm = info["paradigm"]

    # Locate checkpoint
    if ckpt_path is None:
        for cand in info["checkpoints"]:
            if os.path.exists(cand):
                ckpt_path = cand
                break
        if ckpt_path is None:
            raise FileNotFoundError(f"Could not locate checkpoint for {model_name}. Looked in: {info['checkpoints']}")

    # Locate dataset
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

    test_patients_csv = os.path.join(data_root, "csv_files/full_data_csv/patients_list_90_99.csv")
    test_images_csv = os.path.join(data_root, "csv_files/full_data_csv/data_with_v_h_90_99.csv")

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f">> Starting Full 3,000-Frame Benchmark for: {norm_name.upper()}")
    print(f">> Paradigm:   {paradigm}")
    print(f">> Checkpoint: {ckpt_path}")
    print(f">> Device:     {device}")
    print(f"========================================================")

    # Initialize model
    if mtype == "segformer":
        model = SegFormerWrapper(spec).to(device)
    else:
        model = get_model(spec).to(device)

    # Load weights
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]

    # Handle wrapper hierarchy (e.g. net. prefix for MNSegNetWrapper / U2NetWrapper)
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(ckpt.keys())
    if any(k.startswith("net.") for k in model_keys) and not any(k.startswith("net.") for k in ckpt_keys):
        ckpt = {("net." + k if "net." + k in model_keys else k): v for k, v in ckpt.items()}

    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # Dataloader using CLAHE dataset
    dataset = UltrasoundAblationDataset(data_root, test_patients_csv, test_images_csv,
                                        mode="test", aug_mode="none", prep_mode="clahe")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    records = []
    t0 = time.time()

    with torch.no_grad():
        for images, masks, p_ids, img_nos in tqdm(dataloader, desc=f"Evaluating {norm_name}"):
            images = images.to(device)

            # Test-Time Augmentation (Horizontal Flip)
            with torch.amp.autocast("cuda", enabled=True):
                out_orig = model(images)
                probs_orig = out_orig[1] if isinstance(out_orig, (tuple, list)) else torch.sigmoid(out_orig)

                images_flip = torch.flip(images, dims=[3])
                out_flip = model(images_flip)
                probs_flip = out_flip[1] if isinstance(out_flip, (tuple, list)) else torch.sigmoid(out_flip)
                probs_flip_unflipped = torch.flip(probs_flip, dims=[3])

                tta_probs = 0.5 * (probs_orig + probs_flip_unflipped)

            tta_np = tta_probs.cpu().float().numpy()
            masks_np = masks.cpu().float().numpy()

            for b in range(images.size(0)):
                raw_pred = (tta_np[b, 0] > 0.5).astype(np.uint8)
                clean_pred = postprocess_largest_component(raw_pred)
                gt = (masks_np[b, 0] > 0.5).astype(np.uint8)

                m = compute_detailed_metrics(clean_pred, gt)
                m["Patient_ID"] = p_ids[b].item() if hasattr(p_ids[b], "item") else int(p_ids[b])
                m["Image_No"] = img_nos[b].item() if hasattr(img_nos[b], "item") else int(img_nos[b])
                m["Model"] = norm_name
                records.append(m)

    elapsed = time.time() - t0
    df = pd.DataFrame(records)

    # Compute Table I Summary Metrics
    mean_dice = df["Dice"].mean() * 100.0
    mean_iou = df["IoU"].mean() * 100.0
    mean_prec = df["Precision"].mean() * 100.0
    mean_rec = df["Recall"].mean() * 100.0
    mean_csa_err = df["CSA_Abs_Err_mm2"].mean()

    # Wrist region (first 40 frames: Image_No < 40)
    wrist_df = df[df["Image_No"] < 40]
    wrist_dice = wrist_df["Dice"].mean() * 100.0 if len(wrist_df) > 0 else 0.0

    print(f"\n========================================================")
    print(f"BENCHMARK RESULTS FOR: {norm_name.upper()} ({paradigm})")
    print(f"========================================================")
    print(f"Test Frames Evaluated : {len(df):,}")
    print(f"Evaluation Time       : {elapsed:.2f}s ({len(df)/elapsed:.1f} FPS)")
    print(f"--------------------------------------------------------")
    print(f"Test Dice (%)         : {mean_dice:.2f}%")
    print(f"Test IoU (%)          : {mean_iou:.2f}%")
    print(f"Precision (%)         : {mean_prec:.2f}%")
    print(f"Recall (%)            : {mean_rec:.2f}%")
    print(f"CSA Error (mm^2)      : {mean_csa_err:.3f}")
    print(f"Wrist Dice (%)        : {wrist_dice:.2f}%")
    print(f"========================================================")

    if output_csv is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved detailed 3,000-frame metrics to: {output_csv}")

    return {
        "Paradigm": paradigm,
        "Model": norm_name,
        "Test Dice (%)": round(mean_dice, 2),
        "Test IoU (%)": round(mean_iou, 2),
        "Prec. (%)": round(mean_prec, 2),
        "Rec. (%)": round(mean_rec, 2),
        "CSA Err (mm2)": round(mean_csa_err, 3),
        "Wrist Dice (%)": round(wrist_dice, 2),
        "Frames": len(df)
    }

# ==============================================================================
# CLI Orchestrator
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified 3,000-Frame Benchmark Evaluator for 2D Segmentation Models")
    parser.add_argument("--model", type=str, default="all",
                        help=f"Model to evaluate or 'all'. Choices: {list(MODEL_REGISTRY.keys()) + ['all']}")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional custom checkpoint path. If omitted, default benchmark weights are loaded.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size (default: 16)")
    parser.add_argument("--data_root", type=str, default=None, help="Root folder of Aster ultrasound dataset")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional output CSV for frame metrics")
    parser.add_argument("--output_dir", type=str, default="benchmark_evaluation_results_2d",
                        help="Directory to save summary and per-model benchmark CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model.lower() == "all":
        models_to_run = list(MODEL_REGISTRY.keys())
        print(f"\n=========================================================================================")
        print(f">>> Running Unified Evaluation Across ALL {len(models_to_run)} Single-Frame Architectures (Table I) <<<")
        print(f"=========================================================================================")
        all_summaries = []
        for m in models_to_run:
            out_csv = os.path.join(args.output_dir, f"test_metrics_{m}.csv")
            try:
                summary = evaluate_model(m, ckpt_path=None, batch_size=args.batch_size,
                                         device_id=args.device, data_root=args.data_root,
                                         output_csv=out_csv)
                all_summaries.append(summary)
            except Exception as e:
                print(f"Error evaluating {m}: {e}")

        summary_df = pd.DataFrame(all_summaries)
        summary_path = os.path.join(args.output_dir, "benchmark_2d_architectures_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"\n=========================================================================================")
        print(f"OVERALL REPRODUCED BENCHMARK SUMMARY (TABLE I REPRODUCTION)")
        print(f"=========================================================================================")
        print(summary_df.to_string(index=False))
        print(f"\nSummary table saved to: {summary_path}")

    else:
        out_csv = args.output_csv or os.path.join(args.output_dir, f"test_metrics_{args.model.lower()}.csv")
        evaluate_model(args.model, ckpt_path=args.ckpt, batch_size=args.batch_size,
                       device_id=args.device, data_root=args.data_root,
                       output_csv=out_csv)

if __name__ == "__main__":
    main()
