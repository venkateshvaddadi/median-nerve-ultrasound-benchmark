import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append("/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet")
sys.path.append("/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/sota_models")

from train_golden_synergy_benchmark import (
    get_model,
    postprocess_largest_component,
    CALIB_CSA_MM2
)
from train_ablation_segformer import SegFormerWrapper, UltrasoundAblationDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_root = "/media/medimg/Data/venkatesh_data/Aster/data_making/aster_updated_data_nov_09_2022_with_flip/"

test_patients_csv = os.path.join(data_root, "csv_files/full_data_csv/patients_list_90_99.csv")
test_images_csv = os.path.join(data_root, "csv_files/full_data_csv/data_with_v_h_90_99.csv")

# Create dataset exactly as in benchmark
dataset = UltrasoundAblationDataset(data_root, test_patients_csv, test_images_csv,
                                    mode="test", aug_mode="none", prep_mode="clahe")

target_fnums = [10, 50, 90, 130, 170, 210, 250, 290]

# Map each patient (0 to 9 in test cohort, corresponding to Patients 90 to 99)
patient_ids = list(range(10)) # 0: P90, 1: P91, ..., 9: P99
patients_list_df = dataset.patients_list

# Preload all raw images, ground truth masks, and preprocessed tensors
raw_images = {p_idx: {} for p_idx in patient_ids}
gt_masks = {p_idx: {} for p_idx in patient_ids}
dataset_tensors = {p_idx: {} for p_idx in patient_ids}

frame_to_dataset_idx = {p_idx: {} for p_idx in patient_ids}
for idx in range(len(dataset.data)):
    p_id = dataset.data['patient_id'][idx]
    img_no = dataset.data['image_no'][idx]
    if p_id in patient_ids and img_no in target_fnums:
        frame_to_dataset_idx[p_id][img_no] = idx

print("Matched frame to dataset indices for all 10 test patients.")

for p_idx in patient_ids:
    pdir = patients_list_df['path'][p_idx].strip('/')
    real_pid = 90 + p_idx
    print(f"Loading data for Patient {real_pid} ({pdir})...")
    for fnum in target_fnums:
        ip = os.path.join(data_root, pdir, "images", f"{fnum}.jpg")
        im_pil = Image.open(ip).convert('RGB')
        im_np = np.array(im_pil)[4:452, 11:331, :]
        raw_images[p_idx][fnum] = im_np

        idx = frame_to_dataset_idx[p_idx][fnum]
        img_t, mask_t, _, _ = dataset[idx]
        gt_masks[p_idx][fnum] = (mask_t[0].numpy() > 0.5).astype(np.uint8)
        dataset_tensors[p_idx][fnum] = img_t.unsqueeze(0).to(device)

# The 9 Vision Transformer and Mamba benchmark architectures
models_to_run = [
    ("SegFormer-B2 (83.82% SOTA)", "segformer-b2", "segformer", "nvidia/mit-b2", "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/ablation_results/EXP-20_segformer_b0/best_model.pth"),
    ("TransUNet (83.04%)", "transunet", "golden", "transunet", "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/golden_benchmark_results/transunet_golden/best_model.pth"),
    ("VM-UNet (82.65%)", "vmunet", "golden", "vmunet", "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/golden_benchmark_results/vmunet_golden/best_model.pth"),
    ("Swin-UMamba+ (82.41%)", "swin-umamba-plus", "golden", "swin-umamba-plus", "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/golden_benchmark_results/swin-umamba-plus_golden/best_model.pth"),
    ("nnFormer (81.92%)", "nnformer", "golden", "nnformer", "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/local_nnformer_pretrained_test/nnformer_golden/best_model.pth"),
    ("Swin-UMamba (81.80%)", "swin-umamba", "golden", "swin-umamba", "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/golden_benchmark_results/swin-umamba_golden/best_model.pth"),
    ("Swin-UNet (81.66%)", "swin-unet", "golden", "swin-unet", "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/golden_benchmark_results/swin-unet_golden/best_model.pth"),
    ("UNETR (80.58%)", "unetr", "golden", "unetr", "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/golden_benchmark_results/unetr_golden/best_model.pth"),
    ("Swin-UNETR (79.79%)", "swin-unetr", "golden", "swin-unetr", "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/golden_benchmark_results/swin-unetr_golden/best_model.pth"),
]

# Predictions dict: predictions[p_idx][display_name][fnum]
predictions = {p_idx: {m[0]: {} for m in models_to_run} for p_idx in patient_ids}

# Load each model ONCE and run inference across all patients and target frames
for display_name, m_key, m_type, m_spec, ckpt_p in models_to_run:
    print(f"\n==========================================")
    print(f"Loading {display_name}...")
    if m_type == "segformer":
        model = SegFormerWrapper(m_spec).to(device)
    else:
        model = get_model(m_spec).to(device)

    sd = torch.load(ckpt_p, map_location=device)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    model_keys = set(model.state_dict().keys())
    if any(k.startswith("net.") for k in model_keys) and not any(k.startswith("net.") for k in sd.keys()):
        sd = {("net." + k if "net." + k in model_keys else k): v for k, v in sd.items()}

    model.load_state_dict(sd, strict=False)
    model.eval()

    with torch.no_grad():
        for p_idx in patient_ids:
            for fnum in target_fnums:
                inp = dataset_tensors[p_idx][fnum]
                with torch.amp.autocast('cuda', enabled=True):
                    out_orig = model(inp)
                    probs_orig = out_orig[1] if isinstance(out_orig, (tuple, list)) else torch.sigmoid(out_orig)

                    inp_flip = torch.flip(inp, dims=[3])
                    out_flip = model(inp_flip)
                    probs_flip = out_flip[1] if isinstance(out_flip, (tuple, list)) else torch.sigmoid(out_flip)
                    probs_flip_unflipped = torch.flip(probs_flip, dims=[3])

                    tta_probs = 0.5 * (probs_orig + probs_flip_unflipped)

                raw_pred = (tta_probs[0, 0].cpu().float().numpy() > 0.5).astype(np.uint8)
                clean_pred = postprocess_largest_component(raw_pred)
                predictions[p_idx][display_name][fnum] = clean_pred

    del model
    torch.cuda.empty_cache()

print("\nAll predictions computed across all 10 test patients!")

# Overlay and HUD function
kappa = CALIB_CSA_MM2

def overlay_pred_with_gt(img_rgb, pred_mask, gt_mask):
    vis = img_rgb.copy()
    h, w = vis.shape[:2]

    # 1. Ground Truth Contour FIRST (Solid Bright Green, thickness=2)
    if gt_mask is not None and gt_mask.sum() > 0:
        contours_gt, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours_gt, -1, (0, 255, 0), 2, cv2.LINE_AA)

    # 2. Prediction Contour SECOND (Solid Bright Red, thickness=2)
    if pred_mask is not None and pred_mask.sum() > 0:
        contours_pred, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours_pred, -1, (255, 30, 30), 2, cv2.LINE_AA)

    # 3. Compute Metrics for HUD badge
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total_area = pred_mask.sum() + gt_mask.sum()
    if total_area == 0:
        dsc = 1.000
    else:
        dsc = (2.0 * intersection) / (total_area + 1e-7)

    csa_gt = gt_mask.sum() * kappa
    csa_pred = pred_mask.sum() * kappa

    # 4. Inset Metric HUD box (Bottom-Right Corner) - Substantially enlarged for maximum legibility
    box_w = 195
    box_h = 105
    x1 = w - box_w - 6
    y1 = h - box_h - 6
    x2 = w - 6
    y2 = h - 6

    # Semi-transparent pitch black background
    overlay_box = vis.copy()
    cv2.rectangle(overlay_box, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay_box, 0.88, vis, 0.12, 0, vis)

    # Clean border around box
    cv2.rectangle(vis, (x1, y1), (x2, y2), (140, 140, 140), 1, cv2.LINE_AA)

    # Substantially enlarged font rendering (scale 0.74, bold thickness 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.74
    thick = 2

    t1 = f"DSC : {dsc:.3f}"
    cv2.putText(vis, t1, (x1 + 12, y1 + 32), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    t2 = f"CSA1: {csa_gt:.3f}"
    cv2.putText(vis, t2, (x1 + 12, y1 + 65), font, scale, (0, 255, 0), thick, cv2.LINE_AA)

    t3 = f"CSA2: {csa_pred:.3f}"
    cv2.putText(vis, t3, (x1 + 12, y1 + 97), font, scale, (255, 75, 75), thick, cv2.LINE_AA)

    return vis

out_fig_dir = "/media/medimg/Data/venkatesh_data/Aster/experiement_10_ResUnet/qualitative_panels"
os.makedirs(out_fig_dir, exist_ok=True)

# Generate and save panels for ALL patients (Patients 90 to 99)
num_rows = len(models_to_run)
num_cols = len(target_fnums)

for p_idx in patient_ids:
    real_pid = 90 + p_idx
    print(f"Generating panel figure for Patient {real_pid}...")
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2.0 * num_cols, 2.7 * num_rows), dpi=300)
    plt.subplots_adjust(wspace=0.01, hspace=0.015, left=0.052, right=0.995, top=0.975, bottom=0.005)

    for col_idx, fnum in enumerate(target_fnums):
        raw_img = raw_images[p_idx][fnum]
        gt_m = gt_masks[p_idx][fnum]

        if fnum == 10:
            pos_label = f"Frame {fnum}\n(Carpal Inlet)"
        elif fnum == 50:
            pos_label = f"Frame {fnum}\n(Wrist Crease)"
        elif fnum == 90:
            pos_label = f"Frame {fnum}\n(Mid Forearm)"
        elif fnum == 130:
            pos_label = f"Frame {fnum}\n(Deep Transit)"
        elif fnum == 170:
            pos_label = f"Frame {fnum}\n(Muscular Plane)"
        elif fnum == 210:
            pos_label = f"Frame {fnum}\n(Fascicular Tapering)"
        elif fnum == 250:
            pos_label = f"Frame {fnum}\n(Deep Forearm)"
        else:
            pos_label = f"Frame {fnum}\n(Proximal Boundary)"

        axes[0, col_idx].set_title(pos_label, fontsize=10.5, fontweight='bold', pad=8)

        for row_idx, (display_name, _, _, _, _) in enumerate(models_to_run):
            ax = axes[row_idx, col_idx]
            ax.axis('off')
            pred_m = predictions[p_idx][display_name][fnum]
            vis = overlay_pred_with_gt(raw_img, pred_m, gt_m)
            ax.imshow(vis)

    for row_idx, (display_name, _, _, _, _) in enumerate(models_to_run):
        axes[row_idx, 0].text(-0.065, 0.5, display_name, transform=axes[row_idx, 0].transAxes,
                              fontsize=11.0, fontweight='bold', va='center', ha='center', rotation=90)

    out_jpg = os.path.join(out_fig_dir, f"qualitative_2d_sota_patient{real_pid}.jpg")
    plt.savefig(out_jpg, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_jpg}")

print("\nAll patient panels successfully generated and saved!")
