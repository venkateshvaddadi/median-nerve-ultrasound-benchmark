#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Training Script for Classical Ultrasound Segmentation Baselines:
- U-Net
- SegNet
- ResUNet
- Attention U-Net
- UNet++ (NestedUNet)
- BASNet
- U2Net / U2NetP
- MNSeg-Net

Dataset: CTS Dataset (mydataloader)
Loss Function: BCE + SSIM + Soft IoU Loss (+ multi-scale deep supervision where applicable)
Parity: Uses the standard CTS benchmark split (Patients 1-79 Train, 80-89 Val, 90-99 Test)
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add package root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import get_classical_model
from datasets import mydataloader
from loss import SSIMLoss, DiceLoss

CALIB_CSA_MM2 = 0.004328254

def compute_metrics(pred_bin, target_bin):
    TP = np.count_nonzero((pred_bin == 1) & (target_bin == 1))
    FP = np.count_nonzero((pred_bin == 1) & (target_bin == 0))
    FN = np.count_nonzero((pred_bin == 0) & (target_bin == 1))
    TN = np.count_nonzero((pred_bin == 0) & (target_bin == 0))

    dice = (2.0 * TP) / max(1, (2 * TP + FP + FN))
    iou = TP / max(1, (TP + FP + FN))
    prec = TP / max(1, (TP + FP))
    rec = TP / max(1, (TP + FN))
    csa_pred = float(np.sum(pred_bin)) * CALIB_CSA_MM2
    csa_gt = float(np.sum(target_bin)) * CALIB_CSA_MM2
    csa_err = abs(csa_pred - csa_gt)

    return {"dice": dice, "iou": iou, "precision": prec, "recall": rec, "csa_err": csa_err}

def main():
    parser = argparse.ArgumentParser(description="Train Classical Segmentation Baselines")
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "segnet", "resunet", "attention-unet", "unet++", "basnet", "u2net", "mnseg-net"],
                        help="Model architecture name")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--data_path", type=str,
                        default="../data_making/aster_updated_data_nov_09_2022_with_flip/",
                        help="Path to dataset root")
    parser.add_argument("--patients_csv", type=str,
                        default="../data_making/aster_updated_data_nov_09_2022_with_flip/csv_files/patients_list_100.csv",
                        help="Patients CSV file")
    parser.add_argument("--train_csv", type=str,
                        default="../data_making/aster_updated_data_nov_09_2022_with_flip/csv_files/patients_images_train_data.csv",
                        help="Training images CSV")
    parser.add_argument("--val_csv", type=str,
                        default="../data_making/aster_updated_data_nov_09_2022_with_flip/csv_files/patients_images_val_data.csv",
                        help="Validation images CSV")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"=== Training Classical Baseline: {args.model.upper()} on {device} ===")

    # Initialize model
    model = get_classical_model(args.model, in_channels=3, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__} | Trainable Parameters: {total_params:,}")

    # Datasets and Loaders
    train_dataset = mydataloader(args.data_path, args.patients_csv, args.train_csv, training=True)
    val_dataset = mydataloader(args.data_path, args.patients_csv, args.val_csv, training=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Loss Functions
    bce = nn.BCELoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.save_dir, exist_ok=True)
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

        for images, masks in pbar:
            images = images.to(device).float()
            masks = masks.to(device).float()

            optimizer.zero_grad()
            out = model(images)

            # Handle multi-scale outputs from BASNet, U2Net, or MNSegNet
            if isinstance(out, (list, tuple)):
                pred = out[0]
                loss = 0.0
                for side_out in out:
                    if not (side_out.min() >= 0 and side_out.max() <= 1):
                        side_prob = torch.sigmoid(side_out)
                    else:
                        side_prob = side_out
                    loss += bce(side_prob, masks) + dice_loss(side_prob, masks)
            else:
                pred = out
                if not (pred.min() >= 0 and pred.max() <= 1):
                    pred_prob = torch.sigmoid(pred)
                else:
                    pred_prob = pred
                loss = bce(pred_prob, masks) + dice_loss(pred_prob, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        epoch_loss = running_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_dices = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                images = images.to(device).float()
                out = model(images)
                pred = out[0] if isinstance(out, (list, tuple)) else out
                pred_prob = torch.sigmoid(pred) if not (pred.min() >= 0 and pred.max() <= 1) else pred

                preds_np = (pred_prob.squeeze(1).cpu().numpy() > 0.5).astype(np.uint8)
                masks_np = (masks.squeeze(1).cpu().numpy() > 0.5).astype(np.uint8)

                for p_bin, m_bin in zip(preds_np, masks_np):
                    m = compute_metrics(p_bin, m_bin)
                    val_dices.append(m["dice"])

        val_mean_dice = np.mean(val_dices) * 100.0
        print(f"Epoch {epoch:02d} Summary | Train Loss: {epoch_loss:.4f} | Val Dice: {val_mean_dice:.2f}%")

        if val_mean_dice > best_dice:
            best_dice = val_mean_dice
            ckpt_path = os.path.join(args.save_dir, f"best_{args.model}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"==> Best model saved to {ckpt_path} with Val Dice {best_dice:.2f}%")

if __name__ == "__main__":
    main()
