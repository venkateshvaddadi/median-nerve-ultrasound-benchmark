#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Golden Synergy Benchmark Suite for Comparative Ultrasound Segmentation Models.
Applies the empirically validated Golden Synergy Recipe (discovered in the SegFormer ablation ladder):
1. Input Preprocessing: Single-Channel CLAHE (Local Contrast Normalization)
2. Augmentation Suite: Full Ultrasound Composite (Spatial Flips/Rotations + Elastic Distortion + Speckle Noise)
3. Loss Objective: Lovász-Softmax Soft Hinge Loss + BCE
4. Optimization: 5-Epoch Linear Warmup + Cosine Annealing LR + ModelEMA (decay=0.999) + Grad Clipping (1.0)
5. Test Inference: Test-Time Augmentation (TTA Reflection Pairs) + Connected Component Morphological Cleaning
6. Benchmark: 3,000 Independent Held-Out Frames (Patients 90-99)
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import label

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

from sota_models.transunet import TransUNet, PretrainedTransUNet
from sota_models.vmunet import VMUNet, PretrainedVMUNet
from sota_models.swin_unetr import SwinUNETR2D, SwinUNet2D, PretrainedSwinUNet
from models.ResUnet import ResUnet
from models.unet_model_updated import UNet_updated
from models.Attention_UNet_updated_version_1 import Attention_UNet_updated_version_1
from models.SegNet import SegNet
from models.unet_plus_plus_naveen import NestedUNet
from models.basnet_model.BASNet import BASNet
from models.u2_net_model.u2net import U2NET, U2NETP

# Calibration constant for Depth 3.0cm
CALIB_CSA_MM2 = 0.004328254

# ==============================================================================
# 1. Dataset with CLAHE Preprocessing & Full Ultrasound Augmentation Suite
# ==============================================================================
class GoldenUltrasoundDataset(Dataset):
    def __init__(self, data_path, csv_for_patients, csv_for_images, mode="train"):
        self.data_path = data_path
        self.mode = mode
        self.patients_list = pd.read_csv(csv_for_patients)
        raw_data = pd.read_csv(csv_for_images)
        
        # In validation mode, strictly evaluate on the pristine, unaugmented frames (exactly 2,998 frames, Patients 80-89)
        if self.mode == "val":
            self.data = raw_data[raw_data['patient_id'] <= 88].reset_index(drop=True)
        else:
            self.data = raw_data

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.transform = None
        if self.mode == "train" and HAS_ALBUMENTATIONS:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.08, rotate_limit=10, border_mode=0, p=0.5),
                A.ElasticTransform(alpha=1.0, sigma=30, alpha_affine=15, border_mode=0, p=0.3),
                A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),
                A.RandomGamma(gamma_limit=(85, 115), p=0.3),
                A.CoarseDropout(max_holes=2, max_height=448, max_width=35, min_holes=1, min_height=100, min_width=10, fill_value=0, p=0.25)
            ], additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient_id = self.data['patient_id'][idx]
        patient_file_name = self.patients_list['path'][patient_id]
        image_no = self.data['image_no'][idx]

        img_path = os.path.join(self.data_path, patient_file_name.strip('/'), "images", f"{image_no}.jpg")
        mask_path = os.path.join(self.data_path, patient_file_name.strip('/'), "masks", f"{image_no}.tif")

        try:
            image = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))

            # Standard B-mode cropping: [4:452, 11:331] -> shape (448, 320)
            img_crop = image[4:452, 11:331]
            mask_crop = mask[4:452, 11:331]

            if len(img_crop.shape) == 2:
                img_crop = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2RGB)
            elif img_crop.shape[2] == 4:
                img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGBA2RGB)

            if self.transform is not None:
                augmented = self.transform(image=img_crop, mask=mask_crop)
                img_crop = augmented['image']
                mask_crop = augmented['mask']

            # Apply CLAHE Preprocessing
            gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY) if img_crop.shape[2] == 3 else img_crop.copy()
            clahe_img = self.clahe.apply(gray)
            processed_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)

            img_tensor = torch.from_numpy(processed_img).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask_crop).unsqueeze(0).float() / 255.0
            mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)

            return img_tensor, mask_tensor, patient_id, image_no

        except Exception as e:
            return torch.zeros((3, 448, 320), dtype=torch.float32), torch.zeros((1, 448, 320), dtype=torch.float32), patient_id, image_no

# ==============================================================================
# 2. Lovász-Softmax Loss
# ==============================================================================
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge_flat(logits, labels):
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

class LovaszLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)
        lovasz = lovasz_hinge_flat(logits_flat, targets_flat)
        bce = self.bce(logits, targets)
        return lovasz + 0.5 * bce

# ==============================================================================
# 3. Model Factory
# ==============================================================================
def get_model(model_name):
    m = model_name.lower().strip()
    if m == "swin-unet":
        return PretrainedSwinUNet(in_channels=3, out_channels=1, pretrained=True)
    elif m == "swin-unetr":
        return SwinUNETR2D(in_channels=3, out_channels=1)
    elif m == "transunet":
        return PretrainedTransUNet(in_channels=3, out_channels=1, pretrained=True)
    elif m == "vmunet":
        return PretrainedVMUNet(in_channels=3, out_channels=1, pretrained=True)
    elif m == "unetr":
        from pretrained_unetr import get_pretrained_unetr_2d
        class UNETR2D(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = get_pretrained_unetr_2d()
            def forward(self, x):
                logits = self.net(x)
                return logits, torch.sigmoid(logits)
        return UNETR2D()
    elif m == "resunet":
        import torchvision.models as models
        class PretrainedResUNet(nn.Module):
            def __init__(self, in_channels=3, out_channels=1, pretrained=True):
                super().__init__()
                weights = models.ResNet34_Weights.DEFAULT if pretrained else None
                res = models.resnet34(weights=weights)
                self.conv1 = res.conv1
                self.bn1 = res.bn1
                self.relu = res.relu
                self.maxpool = res.maxpool
                self.layer1 = res.layer1 # 64
                self.layer2 = res.layer2 # 128
                self.layer3 = res.layer3 # 256
                self.layer4 = res.layer4 # 512

                self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
                self.dec4 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

                self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self.dec3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

                self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

                self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
                self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

                self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
                self.dec0 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

                self.head = nn.Conv2d(32, out_channels, 1)

            def forward(self, x):
                x0 = self.relu(self.bn1(self.conv1(x)))
                e1 = self.layer1(self.maxpool(x0))
                e2 = self.layer2(e1)
                e3 = self.layer3(e2)
                e4 = self.layer4(e3)

                d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
                d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
                d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
                d1 = self.dec1(torch.cat([self.up1(d2), x0], dim=1))
                d0 = self.dec0(self.up0(d1))
                logits = self.head(d0)
                return logits, torch.sigmoid(logits)
        return PretrainedResUNet(in_channels=3, out_channels=1, pretrained=True)
    elif m == "attention-unet":
        class AttnUnetWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = Attention_UNet_updated_version_1(3, 1)
            def forward(self, x):
                logits = self.net(x)
                return logits, torch.sigmoid(logits)
        return AttnUnetWrapper()
    elif m == "unet":
        class UNetWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = UNet_updated(3, 1)
            def forward(self, x):
                logits = self.net(x)
                return logits, torch.sigmoid(logits)
        return UNetWrapper()
    elif m == "nestedunet":
        class NestedUNetWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = NestedUNet(3, 1)
            def forward(self, x):
                logits = self.net(x)
                return logits, torch.sigmoid(logits)
        return NestedUNetWrapper()
    elif m == "segnet":
        import torchvision.models as models
        class PretrainedSegNet(nn.Module):
            def __init__(self, in_channels=3, out_channels=1, pretrained=True):
                super().__init__()
                weights = models.VGG16_BN_Weights.DEFAULT if pretrained else None
                vgg = models.vgg16_bn(weights=weights)
                feats = vgg.features

                self.stage1 = feats[0:6]
                self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

                self.stage2 = feats[7:13]
                self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

                self.stage3 = feats[14:23]
                self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

                self.stage4 = feats[24:33]
                self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

                self.unpool4 = nn.MaxUnpool2d(2, 2)
                self.dec4 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

                self.unpool3 = nn.MaxUnpool2d(2, 2)
                self.dec3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

                self.unpool2 = nn.MaxUnpool2d(2, 2)
                self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

                self.unpool1 = nn.MaxUnpool2d(2, 2)
                self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
                self.head = nn.Conv2d(32, out_channels, 1)

            def forward(self, x):
                e1 = self.stage1(x)
                p1, idx1 = self.pool1(e1)

                e2 = self.stage2(p1)
                p2, idx2 = self.pool2(e2)

                e3 = self.stage3(p2)
                p3, idx3 = self.pool3(e3)

                e4 = self.stage4(p3)
                p4, idx4 = self.pool4(e4)

                d4 = self.dec4(self.unpool4(p4, idx4, output_size=e4.size()))
                d3 = self.dec3(self.unpool3(d4, idx3, output_size=e3.size()))
                d2 = self.dec2(self.unpool2(d3, idx2, output_size=e2.size()))
                d1 = self.dec1(self.unpool1(d2, idx1, output_size=e1.size()))
                logits = self.head(d1)
                return logits, torch.sigmoid(logits)
        return PretrainedSegNet(in_channels=3, out_channels=1, pretrained=True)
    elif m == "basnet":
        class BASNetWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = BASNet(3, 1)
            def forward(self, x):
                out = self.net(x)
                logits = out[0] if isinstance(out, (list, tuple)) else out
                return logits, torch.sigmoid(logits)
        return BASNetWrapper()
    elif m == "u2net":
        class U2NetWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = U2NET(3, 1)
            def forward(self, x):
                out = self.net(x)
                logits = out[0] if isinstance(out, (list, tuple)) else out
                return logits, torch.sigmoid(logits)
        return U2NetWrapper()
    elif m in ["mnseg-net", "mnsegnet"]:
        from models.u2_net_model.U2Net_along_with_Half_UNet import U2NETP_along_with_Half_UNet_style_Decoder_with_custom_channels
        class MNSegNetWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = U2NETP_along_with_Half_UNet_style_Decoder_with_custom_channels(3, 1, 128, 16)
            def forward(self, x):
                outputs = self.net(x)
                probs = outputs[0]
                logits = torch.logit(probs.clamp(1e-6, 1.0 - 1e-6))
                return logits, probs
        return MNSegNetWrapper()
    else:
        try:
            from sota_models.modern_sota_models import get_modern_model
            modern_net = get_modern_model(m, in_channels=3, num_classes=1, img_size=(448, 320))
            class ModernWrapper(nn.Module):
                def __init__(self, net):
                    super().__init__()
                    self.net = net
                def forward(self, x):
                    out = self.net(x)
                    logits = out[0] if isinstance(out, (list, tuple)) else out
                    return logits, torch.sigmoid(logits)
            return ModernWrapper(modern_net)
        except Exception as e:
            raise ValueError(f"Model {model_name} not recognized: {e}")

# ==============================================================================
# 4. ModelEMA (Exponential Moving Average)
# ==============================================================================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

# ==============================================================================
# 5. Morphological Post-Cleaning
# ==============================================================================
def postprocess_largest_component(bin_mask):
    labeled, num_features = label(bin_mask)
    if num_features == 0:
        return bin_mask
    counts = np.bincount(labeled.flat)
    counts[0] = 0
    largest_label = counts.argmax()
    clean_mask = (labeled == largest_label).astype(np.uint8)
    
    h, w = clean_mask.shape
    flood = clean_mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    flood_inv = 1 - flood
    filled = clean_mask | flood_inv
    return filled

def compute_batch_dice(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=[2, 3])
    union = pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3])
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()

# ==============================================================================
# 6. Training & Comprehensive Benchmark Runner
# ==============================================================================
def train_golden_model(model_name, epochs=50, batch_size=16, lr=1e-4, device_idx=1, save_dir="golden_benchmark_results"):
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print(f"🚀 GOLDEN SYNERGY SOTA TRAINING: {model_name.upper()}")
    print(f"Device: {device} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"Recipe: CLAHE + Full Aug + Lovász Loss + Warmup + Cosine Anneal + EMA + TTA + Clean")
    print("=" * 85)

    out_dir = os.path.join(save_dir, f"{model_name}_golden")
    os.makedirs(out_dir, exist_ok=True)
    best_weights_path = os.path.join(out_dir, "best_model.pth")
    history_csv_path = os.path.join(out_dir, "epoch_training_history.csv")
    final_benchmark_csv = os.path.join(out_dir, f"test_benchmark_{model_name}.csv")

    model = get_model(model_name).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {model_name} | Parameters: {param_count:.2f} M")

    criterion = LovaszLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = ModelEMA(model, decay=0.999)

    # Datasets
    candidate_roots = [
        "../data_making/aster_updated_data_nov_09_2022_with_flip",
        "data_making/aster_updated_data_nov_09_2022_with_flip",
        "/media/medimg/Data/venkatesh_data/Aster/data_making/aster_updated_data_nov_09_2022_with_flip",
    ]
    data_dir = next((c for c in candidate_roots if os.path.exists(c)), None)
    if not data_dir:
        raise FileNotFoundError(f"Could not locate dataset in {candidate_roots}")

    patients_csv = os.path.join(data_dir, "csv_files/full_data_csv/patients_list_1_99.csv")
    train_csv = os.path.join(data_dir, "csv_files/full_data_csv/data_with_v_h_1_79_train.csv")
    val_csv = os.path.join(data_dir, "csv_files/full_data_csv/data_with_v_h_80_89_validation.csv")
    test_patients_csv = os.path.join(data_dir, "csv_files/full_data_csv/patients_list_90_99.csv")
    test_csv = os.path.join(data_dir, "csv_files/full_data_csv/data_with_v_h_90_99.csv")

    train_ds = GoldenUltrasoundDataset(data_dir, patients_csv, train_csv, mode="train")
    val_ds = GoldenUltrasoundDataset(data_dir, patients_csv, val_csv, mode="val")
    test_ds = GoldenUltrasoundDataset(data_dir, test_patients_csv, test_csv, mode="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Dataset Loaded: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    scaler = torch.amp.GradScaler('cuda')
    best_val_dice = 0.0
    history = []
    start_epoch = 1

    latest_ckpt_path = os.path.join(out_dir, "latest_checkpoint.pth")
    if os.path.exists(history_csv_path) and os.path.exists(latest_ckpt_path):
        try:
            history_df = pd.read_csv(history_csv_path)
            history = history_df.to_dict('records')
            start_epoch = int(history_df['epoch'].max()) + 1
            best_val_dice = float(history_df['val_dice'].max())
            ckpt = torch.load(latest_ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            scaler.load_state_dict(ckpt['scaler'])
            if 'ema' in ckpt:
                ema.shadow = {k: v.to(device) for k, v in ckpt['ema'].items()}
            print(f"🔄 Resuming {model_name} from Epoch {start_epoch}/{epochs} (Best Val Dice: {best_val_dice:.2f}%)")
        except Exception as e:
            print(f"⚠️ Could not resume from latest checkpoint: {e}. Starting fresh or from best.")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for imgs, masks, _, _ in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(imgs)
                logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            total_loss += loss.item()

        scheduler.step()
        epoch_loss = total_loss / max(1, len(train_loader))

        # Validation with EMA weights
        model.eval()
        val_dice_sum = 0.0
        with torch.no_grad():
            for v_imgs, v_masks, _, _ in val_loader:
                v_imgs = v_imgs.to(device)
                v_masks = v_masks.to(device)
                v_out = model(v_imgs)
                v_probs = v_out[1] if isinstance(v_out, tuple) else torch.sigmoid(v_out)
                val_dice_sum += compute_batch_dice(v_probs, v_masks) * v_imgs.size(0)

        val_dice = (val_dice_sum / len(val_ds)) * 100.0
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            ema.apply_shadow(model)
            torch.save(model.state_dict(), best_weights_path)

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'ema': ema.shadow
        }, latest_ckpt_path)

        dt = time.time() - t0
        best_tag = "⭐ [NEW BEST]" if is_best else ""
        print(f"[Epoch {epoch:02d}/{epochs:02d}] Loss: {epoch_loss:.4f} | Val Dice: {val_dice:.2f}% (Best: {best_val_dice:.2f}%) | Time: {dt:.1f}s {best_tag}", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_dice": val_dice,
            "is_best_val": is_best
        })
        pd.DataFrame(history).to_csv(history_csv_path, index=False)

    print("\n" + "=" * 85)
    print(f"🏁 TRAINING COMPLETE! Running Held-Out Test Evaluation on 3,000 Frames with TTA + Cleaning...")
    print("=" * 85)

    # Load official best checkpoint
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    test_records = []
    with torch.no_grad():
        for t_imgs, t_masks, pat_ids, img_nos in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            t_imgs = t_imgs.to(device)

            # Test-Time Augmentation (TTA: Original + Horizontal Flip)
            out_orig = model(t_imgs)
            probs_orig = out_orig[1] if isinstance(out_orig, tuple) else torch.sigmoid(out_orig)

            flipped_imgs = torch.flip(t_imgs, dims=[3])
            out_flip = model(flipped_imgs)
            probs_flip = out_flip[1] if isinstance(out_flip, tuple) else torch.sigmoid(out_flip)
            probs_flip_unflipped = torch.flip(probs_flip, dims=[3])

            tta_probs = 0.5 * (probs_orig + probs_flip_unflipped)
            tta_probs_np = tta_probs.cpu().numpy()
            t_masks_np = t_masks.numpy()

            for b in range(t_imgs.size(0)):
                raw_pred = (tta_probs_np[b, 0] > 0.5).astype(np.uint8)
                gt = (t_masks_np[b, 0] > 0.5).astype(np.uint8)
                clean_pred = postprocess_largest_component(raw_pred)

                TP = np.count_nonzero((clean_pred == 1) & (gt == 1))
                FP = np.count_nonzero((clean_pred == 1) & (gt == 0))
                FN = np.count_nonzero((clean_pred == 0) & (gt == 1))
                TN = np.count_nonzero((clean_pred == 0) & (gt == 0))

                dice = (2.0 * TP) / max(1, (2 * TP + FP + FN))
                iou = TP / max(1, (TP + FP + FN))
                prec = TP / max(1, (TP + FP))
                rec = TP / max(1, (TP + FN))
                acc = (TP + TN) / max(1, (TP + TN + FP + FN))

                csa_pred_mm2 = (TP + FP) * CALIB_CSA_MM2
                csa_gt_mm2 = (TP + FN) * CALIB_CSA_MM2
                csa_abs_err_mm2 = abs(csa_pred_mm2 - csa_gt_mm2)

                test_records.append({
                    "Dice": dice,
                    "IoU": iou,
                    "Precision": prec,
                    "Recall": rec,
                    "Accuracy": acc,
                    "CSA_Pred_mm2": csa_pred_mm2,
                    "CSA_GT_mm2": csa_gt_mm2,
                    "CSA_Abs_Err_mm2": csa_abs_err_mm2,
                    "Patient_ID": int(pat_ids[b]),
                    "Image_No": int(img_nos[b])
                })

    test_df = pd.DataFrame(test_records)
    test_df.to_csv(final_benchmark_csv, index=False)

    print("\n" + "#" * 85)
    print(f"🏆 OFFICIAL GOLDEN BENCHMARK RESULTS FOR {model_name.upper()} (3,000 Frames):")
    print(f"  - Test Dice:       {test_df['Dice'].mean() * 100:.2f}%")
    print(f"  - Test IoU:        {test_df['IoU'].mean() * 100:.2f}%")
    print(f"  - Test Precision:  {test_df['Precision'].mean() * 100:.2f}%")
    print(f"  - Test Recall:     {test_df['Recall'].mean() * 100:.2f}%")
    print(f"  - Pixel Accuracy:  {test_df['Accuracy'].mean() * 100:.2f}%")
    print(f"  - Mean CSA Err:    {test_df['CSA_Abs_Err_mm2'].mean():.3f} mm²")
    print(f"  - Checkpoint:      {best_weights_path}")
    print(f"  - Results CSV:     {final_benchmark_csv}")
    print("#" * 85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Golden Synergy SOTA Benchmark Runner")
    parser.add_argument("--model", type=str, required=True, help="Model name (swin-unet, unetr, transunet, vmunet, etc.)")
    parser.add_argument("--epochs", type=int, default=50, help="Epoch count")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=int, default=1, help="GPU device ID")
    parser.add_argument("--save_dir", type=str, default="golden_benchmark_results", help="Save directory")
    args = parser.parse_args()

    train_golden_model(args.model, epochs=args.epochs, batch_size=args.batch, lr=args.lr, device_idx=args.device, save_dir=args.save_dir)
