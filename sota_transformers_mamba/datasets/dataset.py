#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Golden Synergy Ultrasound Dataset & Augmentation Suite for Median Nerve Segmentation.
Discovered empirically in the SegFormer ablation ladder:
1. Input Preprocessing: Single-Channel CLAHE (Local Contrast Normalization)
2. Augmentation Suite: Full Ultrasound Composite:
   - Horizontal Flip (p=0.5)
   - ShiftScaleRotate (shift=0.06, scale=0.08, rotate=10 deg, p=0.5)
   - ElasticTransform (alpha=1.0, sigma=30, alpha_affine=15, p=0.3)
   - GaussNoise (var=10-40, p=0.3)
   - RandomGamma (gamma=85-115, p=0.3)
   - CoarseDropout acoustic shadowing simulation (p=0.25)
3. Standard B-mode Pure Crop: [4:452, 11:331] -> shape (448, 320)
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

class GoldenUltrasoundDataset(Dataset):
    """
    Standard dataset implementation equipped with CLAHE and composite ultrasound augmentations.
    """
    def __init__(self, data_path, csv_for_patients, csv_for_images, mode="train"):
        self.data_path = data_path
        self.mode = mode
        self.patients_list = pd.read_csv(csv_for_patients)
        raw_data = pd.read_csv(csv_for_images)

        # In validation mode, strictly evaluate on the pristine, unaugmented frames (2,998 frames, Patients 80-89)
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
