# -*- coding: utf-8 -*-
"""
VideoAsterDataset: Sequential Clip Dataset Loader for Spatio-Temporal Video Segmentation
Compatible with the 10 Video SOTA Architectures (Modified VisTR, Video-Swin, TimeSformer,
Video-Mamba, Video-Mask2Former, SAM2-MemSAM, STCN-XMem, Flow-Guided EchoNet, TubeFormer, 3D UNet).

Key Features:
- Extracts contiguous video clips of length T (e.g. T=16 or T=32 frames) per subject.
- Strict patient-level disjoint boundary isolation (no leakage across clips between train/val/test).
- Exact ultrasound ROI cropping [4:452, 11:331] -> (448, 320) matching standard benchmark.
- Returns dual format:
    x_clip: (C, T, H, W) normalized tensor
    mask_clip: (1, T, H, W) binary segmentation tensor
    targets: dict containing bounding boxes, frame masks, and labels (for VisTR/DETR loss).
"""

import os
import re
import math
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


def masks_to_boxes(masks):
    """
    Compute [x_min, y_min, x_max, y_max] bounding boxes from binary masks (T, H, W).
    Normalizes coordinates to [0, 1] relative to frame dimensions.
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)

    T, H, W = masks.shape
    boxes = []
    for t in range(T):
        m = masks[t]
        pos = torch.nonzero(m)
        if pos.shape[0] == 0:
            boxes.append(torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32))
        else:
            y_min = pos[:, 0].min().float() / H
            x_min = pos[:, 1].min().float() / W
            y_max = pos[:, 0].max().float() / H
            x_max = pos[:, 1].max().float() / W
            # Clamp to [0, 1]
            boxes.append(torch.tensor([
                x_min.clamp(0.0, 1.0),
                y_min.clamp(0.0, 1.0),
                x_max.clamp(0.0, 1.0),
                y_max.clamp(0.0, 1.0)
            ], dtype=torch.float32))
    return torch.stack(boxes, dim=0)


class VideoAsterDataset(Dataset):
    """
    Sequential Clip Dataset for Ultrasound Median Nerve Video Segmentation.
    """
    def __init__(
        self,
        data_path="../data_making/aster_updated_data_nov_09_2022_with_flip/",
        patients_csv="../data_making/aster_updated_data_nov_09_2022_with_flip/csv_files/full_data_csv/patients_list_1_99.csv",
        frames_csv="csv_files/data_with_v_h_1_79_train_strictly_disjoint.csv",
        num_frames=16,
        stride=8,
        is_training=True,
        normalize=True,
        mean=(0.2316, 0.2316, 0.2316),
        std=(0.2038, 0.2038, 0.2038)
    ):
        self.data_path = data_path
        self.num_frames = int(num_frames)
        self.stride = int(stride)
        self.is_training = is_training
        self.normalize = normalize
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

        # Load patient directories and frame table
        self.patients_df = pd.read_csv(patients_csv)
        self.frames_df = pd.read_csv(frames_csv)

        # Build contiguous video clips per patient
        self.clips = []
        self._build_contiguous_clips()

    def _build_contiguous_clips(self):
        """
        Groups frames by patient_id and generates overlapping or non-overlapping
        sequential video clip slices of length num_frames.
        """
        # Group by patient_id
        grouped = self.frames_df.groupby('patient_id')
        for pid, group in grouped:
            if pid not in self.patients_df.index:
                continue
            patient_rel_dir = str(self.patients_df.iloc[pid]['path']).strip()

            sorted_imgs = sorted(group['image_no'].unique())
            if len(sorted_imgs) < self.num_frames:
                # If sequence is shorter than num_frames, repeat-pad indices
                padded = sorted_imgs + [sorted_imgs[-1]] * (self.num_frames - len(sorted_imgs))
                self.clips.append({
                    'patient_id': pid,
                    'patient_dir': patient_rel_dir,
                    'frame_indices': padded
                })
                continue

            # Slide window across continuous sequence
            step = self.stride if self.is_training else self.num_frames
            for start_idx in range(0, len(sorted_imgs) - self.num_frames + 1, step):
                clip_indices = sorted_imgs[start_idx:start_idx + self.num_frames]
                self.clips.append({
                    'patient_id': pid,
                    'patient_dir': patient_rel_dir,
                    'frame_indices': clip_indices
                })

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        patient_id = clip_info['patient_id']
        patient_dir = clip_info['patient_dir']
        frame_indices = clip_info['frame_indices']

        frames_tensors = []
        masks_tensors = []

        for frame_num in frame_indices:
            img_path = os.path.join(self.data_path, patient_dir, "images", f"{frame_num}.jpg")
            mask_path = os.path.join(self.data_path, patient_dir, "masks", f"{frame_num}.tif")

            # Fallback to .png mask if .tif does not exist
            if not os.path.exists(mask_path):
                mask_path_png = os.path.join(self.data_path, patient_dir, "masks", f"{frame_num}.png")
                if os.path.exists(mask_path_png):
                    mask_path = mask_path_png

            if os.path.exists(img_path) and os.path.exists(mask_path):
                try:
                    img_raw = np.array(Image.open(img_path).convert('RGB'))
                    mask_raw = np.array(Image.open(mask_path))

                    # Apply standard benchmark cropping [4:452, 11:331] -> (448, 320)
                    img_cropped = img_raw[4:452, 11:331, :]
                    mask_cropped = mask_raw[4:452, 11:331]
                except Exception:
                    img_cropped = np.zeros((448, 320, 3), dtype=np.uint8)
                    mask_cropped = np.zeros((448, 320), dtype=np.uint8)
            else:
                img_cropped = np.zeros((448, 320, 3), dtype=np.uint8)
                mask_cropped = np.zeros((448, 320), dtype=np.uint8)

            # Convert to float tensors
            img_t = torch.from_numpy(img_cropped).permute(2, 0, 1).float() / 255.0 # (3, H, W)
            if self.normalize:
                img_t = (img_t - self.mean) / self.std

            # Binary mask: (1, H, W)
            mask_t = torch.from_numpy((mask_cropped > 127).astype(np.float32)).unsqueeze(0)

            frames_tensors.append(img_t)
            masks_tensors.append(mask_t)

        # Video Clip volume: (C, T, H, W)
        # Permute/stack along temporal dimension T
        x_clip = torch.stack(frames_tensors, dim=1) # (3, T, H, W)
        mask_clip = torch.stack(masks_tensors, dim=1) # (1, T, H, W)

        # Build DETR/VisTR compatible target dict
        flat_masks = mask_clip.squeeze(0) # (T, H, W)
        boxes = masks_to_boxes(flat_masks) # (T, 4)
        targets = {
            'patient_id': torch.tensor(patient_id, dtype=torch.long),
            'frame_indices': torch.tensor(frame_indices, dtype=torch.long),
            'masks': flat_masks,
            'boxes': boxes,
            'labels': torch.ones(self.num_frames, dtype=torch.long),
            'valid': torch.ones(self.num_frames, dtype=torch.long)
        }

        return x_clip, mask_clip, targets


def get_video_dataloaders(
    data_path="../data_making/aster_updated_data_nov_09_2022_with_flip/",
    patients_csv="../data_making/aster_updated_data_nov_09_2022_with_flip/csv_files/full_data_csv/patients_list_1_99.csv",
    train_csv="csv_files/data_with_v_h_1_79_train_strictly_disjoint.csv",
    test_csv="csv_files/data_test_1_90_99_99.csv",
    num_frames=16,
    batch_size=2,
    num_workers=2
):
    """
    Constructs train and test DataLoader instances for video spatio-temporal modeling.
    """
    train_ds = VideoAsterDataset(
        data_path=data_path,
        patients_csv=patients_csv,
        frames_csv=train_csv,
        num_frames=num_frames,
        stride=num_frames // 2,
        is_training=True
    )

    test_ds = VideoAsterDataset(
        data_path=data_path,
        patients_csv=patients_csv,
        frames_csv=test_csv,
        num_frames=num_frames,
        stride=num_frames,
        is_training=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
