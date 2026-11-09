#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SegFormer Architecture for Median Nerve Ultrasound Segmentation.
Wraps HuggingFace SegformerForSemanticSegmentation with bilinear interpolation.
Supports variants:
- nvidia/mit-b0 (SegFormer-B0)
- nvidia/mit-b1 (SegFormer-B1)
- nvidia/mit-b2 (SegFormer-B2)
- nvidia/mit-b3 (SegFormer-B3)
- nvidia/mit-b4 (SegFormer-B4)
- nvidia/mit-b5 (SegFormer-B5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SegFormerMedianNerve(nn.Module):
    def __init__(self, model_variant="nvidia/mit-b0", in_channels=3, num_classes=1):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
        
        # If in_channels != 3, we can configure or use 3-channel adapter
        self.config = SegformerConfig.from_pretrained(model_variant, num_labels=num_classes)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_variant, config=self.config, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        logits = self.model(x).logits  # (B, num_classes, H/4, W/4)
        upsampled = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return upsampled, torch.sigmoid(upsampled)
