#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOTA Vision Transformers & Mamba Models Package for Median Nerve Ultrasound Segmentation.
Supported models:
- SegFormer (B0, B1, B2)
- TransUNet (ViT + CNN Decoder)
- VM-UNet (Visual Mamba State Space)
- Swin-Mamba / Swin-UMamba
- Swin-Mamba+ / Swin-UMamba+ (Dagger)
- Swin-UNet
- Swin-UNETR
- nnFormer
- U-Mamba (Encoder / Bottleneck)
- nnUNet
"""

import torch
import torch.nn as nn

from .transunet import TransUNet, PretrainedTransUNet
from .vmunet import VMUNet, PretrainedVMUNet
from .swin_unetr import SwinUNETR2D, SwinUNet2D, PretrainedSwinUNet
from .segformer import SegFormerMedianNerve
from .modern_sota_models import (
    UMamba,
    SwinUMamba,
    nnFormer2D,
    nnUNet2D,
    get_modern_model
)

__all__ = [
    'TransUNet',
    'PretrainedTransUNet',
    'VMUNet',
    'PretrainedVMUNet',
    'SwinUNETR2D',
    'SwinUNet2D',
    'PretrainedSwinUNet',
    'SegFormerMedianNerve',
    'UMamba',
    'SwinUMamba',
    'nnFormer2D',
    'nnUNet2D',
    'get_sota_model'
]

def get_sota_model(model_name: str, in_channels: int = 3, out_channels: int = 1, pretrained: bool = True):
    """
    Factory loader for Vision Transformers and State-Space Mamba models.
    """
    m = model_name.lower().replace("_", "").replace("-", "").strip()

    # 1. SegFormer variants
    if "segformer" in m:
        if "b2" in m:
            return SegFormerMedianNerve("nvidia/mit-b2", in_channels=in_channels, num_classes=out_channels)
        elif "b1" in m:
            return SegFormerMedianNerve("nvidia/mit-b1", in_channels=in_channels, num_classes=out_channels)
        else:
            return SegFormerMedianNerve("nvidia/mit-b0", in_channels=in_channels, num_classes=out_channels)

    # 2. TransUNet
    elif m == "transunet":
        return PretrainedTransUNet(in_channels=in_channels, out_channels=out_channels, pretrained=pretrained)

    # 3. VM-UNet (Visual Mamba UNet)
    elif m == "vmunet":
        return PretrainedVMUNet(in_channels=in_channels, out_channels=out_channels, pretrained=pretrained)

    # 4. Swin-UNet
    elif m == "swinunet":
        return PretrainedSwinUNet(in_channels=in_channels, out_channels=out_channels, pretrained=pretrained)

    # 5. Swin-UNETR
    elif m == "swinunetr":
        return SwinUNETR2D(in_channels=in_channels, out_channels=out_channels)

    # 6. Swin-Mamba+ / Swin-UMamba+ (Dagger)
    elif m in ["swinmamba+", "swinmambaplus", "swinumamba+", "swinumambaplus", "swinumambadagger"]:
        return SwinUMamba(in_channels=in_channels, out_channels=out_channels, is_plus=True, pretrained=pretrained)

    # 7. Swin-Mamba / Swin-UMamba
    elif m in ["swinmamba", "swinumamba"]:
        return SwinUMamba(in_channels=in_channels, out_channels=out_channels, is_plus=False, pretrained=pretrained)

    # 8. nnFormer
    elif m == "nnformer":
        return nnFormer2D(in_channels=in_channels, out_channels=out_channels, pretrained=pretrained)

    # 9. U-Mamba (Encoder / Bottleneck)
    elif m in ["umamba", "umambaenc"]:
        return UMamba(in_channels=in_channels, out_channels=out_channels, variant="enc", pretrained=pretrained)
    elif m == "umambabot":
        return UMamba(in_channels=in_channels, out_channels=out_channels, variant="bot", pretrained=pretrained)

    # 10. nnUNet
    elif m == "nnunet":
        return nnUNet2D(in_channels=in_channels, out_channels=out_channels)

    else:
        # Fallback to modern_sota_models factory
        return get_modern_model(model_name, in_channels=in_channels, num_classes=out_channels)
