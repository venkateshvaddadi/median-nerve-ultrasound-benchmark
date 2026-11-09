#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Baselines Package for Median Nerve Ultrasound Segmentation.
Provides unified factory access to classic semantic segmentation architectures:
- UNet
- SegNet
- UNet++ (NestedUNet)
- ResUNet
- Attention U-Net
- BASNet
- U2Net / U2NetP
- MNSeg-Net (U2NETP with Half-UNet style decoder and custom channels)
- ENet
- DeeplabV3+
- AnamNet
- LEDNet
"""

import torch
import torch.nn as nn

from .unet import UNet
from .segnet import SegNet
from .unetplusplus import NestedUNet
from .resunet import ResUnet
from .attention_unet import Attention_UNet_updated_version_1 as AttentionUNet
from .basnet import BASNet
from .u2net import U2NET, U2NETP
from .mnsegnet import U2NETP_along_with_Half_UNet_style_Decoder_with_custom_channels as MNSegNet
from .enet import ENet
from .deeplabv3plus import DeepLabV3plus
from .anamnet import AnamNet
from .lednet2 import LEDNet

__all__ = [
    'UNet',
    'SegNet',
    'NestedUNet',
    'ResUnet',
    'AttentionUNet',
    'BASNet',
    'U2NET',
    'U2NETP',
    'MNSegNet',
    'ENet',
    'DeepLabV3plus',
    'AnamNet',
    'LEDNet',
    'get_classical_model'
]

def get_classical_model(name: str, in_channels: int = 3, out_channels: int = 1):
    """
    Factory constructor returning instantiated classical baseline models.
    """
    name_clean = name.lower().replace("-", "").replace("_", "").strip()

    if name_clean == "unet":
        return UNet()
    elif name_clean == "segnet":
        return SegNet(num_classes=out_channels, n_init_features=in_channels)
    elif name_clean in ["unetplusplus", "nestedunet", "unet++"]:
        return NestedUNet(num_classes=out_channels, input_channels=in_channels)
    elif name_clean == "resunet":
        return ResUnet(channel=in_channels)
    elif name_clean in ["attentionunet", "attunet"]:
        return AttentionUNet(in_channels, out_channels)
    elif name_clean == "basnet":
        return BASNet(in_channels, out_channels)
    elif name_clean == "u2net":
        return U2NET(in_channels, out_channels)
    elif name_clean == "u2netp":
        return U2NETP(in_channels, out_channels)
    elif name_clean in ["mnsegnet", "mnseg"]:
        return MNSegNet(in_channels, out_channels, 128, 16)
    elif name_clean == "enet":
        return ENet(num_classes=out_channels)
    elif name_clean in ["deeplabv3plus", "deeplabv3+"]:
        return DeepLabV3plus(n_classes=out_channels)
    elif name_clean == "anamnet":
        return AnamNet(num_classes=out_channels)
    elif name_clean in ["lednet", "lednet2"]:
        return LEDNet(num_classes=out_channels)
    else:
        raise ValueError(f"Unknown classical model: '{name}'")
