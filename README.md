# Comprehensive Benchmark for Automated Median Nerve Ultrasound Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-zone)

This repository provides an open-source, reproducible, standardized benchmark suite evaluating **25+ deep learning segmentation architectures** across **three modular suites** on clinical wrist-to-elbow median nerve ultrasound sweeps (100 subjects, 30,000 frames) for automated Carpal Tunnel Syndrome (CTS) quantification.

---

## 🏛️ Modular Repository Organization

Each paradigm is self-contained with its own dedicated model architectures, datasets, loss objectives, and training scripts:

```text
median-nerve-ultrasound-benchmark/
├── classical_models/                 # Classical Biomedical Segmentation Suite
│   ├── datasets/                     # CTS Dataset loader (pure B-mode crop [4:452, 11:331])
│   ├── loss/                         # BCE, SSIM, IoU, and Contour loss functions
│   ├── models/                       # UNet, SegNet, ResUNet, Attention-UNet, UNet++, BASNet, U2Net, MNSeg-Net
│   └── train_classical.py            # Unified classical training pipeline
│
├── sota_transformers_mamba/          # SOTA Vision Transformers & Mamba Suite
│   ├── datasets/                     # Golden Synergy Dataset (Single-Channel CLAHE + Composite Ultrasound Augmentations)
│   ├── models/                       # SegFormer, TransUNet, VM-UNet, Swin-Mamba, Swin-Mamba+, Swin-UNet, Swin-UNETR, nnFormer
│   └── train_golden_synergy.py       # Golden Synergy Training Pipeline (Lovász Soft Hinge + Cosine Anneal + ModelEMA + TTA)
│
├── spatio_temporal_video/            # Spatio-Temporal Video Benchmark Suite
│   ├── datasets/                     # Sequential 3D Video Clip Dataloader
│   ├── models/                       # Modified VisTR (R18, R50, R101), VideoSwinUNet (3D Swin), TimeSformer, Vivim
│   └── train_video.py                # Spatio-Temporal Clip Training Pipeline
│
├── evaluation/                       # Independent 3,000-Frame Benchmark Evaluation Scripts
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 📊 Comprehensive Benchmark Results (3,000 Independent Clinical Test Frames)

Evaluated on held-out test cohort (**Patients 90–99**, complete 300 continuous frames per subject) with strict patient-level disjoint boundary isolation.

### 1. Spatio-Temporal Video Benchmark
| Model | Paradigm / Backbone | Parameters | Test Dice (%) | Test IoU (%) | Precision (%) | Recall (%) | CSA Error ($\text{mm}^2$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SAM 2 / MemSAM** 🥇 | **Streaming Memory Video ViT (ResNet-34)** | **10.22 M** | **91.61 ± 7.43** | **85.05 ± 8.05** | **89.03 ± 8.20** | **94.63 ± 8.06** | **0.782 ± 0.621** |
| **Modified VisTR (R101)** 🥈 | Spatio-Temporal DETR (ResNet-101) | **75.69 M** | **87.04 ± 8.56** | **77.79 ± 9.91** | **83.89 ± 10.90** | **91.40 ± 9.37** | **1.331 ± 1.174** |
| **Video-Mask2Former** 🥉 | 3D Mask Query Transformer (ResNet-50) | **11.43 M** | **85.54 ± 11.83** | **76.08 ± 13.40** | **87.85 ± 13.19** | 84.99 ± 14.04 | **1.446 ± 1.552** |
| **Modified VisTR (R50)** | Spatio-Temporal DETR (ResNet-50) | **56.75 M** | 84.86 ± 10.47 | 74.76 ± 11.95 | 81.55 ± 13.14 | 90.08 ± 11.64 | 1.670 ± 1.376 |
| **Modified VisTR (R18)** | Spatio-Temporal DETR (ResNet-18) | **43.74 M** | 83.26 ± 10.77 | 72.44 ± 12.45 | 78.24 ± 13.62 | 90.71 ± 11.64 | 2.001 ± 1.521 |
| **VideoSwinUNet** | 3D Shifted-Window ViT (`model_swinvit.pt`) | **62.19 M** | 82.86 ± 16.00 | 73.05 ± 17.38 | 82.64 ± 17.30 | 84.95 ± 17.73 | 1.701 ± 1.877 |

### 2. SOTA Vision Transformers & Mamba (Golden Synergy Pipeline)
| Model | Paradigm | Parameters | Test Dice (%) | Test IoU (%) | Precision (%) | Recall (%) | CSA Error ($\text{mm}^2$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SegFormer-B2** | Vision Transformer | 27.35 M | **83.83** | **75.03** | 83.15 | **86.42** | **1.312** |
| **TransUNet** | Hybrid CNN-ViT | 25.21 M | 83.08 | 74.27 | 82.53 | 85.50 | 1.360 |
| **VM-UNet** | Mamba (VSS) | 4.88 M | 82.69 | 74.05 | **84.10** | 83.24 | 1.321 |
| **Swin-UMamba+** | Mamba-Swin | 32.84 M | 82.46 | 73.59 | 82.81 | 83.64 | **1.264** |
| **nnFormer** | Medical Transformer | 34.99 M | 81.94 | 73.10 | 83.24 | 82.60 | 1.400 |
| **Swin-UMamba** | Mamba-Swin | 31.42 M | 81.84 | 73.12 | 83.41 | 82.32 | 1.458 |
| **Swin-UNet** | Swin-T | 32.63 M | 81.68 | 72.79 | 83.22 | 82.51 | 1.471 |

### 3. Classical Biomedical Segmentation Suite
| Model | Paradigm | Parameters | Test Dice (%) | Test IoU (%) | Precision (%) | Recall (%) | CSA Error ($\text{mm}^2$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MNSeg-Net (Proposed)** | Lightweight Nested RSU | **3.34 M** | **82.14** | **73.55** | 80.65 | **85.19** | **1.320** |
| **U$^2$-Net** | Multi-Scale Nested | 44.00 M | 80.65 | 72.20 | 79.42 | 84.12 | 1.520 |
| **BASNet** | Boundary Refinement | 87.00 M | 80.54 | 71.85 | 78.95 | 83.70 | 1.610 |
| **Attention U-Net** | Attention Gate CNN | 34.88 M | 79.82 | 70.91 | 78.20 | 82.45 | 1.680 |
| **U-Net++** | Nested Dense Pathways | 9.04 M | 79.50 | 70.40 | 77.90 | 82.10 | 1.710 |
| **ResUNet** | Residual CNN | 21.28 M | 79.20 | 70.10 | 77.50 | 81.80 | 1.740 |
| **U-Net** | Canonical Encoder-Decoder | 31.04 M | 78.90 | 69.80 | 76.80 | 81.50 | 1.790 |
| **SegNet** | VGG Pooling Indices | 29.44 M | 77.80 | 68.50 | 75.60 | 80.20 | 1.850 |

---

## 🚀 Execution Guide

### 1. Training SOTA Vision Transformers & Mamba (Golden Synergy Pipeline)
```bash
# SegFormer (B0 / B1 / B2)
python sota_transformers_mamba/train_golden_synergy.py --model segformer-b2 --device 0 --epochs 35 --batch_size 16

# TransUNet
python sota_transformers_mamba/train_golden_synergy.py --model transunet --device 0 --epochs 35 --batch_size 16

# VM-UNet (Visual Mamba)
python sota_transformers_mamba/train_golden_synergy.py --model vmunet --device 0 --epochs 35 --batch_size 16

# Swin-Mamba / Swin-Mamba+
python sota_transformers_mamba/train_golden_synergy.py --model swin-mamba --device 0 --epochs 35 --batch_size 16
python sota_transformers_mamba/train_golden_synergy.py --model swin-mamba+ --device 0 --epochs 35 --batch_size 16

# Swin-UNet / Swin-UNETR / nnFormer
python sota_transformers_mamba/train_golden_synergy.py --model swin-unet --device 0 --epochs 35 --batch_size 16
python sota_transformers_mamba/train_golden_synergy.py --model swin-unetr --device 0 --epochs 35 --batch_size 16
python sota_transformers_mamba/train_golden_synergy.py --model nnformer --device 0 --epochs 35 --batch_size 16
```

### 2. Training Classical Models
```bash
# U-Net, SegNet, ResUNet, Attention U-Net
python classical_models/train_classical.py --model unet --device 0 --epochs 25 --batch_size 16
python classical_models/train_classical.py --model segnet --device 0 --epochs 25 --batch_size 16
python classical_models/train_classical.py --model resunet --device 0 --epochs 25 --batch_size 16
python classical_models/train_classical.py --model attention-unet --device 0 --epochs 25 --batch_size 16

# UNet++, BASNet, U2-Net, MNSeg-Net
python classical_models/train_classical.py --model unet++ --device 0 --epochs 25 --batch_size 16
python classical_models/train_classical.py --model basnet --device 0 --epochs 25 --batch_size 16
python classical_models/train_classical.py --model u2net --device 0 --epochs 25 --batch_size 16
python classical_models/train_classical.py --model mnseg-net --device 0 --epochs 25 --batch_size 16
```

### 3. Training Spatio-Temporal Video Models
```bash
# Modified VisTR (ResNet-18)
python spatio_temporal_video/train_video.py --model vistr --device 0 --epochs 25 --batch_size 16

# Modified VisTR (ResNet-50)
python spatio_temporal_video/train_video.py --model vistr_resnet50 --device 0 --epochs 25 --batch_size 16

# VideoSwinUNet (3D Swin)
python spatio_temporal_video/train_video.py --model video_swin --device 1 --epochs 25 --batch_size 4
```

---

## 📜 Citation & References

If you find this benchmark and code useful in your medical imaging research, please cite our corresponding publications and the foundational model architectures:

```bibtex
@article{ravi2024sam2,
  title={{SAM 2}: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}

@article{wang2023memsam,
  title={{MemSAM}: Taming Segment Anything Model for Echocardiography Video Segmentation},
  author={Wang, Jiayuan and others},
  journal={arXiv preprint arXiv:2308.11822},
  year={2023}
}

@inproceedings{wang2021vistr,
  title={End-to-End Video Instance Segmentation with Transformers},
  author={Wang, Yuqing and others},
  booktitle={CVPR},
  year={2021}
}

@inproceedings{cheng2022video_mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Cheng, Bowen and others},
  booktitle={CVPR},
  year={2022}
}

@inproceedings{xie2021segformer,
  title={{SegFormer}: Simple and Efficient Design for Semantic Segmentation with Light-Weight Transformers},
  author={Xie, Enze and others},
  booktitle={NeurIPS},
  year={2021}
}
```
All method references are cataloged in [`references.bib`](references.bib).
