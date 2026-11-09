# Comprehensive Benchmark for Automated Median Nerve Ultrasound Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-zone)

This repository provides an open-source, reproducible, standardized benchmark suite evaluating **25+ deep learning segmentation architectures** across **four paradigm families** on clinical wrist-to-elbow median nerve ultrasound sweeps (100 subjects, 30,000 frames) for automated Carpal Tunnel Syndrome (CTS) quantification.

---

## 🏛️ Benchmark Architecture Zoo

### 1. Spatio-Temporal Video SOTA Models (3D Sequential Clip Volumes)
* **Modified VisTR** (Video Instance Segmentation Transformer with ResNet-18, ResNet-50, ResNet-101)
* **VideoSwinUNet** (3D Shifted-Window SwinUNETR with 3D self-supervised pre-training)
* **TimeSformer** (Divided Space-Time Transformer with Kinetics-400 factorized attention)
* **Spatio-Temporal Video-Mamba** (Vivim / 3D Selective Scan SS3D)

### 2. Vision Transformers & Hybrids (Part I)
* **SegFormer** (MiT-B0, MiT-B1, MiT-B2, MiT-B3)
* **TransUNet** (Hybrid ResNet-50 + ViT-12)
* **Swin-UNet** (Shifted-Window Swin-T)
* **Swin-UNETR** (Hierarchical Medical Swin)
* **UNETR** (Pure ViT-B 16x16)
* **nnFormer** (3D/2D Interlaced Medical Transformer)

### 3. State-Space Models (Mamba Suite)
* **VM-UNet** (Visual State Space / VSS Block)
* **Swin-UMamba** (Hybrid Swin-Mamba)
* **Swin-UMamba+** (Enhanced Hybrid Swin-VSS)
* **U-Mamba (Enc / Bot)** (State-Space CNN encoder-decoder)

### 4. Classical Biomedical Segmentation Suite
* **MNSeg-Net** (Proposed Lightweight Multi-Scale RSU + Half-UNet Decoder, **3.34M Params**)
* **U$^2$-Net** (Nested ReSidual U-Blocks)
* **BASNet** (Boundary-Aware Segmentation Network)
* **Attention U-Net**
* **U-Net++** (Nested Dense Skip Pathways)
* **ResUNet** (Residual Building Blocks)
* **Canonical U-Net** & **SegNet**

---

## 📊 Comprehensive Benchmark Results (3,000 Independent Clinical Test Frames)

Evaluated on held-out test cohort (**Patients 90–99**, complete 300 continuous frames per subject) with strict patient-level disjoint boundary isolation.

### Spatio-Temporal Video Benchmark
| Model | Backbone / Weights | Parameters | Test Dice (%) | Test IoU (%) | Precision (%) | Recall (%) | CSA Error ($\text{mm}^2$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Modified VisTR** | ImageNet ResNet-18 | **43.74 M** | **83.26 ± 10.77** | **72.44 ± 12.45** | 78.24 ± 13.62 | **90.71 ± 11.64** | **2.001 ± 1.521** |
| **VideoSwinUNet** | Official 3D Swin (`model_swinvit.pt`) | **62.19 M** | **75.73 ± 22.13** | **64.86 ± 22.48** | **80.78 ± 22.23** | 74.25 ± 24.57 | 2.402 ± 2.690 |
| **Modified VisTR (R50)** | ImageNet ResNet-50 | **56.75 M** | *In Progress* | *In Progress* | *In Progress* | *In Progress* | *In Progress* |

### Static 2D SOTA Architectural Benchmark
| Model | Paradigm | Parameters | Test Dice (%) | Test IoU (%) | Precision (%) | Recall (%) | CSA Error ($\text{mm}^2$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SegFormer-B2** | Vision Transformer | 27.35 M | **83.83** | **75.03** | 83.15 | **86.42** | **1.312** |
| **TransUNet** | Hybrid CNN-ViT | 25.21 M | 83.08 | 74.27 | 82.53 | 85.50 | 1.360 |
| **VM-UNet** | Mamba (VSS) | 4.88 M | 82.69 | 74.05 | **84.10** | 83.24 | 1.321 |
| **Swin-UMamba+** | Mamba-Swin | 32.84 M | 82.46 | 73.59 | 82.81 | 83.64 | **1.264** |
| **MNSeg-Net (Proposed)** | Lightweight Nested RSU | **3.34 M** | **82.14** | **73.55** | 80.65 | **85.19** | **1.320** |
| **nnFormer** | Medical Transformer | 34.99 M | 81.94 | 73.10 | 83.24 | 82.60 | 1.400 |
| **Swin-UMamba** | Mamba-Swin | 31.42 M | 81.84 | 73.12 | 83.41 | 82.32 | 1.458 |
| **Swin-UNet** | Swin-T | 32.63 M | 81.68 | 72.79 | 83.22 | 82.51 | 1.471 |
| **U$^2$-Net** | Multi-Scale Nested | 44.00 M | 80.65 | 72.20 | 79.42 | 84.12 | 1.520 |
| **BASNet** | Boundary Refinement | 87.00 M | 80.54 | 71.85 | 78.95 | 83.70 | 1.610 |

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/venkateshvaddadi/median-nerve-ultrasound-benchmark.git
cd median-nerve-ultrasound-benchmark
pip install -r requirements.txt
```

### Pretrained Weights
Pretrained backbones for all models (ImageNet-1K ResNets, MONAI 3D Swin, Kinetics-400 Factorized ViT, and Mamba) are automatically cached via torchvision and HuggingFace Hub upon training start.

### Training Video Models
```bash
# Modified VisTR (ResNet-18)
python training/train_video_benchmark.py --model vistr --device 0 --epochs 25 --batch_size 16

# Modified VisTR (ResNet-50)
python training/train_video_benchmark.py --model vistr_resnet50 --device 0 --epochs 25 --batch_size 16

# VideoSwinUNet (3D Swin)
python training/train_video_benchmark.py --model video_swin --device 1 --epochs 25 --batch_size 4
```

### Evaluation on 3,000 Independent Clinical Frames
```bash
python evaluation/eval_test_video_swin_3000.py 0
```

---

## 🏥 Clinical Cross-Sectional Area (CSA) Calibration
Single pixel area is calibrated to **$0.004328254\,\text{mm}^2$** based on clinical transducer depth settings. Cross-Sectional Area (CSA) is computed as:
$$\text{CSA} = \text{Pixel Area} \times \sum_{(x,y)} \mathbb{I}(\hat{Y}(x,y) = 1) \quad (\text{mm}^2)$$

---

## 📜 License & Citation
This project is licensed under the MIT License.
If you use this benchmark or the models in your research, please cite our corresponding work.
