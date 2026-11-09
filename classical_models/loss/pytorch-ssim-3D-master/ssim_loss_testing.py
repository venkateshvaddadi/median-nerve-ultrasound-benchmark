#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 23:07:48 2023

@author: venkatesh
"""

import pytorch_ssim
import torch
from torch.autograd import Variable
#%%
img1 = Variable(torch.rand(6, 1, 64, 64, 64))
img2 = Variable(torch.rand(6, 1, 64, 64, 64))

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()

print(pytorch_ssim.ssim3D(img1, img2))

ssim_loss = pytorch_ssim.SSIM3D(window_size = 11)

print(ssim_loss(img1, img2))

