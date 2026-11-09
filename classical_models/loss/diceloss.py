#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:07:34 2022

@author: venkatesh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
#%%





class log_cosh_dice_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(log_cosh_dice_loss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        # x = self.dice_loss(y_true, y_pred)
        # return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

        dice_loss=1-dice;
        cosh_dice_loss=(torch.exp(dice_loss)+torch.exp(-dice_loss))/2
        log_cosh_dice_loss_value=torch.log(cosh_dice_loss)
        return log_cosh_dice_loss_value

#%%
import torch
from scipy.spatial.distance import directed_hausdorff

def hausdorff_distance(mask1, mask2):
    """
    Calculate Hausdorff Distance between two binary masks.
    
    Args:
    - mask1: Binary mask 1 (PyTorch tensor)
    - mask2: Binary mask 2 (PyTorch tensor)
    
    Returns:
    - hausdorff_dist: Hausdorff Distance
    """
    # Ensure the masks have the same shape
    assert mask1.shape == mask2.shape, "Both masks should have the same shape"
    
    # Convert PyTorch tensors to NumPy arrays
    mask1_np = mask1
    mask2_np = mask2
    
    # Calculate the directed Hausdorff distance
    dist_1_to_2 = directed_hausdorff(mask1_np, mask2_np)[0]
    dist_2_to_1 = directed_hausdorff(mask2_np, mask1_np)[0]
    
    # Hausdorff Distance is the maximum of the two distances
    hausdorff_dist = max(dist_1_to_2, dist_2_to_1)
#%%
    return hausdorff_dist
if __name__=="__main__":
    a=torch.rand(1,3,255,255).cuda()
    b=torch.rand(1,3,255,255).cuda()
    print(a.shape)
    loss=DiceLoss()
    print(loss(a,b).cpu().item(),)
    print(loss)

    loss=log_cosh_dice_loss()
    loss_value=loss(a,b)
    print(loss_value,loss_value.cpu().item())
