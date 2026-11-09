#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 19:13:09 2023

@author: venkatesh
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np






#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:23:33 2023

@author: venkatesh
"""

#!/usr/env/bin python3.6

from typing import List

import torch
import numbers
import math
from torch import Tensor, einsum
from torch import nn
from scipy.ndimage import distance_transform_edt, morphological_gradient, distance_transform_cdt
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from torch.nn import functional as F






#!/usr/bin/env python3.6

from random import random
from pathlib import Path
from multiprocessing.pool import Pool

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union

import torch
import numpy as np
from tqdm import tqdm
from torch import einsum
from torch import Tensor
from functools import partial
from skimage.io import imsave
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt as distance
from scipy.spatial.distance import directed_hausdorff


# functions redefinitions
tqdm_ = partial(tqdm, ncols=175,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [' '{rate_fmt}{postfix}]')

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)

def flatten_(to_flat: Iterable[Iterable[A]]) -> List[A]:
    return [e for l in to_flat for e in l]
def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def uc_(fn: Callable) -> Callable:
    return partial(uncurry, fn)


def uncurry(fn: Callable, args: List[Any]) -> Any:
    return fn(*args)


def id_(x):
    return x


# fns
def soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bcwh->bc", a)[..., None]


def batch_soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bcwh->c", a)[..., None]


def soft_centroid(a: Tensor) -> Tensor:
    b, c, w, h = a.shape

    ws, hs = map_(lambda e: Tensor(e).to(a.device).type(torch.float32), np.mgrid[0:w, 0:h])
    assert ws.shape == hs.shape == (w, h)

    flotted = a.type(torch.float32)
    tot = einsum("bcwh->bc", a).type(torch.float32) + 1e-10

    cw = einsum("bcwh,wh->bc", flotted, ws) / tot
    ch = einsum("bcwh,wh->bc", flotted, hs) / tot

    res = torch.stack([cw, ch], dim=2)
    assert res.shape == (b, c, 2)

    return res


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


# # Metrics and shitz
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> float:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bcwh->bc")
dice_batch = partial(meta_dice, "bcwh->c")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a & b


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a | b


def haussdorf(preds: Tensor, target: Tensor) -> Tensor:
    assert preds.shape == target.shape
    assert one_hot(preds)
    assert one_hot(target)

    B, C, _, _ = preds.shape

    res = torch.zeros((B, C), dtype=torch.float32, device=preds.device)
    n_pred = preds.cpu().numpy()
    n_target = target.cpu().numpy()

    for b in range(B):
        if C == 2:
            res[b, :] = numpy_haussdorf(n_pred[b, 0], n_target[b, 0])
            continue

        for c in range(C):
            res[b, c] = numpy_haussdorf(n_pred[b, c], n_target[b, c])

    return res


def numpy_haussdorf(pred: np.ndarray, target: np.ndarray) -> float:
    assert len(pred.shape) == 2
    assert pred.shape == target.shape

    return max(directed_hausdorff(pred, target)[0], directed_hausdorff(target, pred)[0])


# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    
    
    
    assert sset(seg, list(range(C)))
    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res


# Misc utils
def save_images(segs: Tensor, names: Iterable[str], root: str, mode: str, iter: int) -> None:
    b, w, h = segs.shape  # Since we have the class numbers, we do not need a C axis

    for seg, name in zip(segs, names):
        save_path = Path(root, f"iter{iter:03d}", mode, name).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        imsave(str(save_path), seg.cpu().numpy())


def augment(*arrs: Union[np.ndarray, Image.Image]) -> List[Image.Image]:
    imgs: List[Image.Image] = map_(Image.fromarray, arrs) if isinstance(arrs[0], np.ndarray) else list(arrs)

    if random() > 0.5:
        imgs = map_(ImageOps.flip, imgs)
    if random() > 0.5:
        imgs = map_(ImageOps.mirror, imgs)
    if random() > 0.5:
        angle = random() * 90 - 45
        imgs = map_(lambda e: e.rotate(angle), imgs)
    return imgs


def augment_arr(*arrs_a: np.ndarray) -> List[np.ndarray]:
    arrs = list(arrs_a)  # manoucherie type check

    if random() > 0.5:
        arrs = map_(np.flip, arrs)
    if random() > 0.5:
        arrs = map_(np.fliplr, arrs)

    return arrs


def get_center(shape: Tuple, *arrs: np.ndarray) -> List[np.ndarray]:
    def g_center(arr):
        if arr.shape == shape:
            return arr

        dx = (arr.shape[0] - shape[0]) // 2
        dy = (arr.shape[1] - shape[1]) // 2

        if dx == 0 or dy == 0:
            return arr[:shape[0], :shape[1]]

        res = arr[dx:-dx, dy:-dy][:shape[0], :shape[1]]  # Deal with off-by-one errors
        assert res.shape == shape, (res.shape, shape, dx, dy)

        return res

    return [g_center(arr) for arr in arrs]









#%%



def soft_skeletonize(x, thresh_width=10):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    for i in range(thresh_width):
        min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1)*-1
        max_min_pool_x = torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1)
        contour = torch.nn.functional.relu(max_min_pool_x - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x


#%%%



import torch

# Instantiate contour_loss with required parameters (replace `idc` with actual indices)
loss_calculator = contour_loss(idc=[0, 1])

# Example tensors (replace these with your actual data)
probs = torch.randn(1, 6, 32, 32)  # Example predicted tensor
target = torch.randn(1, 6, 32, 32)  # Example target tensor

# Calculate contour loss
con_loss = loss_calculator(probs, target, None)  # Adjust the last argument as needed

# Print the computed contour loss
print("Contour Loss:", con_loss)


#%%

def compute_morphogradient(segmentation):
    res = np.zeros(segmentation.shape)
    print(segmentation.shape)
    for i in range(segmentation.shape[0]):
        posmask = segmentation[i]
        negmask = ~posmask
        res[i] = morphological_gradient(posmask[0].astype(np.float32), size=(3,3))
    return res

    
class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type(torch.float32)

        loss = - einsum("bcwh,bcwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss


class SurfaceLoss():
    '''
    Boundary loss implementation 
    Inputs:
    @probs: probability maps provded from the output of the network 
    @dc: distance maps computed when the dataset class is initialized
    outputs:
    @loss: boundary loss
    @description: 
    the loss finetunes the probability maps by the groundtruth distance map representations.
    '''
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss



class HDDTBinaryLoss():
    '''
    Hausdorf loss implementation for binary segmentation 
    '''
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, net_output: Tensor, target: Tensor, _: Tensor) -> Tensor:
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        
        pc = net_output[:, self.idc, ...].type(torch.float32)
        gt = target[:, self.idc, ...].type(torch.float32)
        with torch.no_grad():
            pc_dist = compute_edts_forhdloss(pc.detach().cpu().numpy()>0.5)
            gt_dist = compute_edts_forhdloss(gt.detach().cpu().cpu().numpy()>0.5)
        # print('pc_dist.shape: ', pc_dist.shape)
        
        pred_error = (gt - pc)**2
        dist = pc_dist**2 + gt_dist**2 # \alpha=2 in eq(8)

        dist = torch.from_numpy(dist)
        if dist.device != pred_error.device:
            dist = dist.to(pred_error.device).type(torch.float32)

        multipled = torch.einsum("bxyz,bxyz->bxyz", 
                                 pred_error.reshape(-1,1,pred_error.shape[1], pred_error.shape[2]), 
                                 dist.reshape(-1,1,dist.shape[1], dist.shape[2]))
        hd_loss = multipled.mean()

        return hd_loss


class soft_cldice_loss():
    '''
    inputs shape  (batch, channel, height, width).
    calculate clDice loss
    Because pred and target at moment of loss calculation will be a torch tensors
    it is preferable to calculate target_skeleton on the step of batch forming,
    when it will be in numpy array format by means of opencv
    '''
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")
    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)
        b, _, w, h = pc.shape
        cl_pred = soft_skeletonize(pc)
        target_skeleton = soft_skeletonize(tc)
        big_pen: Tensor = (cl_pred - target_skeleton) ** 2
        contour_loss = big_pen / (w * h)
    
        return contour_loss.mean()
    

import numpy as np
import cv2
import torch

def opencv_skelitonize(img):
    skel = np.zeros(img.shape, np.uint8)
    img = img.astype(np.uint8)
    size = np.size(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    return skel









def norm_intersection(center_line, vessel):
    '''
    inputs shape  (batch, channel, height, width)
    intersection formalized by first areas
    x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
    '''
    smooth = 1.
    s = center_line.shape[:2]
    clf = center_line.view(s[0], s[1], -1)
    v = vessel.shape[:2]
    vf = vessel.view(v[0],v[1], -1)
    intersection = (clf * vf).sum(-1)
    return (intersection + smooth) / (clf.sum(-1) + smooth)


def compute_edts_forhdloss(segmentation):
    res = np.zeros(segmentation.shape)
    for i in range(segmentation.shape[0]):
        posmask = segmentation[i]
        negmask = ~posmask
        res[i] = distance_transform_edt(posmask) + distance_transform_edt(negmask)
    return res



####################################################################################

class NaivePenalty():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.C = len(self.idc)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)
        assert probs.shape == target.shape

        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        k = bounds.shape[2]  # scalar or vector
        value: Tensor = self.__fn__(probs[:, self.idc, ...])
        lower_b = bounds[:, self.idc, :, 0]
        upper_b = bounds[:, self.idc, :, 1]

        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape

        too_big: Tensor = (value > upper_b).type(torch.float32)
        too_small: Tensor = (value < lower_b).type(torch.float32)

        big_pen: Tensor = (value - upper_b) ** 2
        small_pen: Tensor = (value - lower_b) ** 2

        res = too_big * big_pen + too_small * small_pen

        loss: Tensor = res / (w * h)

        return loss.mean()









def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)
#%%




def contour(x):
    '''
    Differenciable aproximation of contour extraction
    
    '''   
    min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1)*-1
    max_min_pool_x = torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1)
    contour = torch.nn.functional.relu(max_min_pool_x - min_pool_x)
    return contour

class contour_loss():
    def __init__(self, **kwargs):
        print("contour loss was calling")

    def forward(self, probs, target):
        pc = probs.type(torch.float32)
        tc = target.type(torch.float32)

        b, _, w, h = pc.shape
        cl_pred = contour(pc).sum(axis=(2,3))
        target_contour = contour(tc).sum(axis=(2,3))
        big_pen: Tensor = (cl_pred - target_contour) ** 2
        contour_loss = big_pen / (w * h)
    
        return contour_loss.mean(axis=0)



#%%
import torch
import torch.nn as nn

def contour(x):
    min_pool_x = torch.nn.functional.max_pool2d(x * -1, (3, 3), 1, 1) * -1
    max_min_pool_x = torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1)
    contour_output = torch.nn.functional.relu(max_min_pool_x - min_pool_x)
    return contour_output

class ContourLoss(nn.Module):
    def __init__(self, **kwargs):
        super(ContourLoss, self).__init__()
        print("Contour loss was initialized")

    def forward(self, probs, target):
        pc = probs.type(torch.float32)
        tc = target.type(torch.float32)

        b, _, w, h = pc.shape
        cl_pred = contour(pc).sum(dim=(2, 3))
        target_contour = contour(tc).sum(dim=(2, 3))
        big_pen = (cl_pred - target_contour) ** 2
        contour_loss = big_pen / (w * h)

        return contour_loss.mean()


#%%

probs = torch.randn(4, 1, 456, 320)  # Example predicted tensor
target = torch.randn(4, 1, 456,320)  # Example target tensor

# Instantiate the ContourLoss
loss_calculator = ContourLoss()

loss = loss_calculator(probs, target)
print(loss)




