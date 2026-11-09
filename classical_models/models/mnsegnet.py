#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:01:31 2023

@author: venkatesh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):
    #print(tar.shape[2:])
    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


##### U^2-Net ####
class U2NET(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NET,self).__init__()

        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        #print('hx1.shape',hx1.shape)

        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        #print('hx2.shape',hx2.shape)

        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        #print('hx3.shape',hx3.shape)

        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        #print('hx4.shape',hx4.shape)

        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        #print('hx5.shape',hx5.shape)

        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        #print('hx6.shape',hx6.shape)

        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        #return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
        return F.sigmoid(d0)
### U^2-Net small ###
class U2NETP_along_with_Half_UNet(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NETP_along_with_Half_UNet,self).__init__()

        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(64,16,64)

        # decoder
        self.stage5d = RSU4F(128,16,64)
        self.stage4d = RSU4(128,16,64)
        self.stage3d = RSU5(128,16,64)
        self.stage2d = RSU6(128,16,64)
        self.stage1d = RSU7(128,16,64)


        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(64,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)



        #   motivated from Half UNet Decoder.......
        self.up_hx2_to_x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_hx3_to_x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_hx4_to_x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_hx5_to_x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up_hx6_to_x = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)



        self.final_stage_of_half_UNet = RSU7(64,16,64)
        self.outconv_Half_of_UNet = nn.Conv2d(64,out_ch,1)
        self.decision=nn.Sigmoid()







    def forward(self,x):

        hx = x
        #print('input:','source image',hx.shape)

        #stage 1
        hx1 = self.stage1(hx)
        #print('stage1:','RSU7','hx1.shape',hx1.shape)
        hx = self.pool12(hx1)
        #print('After maxpooling:',hx.shape)

        #stage 2
        hx2 = self.stage2(hx)
        #print('stage2:','RSU6','hx2.shape',hx2.shape)

        hx = self.pool23(hx2)
        #print('maxpooling',hx.shape)

        #stage 3
        hx3 = self.stage3(hx)
        #print('stage3:','RSU5','hx3.shape',hx3.shape)

        hx = self.pool34(hx3)
        #print('maxpooling',hx.shape)

        #stage 4
        hx4 = self.stage4(hx)
        #print('stage4:','RSU4','hx4.shape',hx4.shape)

        hx = self.pool45(hx4)
        #print('maxpooling',hx.shape)

        #stage 5
        hx5 = self.stage5(hx)
        #print('stage5:','RSU4F','hx5.shape:',hx5.shape)

        hx = self.pool56(hx5)
        #print('maxpooling',hx.shape)

        #stage 6
        hx6 = self.stage6(hx)
        #print('stage6:','RSU4F','hx6.shape',hx6.shape)
















        #print("#"*50)
        hx6up = _upsample_like(hx6,hx5)
        #print('_upsample_like:','hx6up',hx6up.shape)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        #print('stage5d:','RSU4F','hx5d.shape',hx5d.shape)

        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        #print('stage4d:','RSU4','hx4d.shape',hx4d.shape)

        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        #print('stage3d:','RSU5','hx3d.shape',hx3d.shape)

        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        #print('stage2d:','RSU6','hx2d.shape',hx2d.shape)

        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))
        #print('stage1d:','RSU7','hx1d.shape',hx1d.shape)


        #side output
        d1 = self.side1(hx1d)
        #print('d1.shape',d1.shape)
        d2 = self.side2(hx2d)
        #print('d2.shape',d2.shape)

        d2 = _upsample_like(d2,d1)
        #print('d2.shape',d2.shape)

        d3 = self.side3(hx3d)
        #print('d3.shape',d3.shape)

        d3 = _upsample_like(d3,d1)
        #print('d3.shape',d3.shape)

        d4 = self.side4(hx4d)
        #print('d4.shape',d4.shape)

        d4 = _upsample_like(d4,d1)
        #print('d4.shape',d4.shape)

        d5 = self.side5(hx5d)
        #print('d5.shape',d5.shape)

        d5 = _upsample_like(d5,d1)
        #print('d5.shape',d5.shape)

        d6 = self.side6(hx6)
        #print('d6.shape',d6.shape)
        d6 = _upsample_like(d6,d1)
        #print('d6.shape',d6.shape)

        #print(torch.cat((d1,d2,d3,d4,d5,d6),1).shape)
        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))
        #print('d0.shape:',d0.shape)
        

        # Half UNet decoder adding
        #print('Half UNet Decooder implementation')
        hx2_upsample_to_orig_shape=self.up_hx2_to_x(hx2)
        #print(hx2_upsample_to_orig_shape.shape)

        hx3_upsample_to_orig_shape=self.up_hx3_to_x(hx3)
        #print(hx3_upsample_to_orig_shape.shape)

        hx4_upsample_to_orig_shape=self.up_hx4_to_x(hx4)
        #print(hx4_upsample_to_orig_shape.shape)

        hx5_upsample_to_orig_shape=self.up_hx5_to_x(hx5)
        #print(hx5_upsample_to_orig_shape.shape)

        hx6_upsample_to_orig_shape=self.up_hx6_to_x(hx6)
        #print(hx6_upsample_to_orig_shape.shape)

        fusion_of_upsampled_x=hx1+hx2_upsample_to_orig_shape+hx3_upsample_to_orig_shape+hx4_upsample_to_orig_shape+hx5_upsample_to_orig_shape+hx6_upsample_to_orig_shape

        temp_output_1=self.final_stage_of_half_UNet(fusion_of_upsampled_x)
        temp_output_2=self.outconv_Half_of_UNet(temp_output_1)
        output_of_Half_UNet=self.decision(temp_output_2)

        #print('output_of_Half_UNet.shape:',output_of_Half_UNet.shape)
        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6),output_of_Half_UNet
#%%
class U2NETP_along_with_Half_UNet_style_Decoder_with_custom_channels(nn.Module):

    def __init__(self,in_ch=3,out_ch=1,no_channels_dealing_in_and_out=64,no_channels_dealing_in_the_middle=16):
        super(U2NETP_along_with_Half_UNet_style_Decoder_with_custom_channels,self).__init__()

        self.stage1 = RSU7(in_ch,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)

        # decoder
        self.stage5d = RSU4F(2*no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.stage4d = RSU4(2*no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.stage3d = RSU5(2*no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.stage2d = RSU6(2*no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.stage1d = RSU7(2*no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)


        self.side1 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)



        #   motivated from Half UNet Decoder.......
        self.up_hx2_to_x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_hx3_to_x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_hx4_to_x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_hx5_to_x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up_hx6_to_x = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)



        self.final_stage_of_half_UNet = RSU7(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.outconv_Half_of_UNet = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,1)
        self.decision=nn.Sigmoid()







    def forward(self,x):

        hx = x
        #print('input:','source image',hx.shape)

        #stage 1
        hx1 = self.stage1(hx)
        #print('stage1:','RSU7','hx1.shape',hx1.shape)
        hx = self.pool12(hx1)
        #print('After maxpooling:',hx.shape)

        #stage 2
        hx2 = self.stage2(hx)
        #print('stage2:','RSU6','hx2.shape',hx2.shape)

        hx = self.pool23(hx2)
        #print('maxpooling',hx.shape)

        #stage 3
        hx3 = self.stage3(hx)
        #print('stage3:','RSU5','hx3.shape',hx3.shape)

        hx = self.pool34(hx3)
        #print('maxpooling',hx.shape)

        #stage 4
        hx4 = self.stage4(hx)
        #print('stage4:','RSU4','hx4.shape',hx4.shape)

        hx = self.pool45(hx4)
        #print('maxpooling',hx.shape)

        #stage 5
        hx5 = self.stage5(hx)
        #print('stage5:','RSU4F','hx5.shape:',hx5.shape)

        hx = self.pool56(hx5)
        #print('maxpooling',hx.shape)

        #stage 6
        hx6 = self.stage6(hx)
        #print('stage6:','RSU4F','hx6.shape',hx6.shape)
















        #print("#"*50)
        hx6up = _upsample_like(hx6,hx5)
        #print('_upsample_like:','hx6up',hx6up.shape)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        #print('stage5d:','RSU4F','hx5d.shape',hx5d.shape)

        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        #print('stage4d:','RSU4','hx4d.shape',hx4d.shape)

        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        #print('stage3d:','RSU5','hx3d.shape',hx3d.shape)

        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        #print('stage2d:','RSU6','hx2d.shape',hx2d.shape)

        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))
        #print('stage1d:','RSU7','hx1d.shape',hx1d.shape)


        #side output
        d1 = self.side1(hx1d)
        #print('d1.shape',d1.shape)
        d2 = self.side2(hx2d)
        #print('d2.shape',d2.shape)

        d2 = _upsample_like(d2,d1)
        #print('d2.shape',d2.shape)

        d3 = self.side3(hx3d)
        #print('d3.shape',d3.shape)

        d3 = _upsample_like(d3,d1)
        #print('d3.shape',d3.shape)

        d4 = self.side4(hx4d)
        #print('d4.shape',d4.shape)

        d4 = _upsample_like(d4,d1)
        #print('d4.shape',d4.shape)

        d5 = self.side5(hx5d)
        #print('d5.shape',d5.shape)

        d5 = _upsample_like(d5,d1)
        #print('d5.shape',d5.shape)

        d6 = self.side6(hx6)
        #print('d6.shape',d6.shape)
        d6 = _upsample_like(d6,d1)
        #print('d6.shape',d6.shape)

        #print(torch.cat((d1,d2,d3,d4,d5,d6),1).shape)
        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))
        #print('d0.shape:',d0.shape)
        

        # Half UNet decoder adding
        #print('Half UNet Decooder implementation')
        hx2_upsample_to_orig_shape=self.up_hx2_to_x(hx2)
        #print(hx2_upsample_to_orig_shape.shape)

        hx3_upsample_to_orig_shape=self.up_hx3_to_x(hx3)
        #print(hx3_upsample_to_orig_shape.shape)

        hx4_upsample_to_orig_shape=self.up_hx4_to_x(hx4)
        #print(hx4_upsample_to_orig_shape.shape)

        hx5_upsample_to_orig_shape=self.up_hx5_to_x(hx5)
        #print(hx5_upsample_to_orig_shape.shape)

        hx6_upsample_to_orig_shape=self.up_hx6_to_x(hx6)
        #print(hx6_upsample_to_orig_shape.shape)

        fusion_of_upsampled_x=hx1+hx2_upsample_to_orig_shape+hx3_upsample_to_orig_shape+hx4_upsample_to_orig_shape+hx5_upsample_to_orig_shape+hx6_upsample_to_orig_shape

        temp_output_1=self.final_stage_of_half_UNet(fusion_of_upsampled_x)
        temp_output_2=self.outconv_Half_of_UNet(temp_output_1)
        output_of_Half_UNet=self.decision(temp_output_2)

        #print('output_of_Half_UNet.shape:',output_of_Half_UNet.shape)
        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6),output_of_Half_UNet
#%%

#        'This is the proposed model'

class U2NETP_along_with_Half_UNet_style_Decoder_with_custom_channels_updated_architecture(nn.Module):

    def __init__(self,in_ch=3,out_ch=1,no_channels_dealing_in_and_out=64,no_channels_dealing_in_the_middle=16):
        super(U2NETP_along_with_Half_UNet_style_Decoder_with_custom_channels_updated_architecture,self).__init__()

        self.stage1 = RSU7(in_ch,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)

        # decoder
        self.stage5d = RSU4F(2*no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.stage4d = RSU4(2*no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.stage3d = RSU5(2*no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.stage2d = RSU6(2*no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.stage1d = RSU7(2*no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)


        self.side1 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(7*out_ch,out_ch,1)



        #   motivated from Half UNet Decoder.......
        self.up_hx2_to_x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_hx3_to_x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_hx4_to_x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_hx5_to_x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up_hx6_to_x = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)



        self.final_stage_of_half_UNet = RSU7(no_channels_dealing_in_and_out,no_channels_dealing_in_the_middle,no_channels_dealing_in_and_out)
        self.outconv_Half_of_UNet = nn.Conv2d(no_channels_dealing_in_and_out,out_ch,1)
        self.decision=nn.Sigmoid()







    def forward(self,x):
        # print('This is the proposed model')
        hx = x
        # print('input:','source image',hx.shape)

        #stage 1
        hx1 = self.stage1(hx)
        # print('stage1:','RSU7','hx1.shape',hx1.shape)
        hx = self.pool12(hx1)
        # print('After maxpooling:',hx.shape)

        #stage 2
        hx2 = self.stage2(hx)
        # print('stage2:','RSU6','hx2.shape',hx2.shape)

        hx = self.pool23(hx2)
        # print('maxpooling',hx.shape)

        #stage 3
        hx3 = self.stage3(hx)
        # print('stage3:','RSU5','hx3.shape',hx3.shape)

        hx = self.pool34(hx3)
        # print('maxpooling',hx.shape)

        #stage 4
        hx4 = self.stage4(hx)
        # print('stage4:','RSU4','hx4.shape',hx4.shape)

        hx = self.pool45(hx4)
        # print('maxpooling',hx.shape)

        #stage 5
        hx5 = self.stage5(hx)
        # print('stage5:','RSU4F','hx5.shape:',hx5.shape)

        hx = self.pool56(hx5)
        # print('maxpooling',hx.shape)

        #stage 6
        hx6 = self.stage6(hx)
        # print('stage6:','RSU4F','hx6.shape',hx6.shape)
















        #print("#"*50)
        hx6up = _upsample_like(hx6,hx5)
        # print('_upsample_like:','hx6up',hx6up.shape)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        # print('stage5d:','RSU4F','hx5d.shape',hx5d.shape)

        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        # print('stage4d:','RSU4','hx4d.shape',hx4d.shape)

        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        # print('stage3d:','RSU5','hx3d.shape',hx3d.shape)

        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        # print('stage2d:','RSU6','hx2d.shape',hx2d.shape)

        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))
        # print('stage1d:','RSU7','hx1d.shape',hx1d.shape)


        #side output
        d1 = self.side1(hx1d)
        # print('d1.shape',d1.shape)
        d2 = self.side2(hx2d)
        #print('d2.shape',d2.shape)

        d2 = _upsample_like(d2,d1)
        #print('d2.shape',d2.shape)

        d3 = self.side3(hx3d)
        #print('d3.shape',d3.shape)

        d3 = _upsample_like(d3,d1)
        #print('d3.shape',d3.shape)

        d4 = self.side4(hx4d)
        #print('d4.shape',d4.shape)

        d4 = _upsample_like(d4,d1)
        #print('d4.shape',d4.shape)

        d5 = self.side5(hx5d)
        #print('d5.shape',d5.shape)

        d5 = _upsample_like(d5,d1)
        #print('d5.shape',d5.shape)

        d6 = self.side6(hx6)
        #print('d6.shape',d6.shape)
        d6 = _upsample_like(d6,d1)
        #print('d6.shape',d6.shape)

        

        # Half UNet decoder adding
        #print('Half UNet Decooder implementation')
        hx2_upsample_to_orig_shape=self.up_hx2_to_x(hx2)
        #print(hx2_upsample_to_orig_shape.shape)

        hx3_upsample_to_orig_shape=self.up_hx3_to_x(hx3)
        #print(hx3_upsample_to_orig_shape.shape)

        hx4_upsample_to_orig_shape=self.up_hx4_to_x(hx4)
        #print(hx4_upsample_to_orig_shape.shape)

        hx5_upsample_to_orig_shape=self.up_hx5_to_x(hx5)
        #print(hx5_upsample_to_orig_shape.shape)

        hx6_upsample_to_orig_shape=self.up_hx6_to_x(hx6)
        #print(hx6_upsample_to_orig_shape.shape)

        fusion_of_upsampled_x=hx1+hx2_upsample_to_orig_shape+hx3_upsample_to_orig_shape+hx4_upsample_to_orig_shape+hx5_upsample_to_orig_shape+hx6_upsample_to_orig_shape
        # print('fusion_of_upsampled_x',fusion_of_upsampled_x.shape)
        temp_output_1=self.final_stage_of_half_UNet(fusion_of_upsampled_x)
        d7=self.outconv_Half_of_UNet(temp_output_1)

        output_of_Half_UNet=self.decision(d7)

        
        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6,d7),1))
        concated_feature=torch.cat((d1,d2,d3,d4,d5,d6,d7),1)
        #print('output_of_Half_UNet.shape:',output_of_Half_UNet.shape)
        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6),output_of_Half_UNet
        # return d0,d1,d2,d3,d4,d5,d6,d7,concated_feature,hx1d
        #return d0,d1,d2,d3,d4,d5,d6,d7,temp_output_1,hx1,hx2_upsample_to_orig_shape,hx3_upsample_to_orig_shape,hx4_upsample_to_orig_shape,hx5_upsample_to_orig_shape,hx6_upsample_to_orig_shape,fusion_of_upsampled_x

        # return F.sigmoid(d0)
#%%
##### U^2-Net ####
class U2NET_along_with_Half_UNet_updated_architecture(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NET_along_with_Half_UNet_updated_architecture,self).__init__()

        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(7*out_ch,out_ch,1)


        #   motivated from Half UNet Decoder.......
        # for converting all stage into 64 channel
        self.hx2_to_64 = RSU4(128,16,64)
        self.hx3_to_64 = RSU4(256,16,64)
        self.hx4_to_64 = RSU4(512,16,64)
        self.hx5_to_64 = RSU4(512,16,64)
        self.hx6_to_64 = RSU4(512,16,64)


        self.up_hx2_to_x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_hx3_to_x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_hx4_to_x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_hx5_to_x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up_hx6_to_x = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)


        self.final_stage_of_half_UNet = RSU7(64,16,64)
        self.outconv_Half_of_UNet = nn.Conv2d(64,out_ch,1)

        self.decision=nn.Sigmoid()


    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        #print('hx1.shape',hx1.shape)

        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        #print('hx2.shape',hx2.shape)

        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        #print('hx3.shape',hx3.shape)

        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        #print('hx4.shape',hx4.shape)

        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        #print('hx5.shape',hx5.shape)

        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        #print('hx6.shape',hx6.shape)


        #print('Decoder')
        hx6up = _upsample_like(hx6,hx5)
        #print('hx6up.shape',hx6up.shape)

        #-------------------- decoder --------------------
        #print('torch.cat((hx6up,hx5),1).shape: ',torch.cat((hx6up,hx5),1).shape)

        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        #print('hx5d.shape',hx5d.shape)
        
        hx5dup = _upsample_like(hx5d,hx4)
        #print('hx5dup.shape',hx5dup.shape)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        #print('hx4d.shape',hx4d.shape)

        hx4dup = _upsample_like(hx4d,hx3)
        #print('hx4dup.shape',hx4dup.shape)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        #print('hx3d.shape',hx3d.shape)

        hx3dup = _upsample_like(hx3d,hx2)
        #print('hx3dup.shape',hx3dup.shape)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        #print('hx2d.shape',hx2d.shape)

        hx2dup = _upsample_like(hx2d,hx1)
        #print('hx2dup.shape',hx2dup.shape)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))
        #print('hx1d.shape',hx1d.shape)


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)


        # half UNet decoder
        # convering into 64 channels
        hx2_to_64_output=self.hx2_to_64(hx2)
        hx3_to_64_output=self.hx3_to_64(hx3)
        hx4_to_64_output=self.hx4_to_64(hx4)
        hx5_to_64_output=self.hx5_to_64(hx5)
        hx6_to_64_output=self.hx6_to_64(hx6)
        #print(hx2_to_64_output.shape)
        #print(hx3_to_64_output.shape)
        #print(hx4_to_64_output.shape)
        #print(hx5_to_64_output.shape)
        #print(hx6_to_64_output.shape)


        # Half UNet decoder bring to original spatial resolution
        #print('Half UNet Decooder implementation')
        hx2_upsample_to_orig_shape=self.up_hx2_to_x(hx2_to_64_output)
        #print(hx2_upsample_to_orig_shape.shape)

        hx3_upsample_to_orig_shape=self.up_hx3_to_x(hx3_to_64_output)
        #print(hx3_upsample_to_orig_shape.shape)

        hx4_upsample_to_orig_shape=self.up_hx4_to_x(hx4_to_64_output)
        #print(hx4_upsample_to_orig_shape.shape)

        hx5_upsample_to_orig_shape=self.up_hx5_to_x(hx5_to_64_output)
        #print(hx5_upsample_to_orig_shape.shape)

        hx6_upsample_to_orig_shape=self.up_hx6_to_x(hx6_to_64_output)
        #print(hx6_upsample_to_orig_shape.shape)

        fusion_of_upsampled_x=hx1+hx2_upsample_to_orig_shape+hx3_upsample_to_orig_shape+hx4_upsample_to_orig_shape+hx5_upsample_to_orig_shape+hx6_upsample_to_orig_shape

        temp_output_1=self.final_stage_of_half_UNet(fusion_of_upsampled_x)

        #print('temp_output_1.shape:',temp_output_1.shape)
        temp_output_2=self.outconv_Half_of_UNet(temp_output_1)
        #print('temp_output_2.shape:',temp_output_2.shape)
        output_of_Half_UNet=self.decision(temp_output_2)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6,temp_output_2),1))



        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6),output_of_Half_UNet
        # return F.sigmoid(d0)

#%%


# U2NetP decoder was removed 
# only Half Unet decoder was appended.
class U2NETP_with_only_with_Half_UNet_style_Decoder(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NETP_with_only_with_Half_UNet_style_Decoder,self).__init__()

        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(64,16,64)

        #   motivated from Half UNet Decoder.......
        self.up_hx2_to_x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_hx3_to_x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_hx4_to_x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_hx5_to_x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up_hx6_to_x = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)



        self.final_stage_of_half_UNet = RSU7(64,16,64)
        self.outconv_Half_of_UNet = nn.Conv2d(64,out_ch,1)
        self.decision=nn.Sigmoid()

    def forward(self,x):

        hx = x
        #print('input:','source image',hx.shape)

        #stage 1
        hx1 = self.stage1(hx)
        #print('stage1:','RSU7','hx1.shape',hx1.shape)
        hx = self.pool12(hx1)
        #print('After maxpooling:',hx.shape)

        #stage 2
        hx2 = self.stage2(hx)
        #print('stage2:','RSU6','hx2.shape',hx2.shape)

        hx = self.pool23(hx2)
        #print('maxpooling',hx.shape)

        #stage 3
        hx3 = self.stage3(hx)
        #print('stage3:','RSU5','hx3.shape',hx3.shape)

        hx = self.pool34(hx3)
        #print('maxpooling',hx.shape)

        #stage 4
        hx4 = self.stage4(hx)
        #print('stage4:','RSU4','hx4.shape',hx4.shape)

        hx = self.pool45(hx4)
        #print('maxpooling',hx.shape)

        #stage 5
        hx5 = self.stage5(hx)
        #print('stage5:','RSU4F','hx5.shape:',hx5.shape)

        hx = self.pool56(hx5)
        #print('maxpooling',hx.shape)

        #stage 6
        hx6 = self.stage6(hx)
        #print('stage6:','RSU4F','hx6.shape',hx6.shape)



        # Half UNet decoder adding
        #print('Half UNet Decooder implementation')
        hx2_upsample_to_orig_shape=self.up_hx2_to_x(hx2)
        #print(hx2_upsample_to_orig_shape.shape)

        hx3_upsample_to_orig_shape=self.up_hx3_to_x(hx3)
        #print(hx3_upsample_to_orig_shape.shape)

        hx4_upsample_to_orig_shape=self.up_hx4_to_x(hx4)
        #print(hx4_upsample_to_orig_shape.shape)

        hx5_upsample_to_orig_shape=self.up_hx5_to_x(hx5)
        #print(hx5_upsample_to_orig_shape.shape)

        hx6_upsample_to_orig_shape=self.up_hx6_to_x(hx6)
        #print(hx6_upsample_to_orig_shape.shape)

        fusion_of_upsampled_x=hx1+hx2_upsample_to_orig_shape+hx3_upsample_to_orig_shape+hx4_upsample_to_orig_shape+hx5_upsample_to_orig_shape+hx6_upsample_to_orig_shape

        temp_output_1=self.final_stage_of_half_UNet(fusion_of_upsampled_x)
        temp_output_2=self.outconv_Half_of_UNet(temp_output_1)
        output_of_Half_UNet=self.decision(temp_output_2)

        return output_of_Half_UNet
#%%

def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        print(str(time.time() - startTime_for_tictoc) )
    else:
        print("Toc: start time not set")
#%%
class U2NETP_Encoder_only_Classifier_with_custom_channels(nn.Module):

    def __init__(self,
                 in_ch=3,
                 no_channels_dealing_in_and_out=64,
                 no_channels_dealing_in_the_middle=16):
        super(U2NETP_Encoder_only_Classifier_with_custom_channels, self).__init__()

        # ---------------- Encoder ----------------
        self.stage1 = RSU7(in_ch,
                           no_channels_dealing_in_the_middle,
                           no_channels_dealing_in_and_out)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(no_channels_dealing_in_and_out,
                           no_channels_dealing_in_the_middle,
                           no_channels_dealing_in_and_out)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(no_channels_dealing_in_and_out,
                           no_channels_dealing_in_the_middle,
                           no_channels_dealing_in_and_out)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(no_channels_dealing_in_and_out,
                           no_channels_dealing_in_the_middle,
                           no_channels_dealing_in_and_out)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(no_channels_dealing_in_and_out,
                            no_channels_dealing_in_the_middle,
                            no_channels_dealing_in_and_out)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(no_channels_dealing_in_and_out,
                            no_channels_dealing_in_the_middle,
                            no_channels_dealing_in_and_out)

        # ---------------- Feature projection ----------------
        self.reduce_conv = nn.Conv2d(no_channels_dealing_in_and_out,
                                     128,
                                     kernel_size=1,
                                     bias=False)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ---------------- Binary classifier ----------------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        hx = x

        # -------- Encoder forward --------
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)

        # -------- Classification head --------
        feat = self.reduce_conv(hx6)
        feat = self.global_pool(feat)
        logits = self.classifier(feat)

        return logits

#%%
class U2NET_Encoder_BinaryClassifier(nn.Module):
    def __init__(self, in_ch=3):
        super(U2NET_Encoder_BinaryClassifier, self).__init__()

        # ---------------- Encoder ----------------
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # ---------------- Feature reduction ----------------
        self.reduce_conv = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ---------------- Your classifier ----------------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        hx = x

        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)

        # Feature projection
        feat = self.reduce_conv(hx6)
        feat = self.global_pool(feat)

        # Binary classification
        logits = self.classifier(feat)
        return logits

