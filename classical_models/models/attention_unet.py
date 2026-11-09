#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:00:39 2023

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 13:43:29 2023

@author: medimg
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 16:29:18 2023

@author: venkatesh
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d( in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        #print("#"*50)
        #print("Attention Block")
        g1 = self.W_gate(gate)
        #print(g1.shape)
        x1 = self.W_x(skip_connection)
        #print(x1.shape)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        #print("#"*50)
        #print(out.shape)
        return out


class Attention_UNet_updated_version_1(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(Attention_UNet_updated_version_1, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.decision=nn.Sigmoid()


    def forward(self, x):
        """        e : encoder layers d : decoder layers s : skip-connections from encoder layers to decoder layers """
        e1 = self.Conv1(x)
        #print('e1',e1.shape)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)
        #print('e2',e2.shape)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)
        #print('e3',e3.shape)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)
        #print('e4',e4.shape)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)
        #print('e5',e5.shape)
        #print("#"*50)
        
        #1024--->512)
        d5 = self.Up5(e5)
        #print('d5',d5.shape)
        #print("#"*50)

        #print('Att5: F_g=512, F_l=512, n_coefficients=256')
        s4 = self.Att5(gate=d5, skip_connection=e4)
        # #print('s4.shape',s4.shape)

        d5 = torch.cat((s4, d5), dim=1) 
        # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpConv5(d5)

        #1024--->512
        d4 = self.Up4(d5)
        # #print("Att4: F_g=256, F_l=256, n_coefficients=128)")
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)


        d4 = self.UpConv4(d4)
        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        # print(d2.shape)

        out = self.Conv(d2)
        out=self.decision(out)

        # return out,d2
        return out

#%%

def getnumberofparams(model):
    pp=0
    for p in (model.parameters()):
        nn=1
        for s in (p.size()):
            nn = nn*s
        pp += nn
    return pp,  (pp*32)/(8*1024*1024)
#%%
#%%

# def tic():
#     # Homemade version of matlab tic and toc functions
#     import time
#     global startTime_for_tictoc
#     startTime_for_tictoc = time.time()

# def toc():
#     import time
#     if 'startTime_for_tictoc' in globals():
#         print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
#         print(str(time.time() - startTime_for_tictoc) )
#     else:
#         print("Toc: start time not set")
# #%%
# model=Attention_UNet_updated_version_1(img_ch=3, output_ch=1)
# torch.save(model.state_dict(),'Attention_UNet_updated_version_1.pth')

# model=model.cuda()
# model=model.float()

# #%%

# input_data=torch.randn(1,3,448,320)
# input_data=input_data.cuda()
# input_data=input_data.float()
# #print(input.shape)

# #%%
# with torch.no_grad():
#     for i in range(100):

#         #print(i)
#         tic()
#         # output,d1,d2,d3,d4,d5,d6,d7=model(input)
#         output=model(input_data)
#         toc()

# #%%
# from torchsummary import summary
# from pthflops import count_ops

# summary(model, (3, 448, 320))
# #%%
# torch.save(model.state_dict(),'U2NETP_with_half_unet_decoder_without_internal_decoders_with_6_layers.pth')
# #%%
# import re
# from ptflops import get_model_complexity_info
# from pthflops import count_ops

# macs, params = get_model_complexity_info(model, (3, 448, 320), as_strings=True,
# print_per_layer_stat=True, verbose=True)
# # Extract the numerical value
# flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
# # Extract the unit
# flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
# print('Computational complexity: {:<8}'.format(macs))
# print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
# print('Number of parameters: {:<8}'.format(params))
# #%%
# from thop import profile

# flops, params = profile(model, inputs=(input_data,))

# print(f"Total FLOPs: {flops}")
# print(f"Total parameters: {params}")
# #%%
# from torchprofile import profile_macs

# flops = profile_macs(model, input_data)
# print("Total FLOPs:", flops)
