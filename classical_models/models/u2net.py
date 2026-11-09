
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
        #print('input:','source image',hx.shape)

        #stage 1
        #print('input:',hx.shape)
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        #print('output:')
        #print('stage1:','RSU7','hx1.shape',hx1.shape)

        #stage 2
        #print('input:','source image',hx.shape)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        #print('stage2:','RSU6','hx2.shape',hx2.shape)

        #stage 3
        #print('input:','source image',hx.shape)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        #print('stage3:','RSU5','hx3.shape',hx3.shape)

        #stage 4
        #print('input:','source image',hx.shape)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        #print('stage4:','RSU4','hx4.shape',hx4.shape)

        #stage 5
        #print('input:','source image',hx.shape)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        #print('stage5:','RSU4F','hx5.shape:',hx5.shape)

        #stage 6
        #print('input:','source image',hx.shape)

        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)
        #print('stage6:','RSU4F','hx6.shape',hx6.shape)

        #-------------------- decoder --------------------
        #print('input:','source image',torch.cat((hx6up,hx5),1).shape)

        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)
        #print('stage5d:','RSU4F','hx5d.shape',hx5d.shape)

        #print('input:','source image',torch.cat((hx5dup,hx4),1).shape)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)
        #print('stage4d:','RSU4','hx4d.shape',hx4d.shape)

        #print('input:','source image',torch.cat((hx4dup,hx3),1).shape)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)
        #print('stage3d:','RSU5','hx3d.shape',hx3d.shape)

        #print('input:','source image',torch.cat((hx3dup,hx2),1).shape)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)
        #print('stage2d:','RSU6','hx2d.shape',hx2d.shape)

        #print('input:','source image',torch.cat((hx2dup,hx1),1).shape)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))
        feature_vector=hx1d
        # #print('hx1d',hx1d.shape)
        #print('stage1d:','RSU7','hx1d.shape',hx1d.shape)

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

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
        # return F.sigmoid(d0),feature_vector
### U^2-Net small ###
class U2NETP(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NETP,self).__init__()

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

    def forward(self,x):

        hx = x
        ##print('x.shape',x.shape)
        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        ##print('hx.shape',hx.shape)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        ##print('hx.shape',hx.shape)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        ##print('hx.shape',hx.shape)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        ##print('hx.shape',hx.shape)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        ##print('hx.shape',hx.shape)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))
        # #print('hx1d',hx1d.shape)
        feature_vector=hx1d

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

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
        # return F.sigmoid(d0),feature_vector

        # return F.sigmoid(d0)
#%%
# from torchsummary import summary
# from pthflops import count_ops
# from thop import profile

# input=torch.randn(1,3,448,320)
# input=input.float()
# input=input.cuda()

# model = U2NET(3,1)
# model=model.float()
# model=model.cuda()

# with torch.no_grad():
#     for i in range(1):
#         output=model(input)
#         # ##print(output[0].shape)

# # summary(model, (3, 448, 320))
# #%%
# count_ops(model, input)
#%%

# def tic():
#     # Homemade version of matlab tic and toc functions
#     import time
#     global startTime_for_tictoc
#     startTime_for_tictoc = time.time()

# def toc():
#     import time
#     if 'startTime_for_tictoc' in globals():
#         ##print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
#         #print(str(time.time() - startTime_for_tictoc) )
#     else:
#         #print("Toc: start time not set")

# #%%


# tic()
# model=U2NET(3,1)
# # model=U2NET(3,1)

# model=model.float()
# model=model.cuda()
# model=model.eval()
# toc()
    
# with torch.no_grad():

#     x= torch.rand(1,3,448,320)
#     x=x.cuda()
#     x=x.float()
#     yy=model(x)
#     ##print('Out Shape :', yy.shape)
# #%%    
# # with torch.no_grad():
# #     for i in range(1):
# #             tic()
# #             x= torch.rand(1,3,448,320)
# #             x=x.cuda()
# #             x=x.float()

# #             output=model(x)
# #             ##print(x.shape)
# #             toc()
# # torch.save(model.state_dict(), 'temp.pth')

# #%%
# from torchsummary import summary

# summary(model, (3,448,320))
# #%%




# temp=model.get_submodule('stage1')
# torch.save(temp.state_dict(),'stage1.pth')
# summary(temp, (3, 448, 320))


# input_data=torch.randn(1,3,448,320)
# input_data=input_data.float()
# input_data=input_data.cuda()
# flops, params = profile(temp, inputs=(input_data,))

# flops=flops/1000000000
# #print(f"Total FLOPs: {flops}")
# #print(f"Total parameters: {params}")

# #%%
# temp=model.get_submodule('stage2')  
# torch.save(temp.state_dict(),'stage2.pth')
# summary(temp, (64, 224, 160))

# input_data=torch.randn(1, 64, 224, 160)
# input_data=input_data.float()
# input_data=input_data.cuda()
# flops, params = profile(temp, inputs=(input_data,))

# flops=flops/1000000000
# #print(f"Total FLOPs: {flops}")
# #print(f"Total parameters: {params}")
# #%%
# temp=model.get_submodule('stage3')
# torch.save(temp.state_dict(),'stage3.pth')
# summary(temp, (128, 112, 80))

# input_data=torch.randn(1,128, 112, 80)
# input_data=input_data.float()
# input_data=input_data.cuda()
# flops, params = profile(temp, inputs=(input_data,))

# flops=flops/1000000000
# #print(f"Total FLOPs: {flops}")
# #print(f"Total parameters: {params}")
# #%%

# temp=model.get_submodule('stage4')
# torch.save(temp.state_dict(),'stage4.pth')
# summary(temp, (256, 56, 40))

# input_data=torch.randn(1,256, 56, 40)
# input_data=input_data.float()
# input_data=input_data.cuda()
# flops, params = profile(temp, inputs=(input_data,))

# flops=flops/1000000000
# #print(f"Total FLOPs: {flops}")
# #print(f"Total parameters: {params}")
# #%%
# temp=model.get_submodule('stage5')
# torch.save(temp.state_dict(),'stage5.pth')
# summary(temp, (512, 28, 20))

# input_data=torch.randn(1,512, 28, 20)
# input_data=input_data.float()
# input_data=input_data.cuda()
# flops, params = profile(temp, inputs=(input_data,))

# flops=flops/1000000000
# #print(f"Total FLOPs: {flops}")
# #print(f"Total parameters: {params}")
# #%%
# temp=model.get_submodule('stage6')
# torch.save(temp.state_dict(),'stage6.pth')
# summary(temp, ( 512, 14, 10))     

# input_data=torch.randn(1,512, 14, 10)
# input_data=input_data.float()
# input_data=input_data.cuda()
# flops, params = profile(temp, inputs=(input_data,))

# flops=flops/1000000000
# #print(f"Total FLOPs: {flops}")
# #print(f"Total parameters: {params}")

# #%%
# temp=model.get_submodule('stage5d')
# torch.save(temp.state_dict(),'stage5d.pth')
# summary(temp, (1024, 28, 20))


# input_data=torch.randn(1,1024, 28, 20)
# input_data=input_data.float()
# input_data=input_data.cuda()
# flops, params = profile(temp, inputs=(input_data,))

# flops=flops/1000000000
# #print(f"Total FLOPs: {flops}")
# #print(f"Total parameters: {params}")
# #%%
# temp=model.get_submodule('stage4d')
# torch.save(temp.state_dict(),'stage4d.pth')     
# summary(temp, (1024, 56, 40))

# input_data=torch.randn(1,1024,56,40)
# input_data=input_data.float()
# input_data=input_data.cuda()
# flops, params = profile(temp, inputs=(input_data,))

# flops=flops/1000000000
# #print(f"Total FLOPs: {flops}")
# #print(f"Total parameters: {params}")

# #%%
# temp=model.get_submodule('stage3d')
# torch.save(temp.state_dict(),'stage3d.pth')
# summary(temp, (512, 112, 80))

# input_data=torch.randn(1,512,112,80)
# input_data=input_data.float()
# input_data=input_data.cuda()
# flops, params = profile(temp, inputs=(input_data,))

# flops=flops/1000000000
# #print(f"Total FLOPs: {flops}")
# #print(f"Total parameters: {params}")
# #%%

# temp=model.get_submodule('stage2d')
# torch.save(temp.state_dict(),'stage2d.pth')
# summary(temp, (256, 224, 160))

# input_data=torch.randn(1,256,224,160)
# input_data=input_data.float()
# input_data=input_data.cuda()
# flops, params = profile(temp, inputs=(input_data,))

# flops=flops/1000000000
# #print(f"Total FLOPs: {flops}")
# #print(f"Total parameters: {params}")
# #%%

# temp=model.get_submodule('stage1d')
# torch.save(temp.state_dict(),'stage1d.pth')
# summary(temp, (128, 448, 320))

# input_data=torch.randn(1,128,448,320)
# input_data=input_data.float()
# input_data=input_data.cuda()
# flops, params = profile(temp, inputs=(input_data,))

# flops=flops/1000000000
# #print(f"Total FLOPs: {flops}")
# #print(f"Total parameters: {params}")
# #%%





# temp=model.get_submodule('final_stage_of_half_UNet')
# torch.save(temp.state_dict(),'final_stage_of_half_UNet.pth')
# summary(temp, (64, 448, 320))





# input_data=torch.randn(1,64,448,320)
# input_data=input_data.float()
# input_data=input_data.cuda()
# flops, params = profile(temp, inputs=(input_data,))

# flops=flops/1000000000
# #print(f"Total FLOPs: {flops}")
# #print(f"Total parameters: {params}")


#%%

# ##print('summary of the computations:')
# from pthflops import count_ops
# count_ops(model, x)
# #%%
# summary(model, (3, 448,320))
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
# #%%
# #print('Computational complexity: {:<8}'.format(macs))
# #print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
# #print('Number of parameters: {:<8}'.format(params))
# #%%
# from thop import profile

# flops, params = profile(model, inputs=(x,))

# #print(f"Total FLOPs: {flops}")
# #print(f"Total parameters: {params}")
# #%%
# from torchprofile import profile_macs

# flops = profile_macs(model, x)
# #print("Total FLOPs:", flops)
