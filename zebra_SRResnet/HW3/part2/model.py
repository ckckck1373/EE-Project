import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable





# # Simple model as shown in Figure. 4
# class ResBlock(nn.Module):
#     def __init__(self, bias=True, act=nn.ReLU(True)):
#         super(ResBlock, self).__init__()
#         modules = []
#         modules.append(nn.Conv2d(16, 16, 3, padding=1))
#         modules.append(act)
#         modules.append(nn.Conv2d(16, 16, 3, padding=1))
#         self.body = nn.Sequential(*modules)
#     def forward(self, x):
#         res = self.body(x)
#         res += x
#         return res
        
# # Simple model as shown in Figure. 4
# class ResBlock(nn.Module):
#     def __init__(self, nFeat, kernel_size=3, bn=False, bias=True, act=nn.ReLU(True)):
#         super(ResBlock, self).__init__()
#         modules = []
#         modules.append(nn.Conv2d(
#             nFeat, nFeat, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias))
#         modules.append(act)
#         modules.append(nn.Conv2d(
#             nFeat, nFeat, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias))
#         self.body = nn.Sequential(*modules)
#     def forward(self, x):
#         res = self.body(x)
#         res += x
#         return res


# you can refer to ResBlock and  Figure 6 to construct the model definition of Upsampler and ZebraSRNet
#======================================================
class upsampler(nn.Module):
    def __init__(self, scale, nFeat, act=False):
        super(upsampler, self).__init__()
        modules=[]
        modules.append(nn.Conv2d(
            nFeat, 4*nFeat , 3, bias=True, padding=1 ))
        modules.append(nn.PixelShuffle(2))
        self.body = nn.Sequential(*modules)
        # add the definition of layer here

    def forward(self, x):
        output = self.body(x)
        return output


# 9/27 還沒有run過
class ZebraSRNet(nn.Module):
    def __init__(self, nFeat=16, nResBlock=2, nChannel=3, scale=4, kernel_size=3, act=nn.ReLU(True), bias=True):
        super(ZebraSRNet, self).__init__()
        
        
        
    def ResBlock(self, nFeat=16, kernel_size=3, bias=True, act=nn.ReLU(True)):   
        modules=[]
        modules.append(nn.Conv2d(
            nFeat, nFeat, kernel_size=3, padding=3//2, bias=bias))
        modules.append(act)

        modules.append(nn.Conv2d(
            nFeat, nFeat, kernel_size=3, padding=3//2, bias=bias))
        self.body = nn.Sequential(*modules)    



    def forward(self, x):
        # connect the layer together according to the Fig. 6 in the pdf 
        x = self.Conv2d(3, 16, 3, 1, bias=True)
        x1 = self.body(x)
        x1 = x1 + x
        x2 = self.body(x1)
        x2 = x2 + x1 + x
        x3 = upsampler.forward(self,x2) # 可以這樣用其他class的function嗎?
        x4 = upsampler.forward(self,x3)
        res = nn.Convd(16, 3, 3, padding=1, bias=True )
        return res





