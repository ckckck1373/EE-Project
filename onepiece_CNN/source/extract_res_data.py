import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

data_input = np.genfromtxt('res1_weight_truncated.dat', dtype = 'float32')
data_weight_res1_conv1 =  np.genfromtxt('res1_weight_truncated.dat', dtype = 'float32')
data_weight_res1_conv1 = torch.from_numpy(data_weight_res1_conv1)
data_bias = torch.zeros([24], dtype=torch.float32)

class Conv:
    def __init__(self, nFeat, kernel_size=3, bn=False, bias=True, act=nn.ReLU(True)):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(24, 24, 3, 1, 1)
        self.conv1.weight = data_weight_res1_conv1
        self.conv1.bias = data_bias

    def forward(self, x):
        f_x = self.conv1(x)
        return f_x







