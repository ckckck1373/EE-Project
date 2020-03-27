import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
################################################################################
# data_weight_res1_conv1 =  np.genfromtxt('res1_weight_truncated.dat', dtype = 'float32')
# data_weight_res1_conv1 = torch.from_numpy(data_weight_res1_conv1)
# data_bias = torch.zeros([24], dtype=torch.float32)
################################################################################

# ######################################## decimal -> binary funciton(小數)##################################
# def decimalToBinary(n):
#     temp_bin = [0]
#     if(len(temp_bin)==8):
#         return temp_bin

#     if(n<0):
#         temp_bin[0]=1

#     if(abs(n*2)>=1):
#         temp_bin.append(1)
#         n = abs(2*n) -1
#         decimalToBinary(n)
#     else: 
#         temp_bin.append(0)

#         n = abs(2*n)
#         decimalToBinary(n)

# ######################################## decimal -> binary funciton(整數)##################################
# temp_bin = [-1]
# def decimalToBinary(n):
#     if(len(temp_bin)==12):
#         return temp_bin[:11]

#     if(n%2==1):
#         temp_bin.insert(0, 1)
#         n = n // 2
#         decimalToBinary(n)
#     else: 
#         temp_bin.insert(0, 0)
#         n = n // 2
#         decimalToBinary(n)

######################################################################################################



class ResBlock(nn.Module):
    def __init__(self, nFeat, kernel_size=3, bn=False, bias=True, act=nn.ReLU(True)):
        super(ResBlock, self).__init__()
        #for extract
        modules = []
        modules.append(nn.Conv2d(
            nFeat, nFeat, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias))

        modules.append(act)
        modules.append(nn.Conv2d(
            nFeat, nFeat, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res




# ##################################### new version of ResBlock####################################
# class ResBlock(nn.Module):
#     def __init__(self, nFeat, kernel_size=3, bn=False, bias=True, act=nn.ReLU(True)):
#         super(ResBlock, self).__init__()
#         self.conv1 = nn.Conv2d(nFeat, nFeat, 3, 1, 1)
#         self.relu = act
#         self.conv2 = nn.Conv2d(nFeat, nFeat, 3, 1, 1)



#     def forward(self, x):
#         res = self.conv1(x)
#         res = self.relu(res)
#         res = self.conv2(res)
#         res += x
#         return res
######################################################################################################

        
class upsampler(nn.Module):
    def __init__(self, scale, nFeat, act=False):
        super(upsampler, self).__init__()
        # add the definition of layer here
        self.conv = nn.Conv2d(nFeat, nFeat*4, 3, 1, 1)
        ### nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.shuffle = nn.PixelShuffle(scale)
        ### nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        return x



class onepieceSRNet(nn.Module):
    def __init__(self, nFeat, nResBlock, nChannel=3, scale=4):
        super(onepieceSRNet, self).__init__()
        # add the definition of layer here
        self.conv1 = nn.Conv2d(nChannel, nFeat, 3, 1, 1)
        modules = []
        for _ in range(nResBlock):
            modules.append(ResBlock(nFeat))
        self.body = nn.Sequential(*modules)
        self.upsamp1 = upsampler(scale//2, nFeat)
        self.upsamp2 = upsampler(scale//2, nFeat)
        self.conv2 = nn.Conv2d(nFeat, nChannel, 3, 1, 1)
    
    def forward(self, x):

        
        #######################not need      #######################
        # with open("conv1_out.dat", "w") as f1:
        #     for i in range(24):
        #         for j in range(180):
        #             for k in range(320): 

        #                 temp1 = x[0][i][j][k].data.tolist()
        #                 f1.write(str(temp1) + " ")
        #             f1.write("\n")    
        # f1.close()
        ##############################################################


        ######################## print ############################
        # temp_output_data = x # dtype = tensor.float32
        # printdata(temp_output_data)

        ###################################################################################
        # self.conv1 = nn.Conv2d(nFeat, nFeat, 3, 1, 1)
        # self.conv1.weight = data_weight_res1_conv1
        # self.conv1.bias = data_bias
        ####################################################################################

        x = self.conv1(x)

        ## take data function ##
        temp_output_data = x # dtype = tensor.float32
        printdata(temp_output_data)
        ##

        f_x = self.body(x)
        f_x = f_x + x

        f_x = self.upsamp1(f_x)
        f_x = self.upsamp2(f_x)
        output = self.conv2(f_x)
        return output

        ##############################################################





#################################################write conv1_output function##############################################
def printdata(temp_output_data):
    temp_bin = [-1]
    ####### prepare ######
    array_total = temp_output_data[:1, :24, :180,:320]


    padding = nn.ZeroPad2d(3)
    array_total = padding(array_total)
    

    # print(array_total.shape)
    array_total = array_total[:1, :24, 2:186, 2:326]
    # print(array_total.shape)


    #################################### decimal (Integar part)-> binary funciton####################################

    def DecimalINTto2comple(n):
        result = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        change = False
        if(n<0):
            change = True
            n = -n-1
        
        binary_number = ("{:0>16b}".format(n))

        if(change == True):
            for i in range(16):
                if(int(binary_number[i])==0):
                    result[i] = 1
                else:
                    result[i] = 0
        else: 
            for i in range(16):
                if(int(binary_number[i])==0):
                    result[i] = 0
                else:
                    result[i] = 1
        
        return result[0:16]


    ###############################################################################################################

    with open("conv1_output_bank0_2'scomplement.dat", "w") as fb1:
        with open("conv1_output_bank1_2'scomplement.dat", "w") as fb2:
            with open("conv1_output_bank2_2'scomplement.dat", "w") as fb3:
                with open("conv1_output_bank3_2'scomplement.dat", "w") as fb4:
                    for y_index in range(92):
                        for x_index in range(162):
                            for ch in range(24): ## after ch3  temp=0
                                for j in range(2):
                                    for i in range(2):

                                        # decimalToBinaryINT(int(array_total[0][ch][y_index*2+j][x_index*2+i]))

                                        temp_data = DecimalINTto2comple(int(array_total[0][ch][y_index*2+j][x_index*2+i]))

                                        if((x_index%2==0)&(y_index%2==0)): # orange bank
                                            if((ch==23)&(i==1)&(j==1)):
                                                fb1.write(str(temp_data[0])+str(temp_data[1])+str(temp_data[2])+str(temp_data[3])+str(temp_data[4])+str(temp_data[5])+str(temp_data[6])+str(temp_data[7])+str(temp_data[8])+str(temp_data[9])+str(temp_data[10])+str(temp_data[11])+str(temp_data[12])+str(temp_data[13])+str(temp_data[14])+str(temp_data[15]) +"\n")
                                                temp_bin = [-1]
                                            else:
                                                fb1.write(str(temp_data[0])+str(temp_data[1])+str(temp_data[2])+str(temp_data[3])+str(temp_data[4])+str(temp_data[5])+str(temp_data[6])+str(temp_data[7])+str(temp_data[8])+str(temp_data[9])+str(temp_data[10])+str(temp_data[11])+str(temp_data[12])+str(temp_data[13])+str(temp_data[14])+str(temp_data[15]) +"_")
                                                temp_bin = [-1]
                                        elif((x_index%2==1)&(y_index%2==0)): # yellow bank
                                            if((ch==23)&(i==1)&(j==1)):
                                                fb2.write(str(temp_data[0])+str(temp_data[1])+str(temp_data[2])+str(temp_data[3])+str(temp_data[4])+str(temp_data[5])+str(temp_data[6])+str(temp_data[7])+str(temp_data[8])+str(temp_data[9])+str(temp_data[10])+str(temp_data[11])+str(temp_data[12])+str(temp_data[13])+str(temp_data[14])+str(temp_data[15]) + "\n")
                                                temp_bin = [-1]
                                            else:
                                                fb2.write(str(temp_data[0])+str(temp_data[1])+str(temp_data[2])+str(temp_data[3])+str(temp_data[4])+str(temp_data[5])+str(temp_data[6])+str(temp_data[7])+str(temp_data[8])+str(temp_data[9])+str(temp_data[10])+str(temp_data[11])+str(temp_data[12])+str(temp_data[13])+str(temp_data[14])+str(temp_data[15]) +"_")
                                                temp_bin = [-1]

                                        elif((x_index%2==0)&(y_index%2==1)): # Blue bank
                                            if((ch==23)&(i==1)&(j==1)):
                                                fb3.write(str(temp_data[0])+str(temp_data[1])+str(temp_data[2])+str(temp_data[3])+str(temp_data[4])+str(temp_data[5])+str(temp_data[6])+str(temp_data[7])+str(temp_data[8])+str(temp_data[9])+str(temp_data[10])+str(temp_data[11])+str(temp_data[12])+str(temp_data[13])+str(temp_data[14])+str(temp_data[15]) +"\n")
                                                temp_bin = [-1]

                                            else:
                                                fb3.write(str(temp_data[0])+str(temp_data[1])+str(temp_data[2])+str(temp_data[3])+str(temp_data[4])+str(temp_data[5])+str(temp_data[6])+str(temp_data[7])+str(temp_data[8])+str(temp_data[9])+str(temp_data[10])+str(temp_data[11])+str(temp_data[12])+str(temp_data[13])+str(temp_data[14])+str(temp_data[15]) +"_")
                                                temp_bin = [-1]

                                        else: # Green bank 
                                            if((ch==23)&(i==1)&(j==1)):
                                                fb4.write(str(temp_data[0])+str(temp_data[1])+str(temp_data[2])+str(temp_data[3])+str(temp_data[4])+str(temp_data[5])+str(temp_data[6])+str(temp_data[7])+str(temp_data[8])+str(temp_data[9])+str(temp_data[10])+str(temp_data[11])+str(temp_data[12])+str(temp_data[13])+str(temp_data[14])+str(temp_data[15]) +"\n")
                                                temp_bin = [-1]

                                            else:
                                                fb4.write(str(temp_data[0])+str(temp_data[1])+str(temp_data[2])+str(temp_data[3])+str(temp_data[4])+str(temp_data[5])+str(temp_data[6])+str(temp_data[7])+str(temp_data[8])+str(temp_data[9])+str(temp_data[10])+str(temp_data[11])+str(temp_data[12])+str(temp_data[13])+str(temp_data[14])+str(temp_data[15])+"_") 
                                                temp_bin = [-1]    
        
                fb4.close()
            fb3.close()
        fb2.close()
    fb1.close()
        

################################################VARIFY#########################################################