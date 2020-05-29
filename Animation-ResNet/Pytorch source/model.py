import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import argparse
torch.set_printoptions(precision=10) ## PyTorch uses less precision to print...
##
import matplotlib.pyplot as plt
from matplotlib import interactive
from collections import defaultdict
##
## quantization info ##

bw_activation=16
fl_point=7 #8
quantization=0
#######################


## plot hist ##
plot_enable=0
###############

### global variable ###
num = 0
num_up = 0
#######################

### Important ###
################################################################################
# 1. change numpy data to tensor data type would not rounding(only print less point)
################################################################################
activation_fl_dict={}
# origin
# - : too fat
# + : too thin
# for layer in range(14):
#     # if(layer==1): # Conv1
#     #     activation_fl_dict[layer] = fl_point
#     # elif(layer==2): # Res1_conv1
#     #     activation_fl_dict[layer] = fl_point +1
#     # elif(layer==3): # Res1_conv2
#     #     activation_fl_dict[layer] = fl_point -1
#     # elif(layer==4): # Res2_conv1
#     #     activation_fl_dict[layer] = fl_point +2
#     # elif(layer==5): # Res2_conv2
#     #     activation_fl_dict[layer] = fl_point -1
#     # elif(layer==6): # Res3_conv1
#     #     activation_fl_dict[layer] = fl_point +2
#     # elif(layer==7): # Res3_conv2
#     #     activation_fl_dict[layer] = fl_point -1
#     # elif(layer==8): # Res4_conv1
#     #     activation_fl_dict[layer] = fl_point +2
#     # elif(layer==9): # Res4_conv2
#     #     activation_fl_dict[layer] = fl_point -1 #0
#     # elif(layer==10): # Up1
#     #     activation_fl_dict[layer] = fl_point
#     # elif(layer==11): # Up2
#     #     activation_fl_dict[layer] = fl_point -1
#     # elif(layer==12): # Conv2
#     #     activation_fl_dict[layer] = fl_point #-1
#     # else: 
#     #     activation_fl_dict[layer] = fl_point  #-1

# screen_5 epoch200 version2
# - : too fat
# + : too thin
for layer in range(14):
    # if(layer==1): # Conv1
    #     activation_fl_dict[layer] = fl_point
    # elif(layer==2): # Res1_conv1
    #     activation_fl_dict[layer] = fl_point +2
    # elif(layer==3): # Res1_conv2
    #     activation_fl_dict[layer] = fl_point -1
    # elif(layer==4): # Res2_conv1
    #     activation_fl_dict[layer] = fl_point +2
    # elif(layer==5): # Res2_conv2
    #     activation_fl_dict[layer] = fl_point -1
    # elif(layer==6): # Res3_conv1
    #     activation_fl_dict[layer] = fl_point +2
    # elif(layer==7): # Res3_conv2
    #     activation_fl_dict[layer] = fl_point -1
    # elif(layer==8): # Res4_conv1
    #     activation_fl_dict[layer] = fl_point +2
    # elif(layer==9): # Res4_conv2
    #     activation_fl_dict[layer] = fl_point -1 #0
    # elif(layer==10): # Up1
    #     activation_fl_dict[layer] = fl_point -1
    # elif(layer==11): # Up2
    #     activation_fl_dict[layer] = fl_point -1
    # elif(layer==12): # Conv2
    #     activation_fl_dict[layer] = fl_point -1
    # else: 
        activation_fl_dict[layer] = fl_point  #-1
print(activation_fl_dict)

#### ToBinary ###
def bindigits(n, bits):
    s = bin(n & int("1"*bits, 2))[2:]
    return ("{0:0>%s}" % (bits)).format(s)



#### ResBlock ###
class ResBlock(nn.Module):
    def __init__(self, nFeat, kernel_size=3, bn=False, bias=True, act=nn.ReLU(inplace=True)):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(nFeat, nFeat, 3, 1, 1)
        self.relu = act
        self.conv2 = nn.Conv2d(nFeat, nFeat, 3, 1, 1)
   
    def forward(self, x):
        global num
        if(quantization==0):
            num=num+1
            ## origin
            res = self.conv1(x)


            res = self.relu(res)
            if(plot_enable):
                plt_his('res' + str(num) +'_' + 'conv1', res, 2*num)

            #printdata(res*256, "RES1_1_golden", False)
            res = self.conv2(res)
            if(plot_enable):
                plt_his('res' + str(num) +'_' + 'conv2', res, 2*num+1)
            res += x
            # printdata(res*256, "RES1_2_forward_golden", False)

            if(num==4): 
                num=0
            return res
            ##
        else:
            ##quantize
            num=num+1
            res = self.relu(self.conv1(x))
            res = self.quantize_activation(res, bw_activation, activation_fl_dict[2*num])

            if(plot_enable):
                plt_his('res' + str(num) +'_' + 'conv1', res, 2*num)

            # if(num==1):
            #     printdata(res*(2**activation_fl_dict[2]), "RES1_1_golden", False)

            res = self.conv2(res)
            #res = self.quantize_activation(res, bw_activation, fl_point)
            res = res + x

            
            res = self.quantize_activation(res, bw_activation, activation_fl_dict[2*num+1]) 

            if(plot_enable):
                plt_his('res' + str(num) +'_' + 'conv2', res, 2*num+1)

            # if(num==1):
            #     printdata(res*(2**activation_fl_dict[3]), "RES1_2_forward_golden", False)
                

            if(num==4): 
                num=0
            return res
            

    def quantize_activation(self, num_group, bit_width, frational_length):
        interval = 2 ** (-1 * frational_length)
        half_interval = interval / 2
        max_val = (2 ** (bit_width - 1) - 1) * interval
        min_val = - (2 ** (bit_width - frational_length - 1))

        quan_data = torch.floor((num_group.data + half_interval) / interval)
        quan_data = quan_data * interval

        quan_data[quan_data >= max_val] = max_val
        quan_data[quan_data <= min_val] = min_val
        num_group.data = quan_data.cuda()
        return num_group
        


class upsampler(nn.Module):
    def __init__(self, scale, nFeat, act=False):
        super(upsampler, self).__init__()
        self.conv = nn.Conv2d(nFeat, nFeat*4, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        global num_up
        if(quantization==0):
            num_up=num_up+1
            x = self.conv(x)
            x = self.shuffle(x)
            if(num_up==2): 
                num_up=0
            return x
        else:
            # quantization
            num_up=num_up+1
            x = self.conv(x)
            x = self.quantize_activation(x, bw_activation, activation_fl_dict[num_up+9])
            x = self.shuffle(x)
            #x = self.quantize_activation(x, bw_activation, fl_point)
            if(num_up==2): 
                num_up=0
            return x

    def quantize_activation(self, num_group, bit_width, frational_length):
        interval = 2 ** (-1 * frational_length)
        half_interval = interval / 2

        max_val = (2 ** (bit_width - 1) - 1) * interval
        min_val = - (2 ** (bit_width - frational_length - 1))

        quan_data = torch.floor((num_group.data + half_interval) / interval)
        quan_data = quan_data * interval

        quan_data[quan_data >= max_val] = max_val
        quan_data[quan_data <= min_val] = min_val
        num_group.data = quan_data.cuda()
        return num_group
    


class onepieceSRNet(nn.Module):
    def __init__(self, nFeat=24, nResBlock=4, nChannel=3, scale=4):
        super(onepieceSRNet, self).__init__()
        # add the definition of layer here
        self.conv1 = nn.Conv2d(nChannel, nFeat, 3, 1, 1)
        self.res1 = ResBlock(nFeat)
        self.res2 = ResBlock(nFeat)
        self.res3 = ResBlock(nFeat)
        self.res4 = ResBlock(nFeat)
        self.upsamp1 = upsampler(scale//2, nFeat)
        self.upsamp2 = upsampler(scale//2, nFeat)
        self.conv2 = nn.Conv2d(nFeat, nChannel, 3, 1, 1)
        
    
    def forward(self, x):
        ## origin
        if(quantization==0):
            #printdata(x*128, "conv1_input", False)
            x = self.conv1(x)
            if(plot_enable):
                plt_his("conv1_out", x, 1)

            #printdata(x*256, "conv1_golden", False)

            f_x = self.res1(x)
            f_x = self.res2(f_x)
            f_x = self.res3(f_x)
            f_x = self.res4(f_x)

            # printdata(f_x*256, "RES4_2_forward_golden", False)

            # 
            # f_x = f_x + x 
            # 
            f_x = self.upsamp1(f_x)

            if(plot_enable):
                plt_his("up1_out", f_x, 10)

            # printdata(f_x*256, "UP_2_golden", False)

            f_x = self.upsamp2(f_x)

            if(plot_enable):
                plt_his("up2_out", f_x, 11)

            #printdata(f_x*256, "UP_2_golden", False)
            output = self.conv2(f_x)

            if(plot_enable):
                plt_his("out", output, 12)

            #printdata(output*256, "Output_golden", False)
            
            return output
        else: 
            ## quantization
            
            #printdata(x*128), "input_image", False)

            x = self.conv1(x)
            if(plot_enable):
                plt_his("conv1_out", x, 1)

            x = self.quantize_activation(x, bw_activation, activation_fl_dict[1])
            #printdata(x*(2**activation_fl_dict[0]), "conv1_golden", False)

           # printdataindecimal(x*128, "golden_conv1")
            f_x = self.res1(x)
            f_x = self.res2(f_x)
            f_x = self.res3(f_x)
            f_x = self.res4(f_x)

            #printdata(f_x*(2**activation_fl_dict[9]), "RES4_2_forward_golden", False)
            f_x = self.upsamp1(f_x)  
            if(plot_enable):
                plt_his("up1_out", f_x, 10)
            #printdata(f_x*(2**activation_fl_dict[10]), "UP_1_golden", False)

            f_x = self.upsamp2(f_x)
            if(plot_enable):
                plt_his("up2_out", f_x, 11)
            # printdata(f_x*(2**activation_fl_dict[11]), "UP_2_golden", False)
             

            output = self.conv2(f_x)
            
            output = self.quantize_activation(output, bw_activation, activation_fl_dict[12])




            if(plot_enable):
                plt_his("out", output, 12)

            # printdata(output*(2**activation_fl_dict[12]), "conv2_golden", False)
            return output
            


    # def collect_data(self, x):
    #     data_box = {}
    #     data_box['input'] = x
    #     data_box['conv1'] = self.conv1(data_box['input'])
    #     data_box['res1_forward'] = self.res1(data_box['conv1'])
    #     data_box['res2_forward'] = self.res2(data_box['res1_forward'])
    #     data_box['res3_forward'] = self.res3(data_box['res1_forward'])
    #     data_box['res4_forward'] = self.res4(data_box['res1_forward'])
    #     data_box['upsamp1'] = self.upsamp1(data_box['res4_forward'])
    #     data_box['upsamp2'] = self.upsamp2(data_box['upsamp1'])
    #     data_box['output'] = self.conv2(data_box['upsamp2'])
    #     # data_box['conv2'] = self.relu(self.conv2(data_box['conv1']))
    #     # k = data_box['conv3'].view(-1, 64 * 4 * 4)
    #     # data_box['fc1'] = self.relu(self.fc1(k))
    #     # data_box['fc2'] = self.fc2(data_box['fc1'])
    #     return data_box

    # def quan_input(self, x, fl_dict):
    #     print('quan_input')
    #     x = self.quantize_activation(x, args.bw_activation, fl_dict['input'])
    #     return x

    # def quan_conv1(self, x, fl_dict):
    #     print('quan_conv1')
    #     x = self.quantize_activation(x, args.bw_activation, fl_dict['input'])
    #     x = self.quantize_activation(x, args.bw_activation, fl_dict['unshuffle'])
    #     x = self.conv1(x)
    #     x = self.quantize_activation(x, args.bw_activation, fl_dict['conv1'])
    #     return x

    def quantize_activation(self, num_group, bit_width, frational_length):
        interval = 2 ** (-1 * frational_length)
        half_interval = interval / 2

        max_val = (2 ** (bit_width - 1) - 1) * interval
        min_val = - (2 ** (bit_width - frational_length - 1))

        quan_data = torch.floor((num_group.data + half_interval) / interval)
        quan_data = quan_data * interval

        quan_data[quan_data >= max_val] = max_val
        quan_data[quan_data <= min_val] = min_val
        num_group.data = quan_data.cuda()
        return num_group
        


########################################plt########################################################
def plt_his(name, x, num):
    
    plt.figure(num)
    x = x.cpu()
    x = x.detach().numpy() 

    x = x.ravel() # turn data to 1D array
    n_bins=35
    plt.title(str(name))
    plt.hist(x, n_bins)
    
    if(num!=12):
        interactive(True)
        plt.show()
        plt.savefig("./plot/" + name + ".png")
    else:
        interactive(False)
        plt.savefig("./plot/" + name + ".png")
        plt.show()
        


    

    

#################################################write conv1_output function##############################################
def printdata(temp_output_data, filename, en_relu): 
    temp_bin = [-1]
    ####### prepare ######
    
    if(filename=="UP_1_golden"):
        array_total = temp_output_data[:1, :24, :360,:640] # upsamp1 
    elif((filename=="UP_2_golden") or (filename=="conv2_golden")):
        array_total = temp_output_data[:1, :24, :720,:1280] # upsamp2 and output
    else:
        array_total = temp_output_data[:1, :24, :180,:320]

    padding = nn.ZeroPad2d(3)
    array_total = padding(array_total)
    
    if(filename=="UP_1_golden"):
        size_x=322
        size_y=182
        array_total = array_total[:1, :24, 2:366, 2:646] # upsamp1 
    elif((filename=="UP_2_golden") or (filename=="conv2_golden")):
        size_x=642
        size_y=362
        array_total = array_total[:1, :24, 2:726, 2:1286] # upsamp2 and output
    else:
        size_x=162
        size_y=92
        array_total = array_total[:1, :24, 2:186, 2:326]


    ######################################### turn to binary bits###################################################
    def bindigits(n, bits):
        s = bin(n & int("1"*bits, 2))[2:]
        return ("{0:0>%s}" % (bits)).format(s)
    #################################################################################################################

    #################################### decimal (Integar part)-> binary funciton####################################

    # def DecimalINTto2comple(n):
    #     result = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #     change = False
    #     if(n<0):
    #         change = True
    #         n = -n-1
        
    #     binary_number = ("{:0>20b}".format(n))

    #     if(change == True):
    #         for i in range(20):
    #             if(int(binary_number[i])==0):
    #                 result[i] = 1
    #             else:
    #                 result[i] = 0
    #     else: 
    #         for i in range(20):
    #             if(int(binary_number[i])==0):
    #                 result[i] = 0
    #             else:
    #                 result[i] = 1
        
    #     # result.append(0)    
    #     return result[0:20]


    ###############################################################################################################

    ## relu lambda func (if x>=0 return x; else zero)
    relu = lambda x : x if (x[0]==0) else [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    with open("quan_param_new_16_8/" + filename + "_bank0.dat", "w") as fb1:
        with open("quan_param_new_16_8/" + filename + "_bank1.dat", "w") as fb2:
            with open("quan_param_new_16_8/" + filename + "_bank2.dat", "w") as fb3:
                with open("quan_param_new_16_8/" + filename + "_bank3.dat", "w") as fb4:
                    for y_index in range(size_y): 
                        for x_index in range(size_x):
                            for ch in range(24): 
                                for j in range(2):
                                    for i in range(2):
                                        
                                        ### input and output need append zero
                                        if(filename=="input_image" or filename=="conv2_golden"):
                                            if(ch<3):
                                                temp_data = bindigits(int(array_total[0][ch][y_index*2+j][x_index*2+i]), 16)
                                            else: 
                                                temp_data = "0000000000000000"
                                        else:

                                            ### for adding truncated bias and relu condition
                                            if(en_relu==0):
                                                temp_data = bindigits(int(array_total[0][ch][y_index*2+j][x_index*2+i]), 16)
                                            else: ## need relu 
                                                temp_data = relu(bindigits(int(array_total[0][ch][y_index*2+j][x_index*2+i]), 16))

                                        if((x_index%2==0)&(y_index%2==0)): # orange bank
                                            if((ch==23)&(i==1)&(j==1)):
                                                fb1.write(str(temp_data[0])+str(temp_data[1])+str(temp_data[2])+str(temp_data[3])+str(temp_data[4])+str(temp_data[5])+str(temp_data[6])+str(temp_data[7])+str(temp_data[8])+str(temp_data[9])+str(temp_data[10])+str(temp_data[11])+str(temp_data[12])+str(temp_data[13])+str(temp_data[14])+str(temp_data[15])+"\n")
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
                                                fb4.write(str(temp_data[0])+str(temp_data[1])+str(temp_data[2])+str(temp_data[3])+str(temp_data[4])+str(temp_data[5])+str(temp_data[6])+str(temp_data[7])+str(temp_data[8])+str(temp_data[9])+str(temp_data[10])+str(temp_data[11])+str(temp_data[12])+str(temp_data[13])+str(temp_data[14])+str(temp_data[15]) +"_") 
                                                temp_bin = [-1]    
        
                fb4.close()
            fb3.close()
        fb2.close()
    fb1.close()
        

###############################################################################################################


def printdataindecimal(temp_output_data, filename):
    array_total = temp_output_data[:1, :24, :180,:320]


    padding = nn.ZeroPad2d(3)
    array_total = padding(array_total)
    

    array_total = array_total[:1, :24, 2:186, 2:326]


    with open(filename + "_bank0_float32.dat", "w") as fb1:
        with open(filename + "_bank1_float32.dat", "w") as fb2:
            with open(filename + "_bank2_float32.dat", "w") as fb3:
                with open(filename + "_bank3_float32.dat", "w") as fb4:
                    for y_index in range(92):
                        for x_index in range(162):
                            for ch in range(24): 
                                for j in range(2):
                                    for i in range(2):

                                        # if(ch<3):
                                        temp_data = (float(array_total[0][ch][y_index*2+j][x_index*2+i]))
                                        # else: 
                                        #     temp_data = "00000000000000000000"

                                        if((x_index%2==0)&(y_index%2==0)): # orange bank
                                            if((ch==23)&(i==1)&(j==1)):
                                                fb1.write(str(temp_data)+"\n")
                                                temp_bin = [-1]
                                            else:
                                                fb1.write(str(temp_data) +"_")
                                                temp_bin = [-1]
                                        elif((x_index%2==1)&(y_index%2==0)): # yellow bank
                                            if((ch==23)&(i==1)&(j==1)):
                                                fb2.write(str(temp_data)+ "\n")
                                                temp_bin = [-1]
                                            else:
                                                fb2.write(str(temp_data)+"_")
                                                temp_bin = [-1]

                                        elif((x_index%2==0)&(y_index%2==1)): # Blue bank
                                            if((ch==23)&(i==1)&(j==1)):
                                                fb3.write(str(temp_data)+"\n")
                                                temp_bin = [-1]

                                            else:
                                                fb3.write(str(temp_data)+"_")
                                                temp_bin = [-1]

                                        else: # Green bank 
                                            if((ch==23)&(i==1)&(j==1)):
                                                fb4.write(str(temp_data)+"\n")
                                                temp_bin = [-1]

                                            else:
                                                fb4.write(str(temp_data)+"_") 
                                                temp_bin = [-1]    
        
                fb4.close()
            fb3.close()
        fb2.close()
    fb1.close()