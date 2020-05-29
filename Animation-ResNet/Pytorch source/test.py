from __future__ import print_function

import torch
# version1 add the function which can generate the excel automatically 2/26/2020
import xlsxwriter
# 
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
import numpy as np

import skimage
import skimage.io as sio
import scipy.misc
import argparse
import os
from calc_psnr import calc_psnr
import sys
sys.setrecursionlimit(1000000)

## quantize ## 
## eat truncated
quantization=1

## extract para ##
bw_param=8 #8
enable=1 
# option #
# 1 for float (for pytorch)
# 0 for bin (for hardware)
print_ver=1

#################


## test info ## 16
test_num = 16
###############

#===== Arguments =====

# Testing settings
parser = argparse.ArgumentParser(description='NTHU EE - project - onepiece SRNet')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--combo', type=bool, required=True, help='use combo?') 
#
parser.add_argument('--quan_param_path', type=str, required=True, help='quan_param_path') 

#
parser.add_argument('--screen_num', type=int, default=0 , help='Which screen do you want to use?') 
args = parser.parse_args()

print(args)


######################################### turn to binary bits###################################################
def bindigits(n, bits):
    s = bin(n & int("1"*bits, 2))[2:]
    return ("{0:0>%s}" % (bits)).format(s)
#################################################################################################################


##########################################quantize_parameter####################################################
def quantize_parameter(num_group, bit_width, frational_length):
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


#################################################################################################################
if args.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

## start ##

base = ""
output_file_name = ""
input_file_name = ""
ref_file_name = ""
base += args.model[23:28] # FXXBX

net = torch.load(args.model)

    
####################### Find MAX MIN ##########################    
# for target_name, layer_param in net.state_dict().items():
#     target_new_name = target_name.split(".")

#     if(target_new_name[0][0:3]=="res" or target_new_name[0][0:6]=="upsamp"):
#         key_name = target_new_name[0] + "_" + target_new_name[1] + "_" +  target_new_name[2]
#     else:
#         key_name = target_new_name[0] + "_" + target_new_name[1]

#     int_param = layer_param.cpu().numpy()
#     num_max = torch.abs(layer_param.max())
#     num_min = torch.abs(layer_param.min())
#     print(key_name)
#     print("num_max: ", float(num_max))
#     print("num_min: ", float(num_min))




####################### Quantize the weights and biases #######################
param_fl_dict = {}

for target_name, layer_param in net.state_dict().items():
    target_new_name = target_name.split(".")
    if(target_new_name[0][0:3]=="res" or target_new_name[0][0:6]=="upsamp"):
        key_name = target_new_name[0] + "_" + target_new_name[1] + "_" +  target_new_name[2]
    else:
        key_name = target_new_name[0] + "_" + target_new_name[1]

    # origin epoch 100 version
    # if(key_name=="conv1_weight"):
    #     param_fl_dict[key_name] = bw_param+2 
    # elif(key_name=="conv1_bias"):
    #     param_fl_dict[key_name] = bw_param+2 
    # elif(key_name=="res1_conv1_weight"):
    #     param_fl_dict[key_name] = bw_param+1    
    # elif(key_name=="res1_conv2_weight"):
    #     param_fl_dict[key_name] = bw_param+1
    # elif(key_name=="res1_conv2_bias"):
    #     param_fl_dict[key_name] = bw_param+2
    # elif(key_name=="res2_conv1_weight"):  
    #     param_fl_dict[key_name] = bw_param+1   
    # elif(key_name=="res2_conv1_bias"):  
    #     param_fl_dict[key_name] = bw_param+1
    # elif(key_name=="res2_conv2_bias"):  
    #     param_fl_dict[key_name] = bw_param+3
    # elif(key_name=="res3_conv1_weight"):  
    #     param_fl_dict[key_name] = bw_param+1
    # elif(key_name=="res3_conv2_bias"):  
    #     param_fl_dict[key_name] = bw_param+3
    # elif(key_name=="res4_conv1_weight"):  
    #     param_fl_dict[key_name] = bw_param+1  
    # elif(key_name=="res4_conv1_bias"):  
    #     param_fl_dict[key_name] = bw_param+1 ## new
    # elif(key_name=="res4_conv2_weight"):  
    #     param_fl_dict[key_name] = bw_param+1 
    # elif(key_name=="res4_conv2_bias"):  
    #     param_fl_dict[key_name] = bw_param+3
    # elif(key_name=="upsamp1_conv_weight"):
    #     param_fl_dict[key_name] = bw_param+2
    # elif(key_name=="upsamp1_conv_bias"):
    #     param_fl_dict[key_name] = bw_param+3
    # elif(key_name=="upsamp2_conv_weight"):
    #     param_fl_dict[key_name] = bw_param+3    
    # elif(key_name=="upsamp2_conv_bias"):
    #     param_fl_dict[key_name] = bw_param+3   
    # elif(key_name=="conv2_weight"):
    #     param_fl_dict[key_name] = bw_param+3   #3
    # elif(key_name=="conv2_bias"):
    #     param_fl_dict[key_name] = bw_param+3   #3       
    # else: 
    #     param_fl_dict[key_name] = bw_param

    # screen_5 200epoch version 
    if(key_name=="conv1_weight"): # 2
        param_fl_dict[key_name] = bw_param+2 
    elif(key_name=="conv1_bias"):
        param_fl_dict[key_name] = bw_param+1
    elif(key_name=="res1_conv1_weight"):
        param_fl_dict[key_name] = bw_param+1    
    elif(key_name=="res1_conv2_bias"):
        param_fl_dict[key_name] = bw_param+1
    elif(key_name=="res2_conv1_weight"):  
        param_fl_dict[key_name] = bw_param+1   
    elif(key_name=="res2_conv2_bias"):  
        param_fl_dict[key_name] = bw_param+1
    elif(key_name=="res3_conv1_weight"):  
        param_fl_dict[key_name] = bw_param+1
    elif(key_name=="res3_conv2_bias"):  
        param_fl_dict[key_name] = bw_param+3 
    elif(key_name=="res4_conv2_bias"):  
        param_fl_dict[key_name] = bw_param+3
    elif(key_name=="upsamp1_conv_weight"):
        param_fl_dict[key_name] = bw_param+1
    elif(key_name=="upsamp1_conv_bias"):
        param_fl_dict[key_name] = bw_param+2
    elif(key_name=="upsamp2_conv_weight"):
        param_fl_dict[key_name] = bw_param+2   
    elif(key_name=="upsamp2_conv_bias"):
        param_fl_dict[key_name] = bw_param+3   
    elif(key_name=="conv2_weight"):
        param_fl_dict[key_name] = bw_param+3   #3
    elif(key_name=="conv2_bias"):
        param_fl_dict[key_name] = bw_param+3   #3       
    else: 
        param_fl_dict[key_name] = bw_param



##############################################################################


################################## quantization ##################################
if(enable==1 and print_ver==1): # decimal for pytorch
    if not os.path.isdir(args.quan_param_path):
        os.mkdir(args.quan_param_path)

    for target_name, layer_param in net.state_dict().items():

        target_new_name = target_name.split(".")

        if(target_new_name[0][0:3]=="res" or target_new_name[0][0:6]=="upsamp"):
            key_name = target_new_name[0] + "_" + target_new_name[1] + "_" +  target_new_name[2]
        else:
            key_name = target_new_name[0] + "_" + target_new_name[1]

        #int_param = (layer_param.cpu().numpy() * (2**param_fl_dict[key_name])).astype('int32')

        # float
        f = open(args.quan_param_path + "_deci" + '/' + key_name + ".dat", 'w')

        int_param = ((layer_param.cpu().numpy() * (2**(param_fl_dict[key_name]-1)))).astype('int')# 
        #int_param = (layer_param.cpu().numpy() ).astype('float32')##   


        # seperate the weight and bias, and reshape the weight data to 2 dimension.
        # CONV has (chin, chout, kernel_height, kernel_width)
        if(len(int_param.shape) > 2):    
            size_1 = int(int_param.shape[0]) * int(int_param.shape[1])
            size_2 = int(int_param.shape[2]) * int(int_param.shape[3])
            re_param = np.reshape(int_param, (size_1, size_2))
            re_param = re_param.astype('int32')

            print(target_name)
            print(re_param)
            for i in range(re_param.shape[0]):
                for j in range(re_param.shape[1]):
                    quanti_output = min((2**(bw_param-1)-1), re_param[i, j])
                    quanti_output = max((-1*2**(bw_param-1)), quanti_output)
                    if(j != re_param.shape[1]-1): 
                        #f.write(str(quanti_output)+' ')
                        f.write(str(re_param[i, j])+' ')
                    else: 
                        #f.write(str(quanti_output))
                        f.write(str(re_param[i, j]))
                f.write('\n')

        # FC has (neuron_in, neuron_out)
        elif(len(int_param.shape) == 2):
            print(target_name)
            print(int_param.shape)
            re_param = np.reshape(int_param, (int_param.shape[0], int_param.shape[1]))
            re_param = re_param.astype('int32')
            print(re_param)
            for i in range(int_param.shape[0]):
                for j in range(int_param.shape[1]):
                    quanti_output = min((2**(bw_param-1)-1), re_param[i, j])
                    quanti_output = max((-1*2**(bw_param-1)), quanti_output)
                    if(j != int_param.shape[1]-1): 
                        #f.write(str(quanti_output)+' ')
                        f.write(str(re_param[i, j])+' ')
                    else: 
                        #f.write(str(quanti_output))
                        f.write(str(re_param[i, j]))
                f.write('\n')

        # bias only has one dimension
        else:
            re_param = np.reshape(int_param, (int_param.shape[0]))
            re_param = re_param.astype('int32')
            print(target_name)
            print(int_param.shape)
            print(re_param)
            for i in range(int_param.shape[0]): 
                quanti_output = min((2**(bw_param-1)-1, re_param[i]))
                quanti_output = max((-1*2**(bw_param-1)), quanti_output)
                
                #f.write(str(quanti_output)+'\n')
                f.write(str(re_param[i])+'\n')
        f.close()

#############################################################################






######################################BIN##################################################
if(enable==1 and print_ver==0):
    if not os.path.isdir(args.quan_param_path):
        os.mkdir(args.quan_param_path)

    for target_name, layer_param in net.state_dict().items():

        target_new_name = target_name.split(".")

        if(target_new_name[0][0:3]=="res" or target_new_name[0][0:6]=="upsamp"):
            key_name = target_new_name[0] + "_" + target_new_name[1] + "_" +  target_new_name[2]
        else:
            key_name = target_new_name[0] + "_" + target_new_name[1]

        #int_param = (layer_param.cpu().numpy() * (2**param_fl_dict[key_name])).astype('int32')

        f = open(args.quan_param_path + '/' + key_name + ".dat", 'w')
        int_param = (layer_param.cpu().numpy() * (2**(param_fl_dict[key_name]-1))).astype('int32')


        # seperate the weight and bias, and reshape the weight data to 2 dimension.
        # CONV has (chin, chout, kernel_height, kernel_width)
        if(len(int_param.shape) > 2):    
            print(target_name)
            print(int_param.shape)
            size_1 = int(int_param.shape[0]) 
            size_2 = int(int_param.shape[1]) * int(int_param.shape[2]) * int(int_param.shape[3])

            re_param = np.reshape(int_param, (size_1, size_2))

            if(size_2<216):
                flag=True
                size_2=216
                zero_comp = "00000000" # conv1_weight need to append zeros
            else: 
                flag=False

            re_param = re_param.astype('int32')
            
            print(re_param)
            if(flag==True):
                for i in range(re_param.shape[0]):
                    for j in range(size_2):
                        if(j != size_2-1): 
                            if(j<27):
                                f.write(bindigits(re_param[i, j], bw_param)+'_')
                            else:
                                f.write(zero_comp+'_')
                        else: 
                            if(print_ver==1):
                                f.write(zero_comp)##
                            else: 
                                f.write(zero_comp)
                    f.write('\n')
            else:
                for i in range(re_param.shape[0]):
                    for j in range(216):
                        if(j != 216-1): 
                            f.write(bindigits(re_param[i, j], bw_param)+'_')
                        else: 
                            f.write(bindigits(re_param[i, j], bw_param))
                    f.write('\n')
        # FC has (neuron_in, neuron_out)
        elif(len(int_param.shape) == 2):
            print(target_name)
            print(int_param.shape)
            re_param = np.reshape(int_param, (int_param.shape[0], int_param.shape[1]))
            re_param = re_param.astype('int32')
            print(re_param)
            for i in range(int_param.shape[0]):
                for j in range(size_2):
                    if(j != size_2-1): 
                        f.write(bindigits(re_param[i, j], bw_param)+'_')
                    else: 
                        f.write(bindigits(re_param[i, j], bw_param))
                f.write('\n')
        # bias only has one dimension
        else:
            print(target_name)
            print(int_param.shape)
            re_param = np.reshape(int_param, (int_param.shape[0]))
            re_param = re_param.astype('int32')
            print(re_param)
            for i in range(int_param.shape[0]): 
                if(print_ver==1):
                    f.write(str(re_param[i])+'\n')
                else: 
                    f.write(bindigits(re_param[i], bw_param)+'\n')##
        f.close()
##########################################################################################################





##########################################################################################################
# 3/21 np can store the right data, 
# however, the torch cannot will adjust the value automatically (ex: 0.9375->0.938)
# so I will store the changed weight here.
##########################################################################################################
if(quantization==1):
    ## conv1 ##
    data_weight_conv1 = np.genfromtxt('quan_param_path_new_deci/conv1_weight.dat', dtype = 'float32')
    data_weight_conv1 = np.reshape(data_weight_conv1, (24, 3, 3, 3))
    data_weight_conv1 = data_weight_conv1/pow(2, param_fl_dict["conv1_weight"]-1)
    data_weight_conv1 = torch.from_numpy(data_weight_conv1) 
    net.state_dict()["conv1.weight"].data.copy_(data_weight_conv1)

    #bias
    data_bias_conv1 = np.genfromtxt('quan_param_path_new_deci/conv1_bias.dat', dtype = 'float32')
    data_bias_conv1 = np.reshape(data_bias_conv1, (24))
    data_bias_conv1 = data_bias_conv1/pow(2, param_fl_dict["conv1_bias"]-1)
    data_bias_conv1 = torch.from_numpy(data_bias_conv1) 
    net.state_dict()["conv1.bias"].data.copy_(data_bias_conv1)


    #  res1_conv1 trancated data(hardware)

    # weight
    data_weight_res1_conv1 = np.genfromtxt('quan_param_path_new_deci/res1_conv1_weight.dat', dtype = 'float32')
    data_weight_res1_conv1 = np.reshape(data_weight_res1_conv1, (24, 24, 3, 3))
    data_weight_res1_conv1 = data_weight_res1_conv1/pow(2, param_fl_dict["res1_conv1_weight"]-1)
    data_weight_res1_conv1 = torch.from_numpy(data_weight_res1_conv1) # change numpy -> tensor 
    net.state_dict()["res1.conv1.weight"].data.copy_(data_weight_res1_conv1)

    # bias
    data_bias_res1_conv1 = np.genfromtxt('quan_param_path_new_deci/res1_conv1_bias.dat', dtype = 'float32')
    data_bias_res1_conv1 = np.reshape(data_bias_res1_conv1, (24))
    data_bias_res1_conv1 = data_bias_res1_conv1/pow(2, param_fl_dict["res1_conv1_bias"]-1)
    data_bias_res1_conv1 = torch.from_numpy(data_bias_res1_conv1) 
    net.state_dict()["res1.conv1.bias"].data.copy_(data_bias_res1_conv1)

    #  res1_conv2 trancated data(hardware)

    # weight
    data_weight_res1_conv2 = np.genfromtxt('quan_param_path_new_deci/res1_conv2_weight.dat', dtype = 'float32')
    data_weight_res1_conv2 = np.reshape(data_weight_res1_conv2, (24, 24, 3, 3))
    data_weight_res1_conv2 = data_weight_res1_conv2/pow(2, param_fl_dict["res1_conv2_weight"]-1)
    data_weight_res1_conv2 = torch.from_numpy(data_weight_res1_conv2) # change numpy -> tensor 
    net.state_dict()["res1.conv2.weight"].data.copy_(data_weight_res1_conv2)

    # bias
    data_bias_res1_conv2 = np.genfromtxt('quan_param_path_new_deci/res1_conv2_bias.dat', dtype = 'float32')
    data_bias_res1_conv2 = np.reshape(data_bias_res1_conv2, (24))
    data_bias_res1_conv2 = data_bias_res1_conv2/pow(2, param_fl_dict["res1_conv2_bias"]-1)
    data_bias_res1_conv2 = torch.from_numpy(data_bias_res1_conv2) 
    net.state_dict()["res1.conv2.bias"].data.copy_(data_bias_res1_conv2)

    # # res2_conv1 trancated data(hardware)

    # weight
    data_weight_res2_conv1 = np.genfromtxt('quan_param_path_new_deci/res2_conv1_weight.dat', dtype = 'float32')
    data_weight_res2_conv1 = np.reshape(data_weight_res2_conv1, (24, 24, 3, 3))
    data_weight_res2_conv1 = data_weight_res2_conv1/pow(2, param_fl_dict["res2_conv1_weight"]-1)
    data_weight_res2_conv1 = torch.from_numpy(data_weight_res2_conv1) # change numpy -> tensor 
    net.state_dict()["res2.conv1.weight"].data.copy_(data_weight_res2_conv1)

    # bias
    data_bias_res2_conv1 = np.genfromtxt('quan_param_path_new_deci/res2_conv1_bias.dat', dtype = 'float32')
    data_bias_res2_conv1 = np.reshape(data_bias_res2_conv1, (24))
    data_bias_res2_conv1 = data_bias_res2_conv1/pow(2, param_fl_dict["res2_conv1_bias"]-1)
    data_bias_res2_conv1 = torch.from_numpy(data_bias_res2_conv1) 
    net.state_dict()["res2.conv1.bias"].data.copy_(data_bias_res2_conv1)

    # res2_conv2 trancated data(hardware)

    # weight
    data_weight_res2_conv2 = np.genfromtxt('quan_param_path_new_deci/res2_conv2_weight.dat', dtype = 'float32')
    data_weight_res2_conv2 = np.reshape(data_weight_res2_conv2, (24, 24, 3, 3))
    data_weight_res2_conv2 = data_weight_res2_conv2/pow(2, param_fl_dict["res2_conv2_weight"]-1)
    data_weight_res2_conv2 = torch.from_numpy(data_weight_res2_conv2) # change numpy -> tensor 
    net.state_dict()["res2.conv2.weight"].data.copy_(data_weight_res2_conv2)

    # bias
    data_bias_res2_conv2 = np.genfromtxt('quan_param_path_new_deci/res2_conv2_bias.dat', dtype = 'float32')
    data_bias_res2_conv2 = np.reshape(data_bias_res2_conv2, (24))
    data_bias_res2_conv2 = data_bias_res2_conv2/pow(2, param_fl_dict["res2_conv2_bias"]-1)
    data_bias_res2_conv2 = torch.from_numpy(data_bias_res2_conv2) 
    net.state_dict()["res2.conv2.bias"].data.copy_(data_bias_res2_conv2)


    # res3_conv1 trancated data(hardware)

    # weight
    data_weight_res3_conv1 = np.genfromtxt('quan_param_path_new_deci/res3_conv1_weight.dat', dtype = 'float32')
    data_weight_res3_conv1 = np.reshape(data_weight_res3_conv1, (24, 24, 3, 3))
    data_weight_res3_conv1 = data_weight_res3_conv1/pow(2, param_fl_dict["res3_conv1_weight"]-1)
    data_weight_res3_conv1 = torch.from_numpy(data_weight_res3_conv1) # change numpy -> tensor 
    net.state_dict()["res3.conv1.weight"].data.copy_(data_weight_res3_conv1)

    # bias
    data_bias_res3_conv1 = np.genfromtxt('quan_param_path_new_deci/res3_conv1_bias.dat', dtype = 'float32')
    data_bias_res3_conv1 = np.reshape(data_bias_res3_conv1, (24))
    data_bias_res3_conv1 = data_bias_res3_conv1/pow(2, param_fl_dict["res3_conv1_bias"]-1)
    data_bias_res3_conv1 = torch.from_numpy(data_bias_res3_conv1) 
    net.state_dict()["res3.conv1.bias"].data.copy_(data_bias_res3_conv1)

    # res3_conv2 trancated data(hardware)

    # weight
    data_weight_res3_conv2 = np.genfromtxt('quan_param_path_new_deci/res3_conv2_weight.dat', dtype = 'float32')
    data_weight_res3_conv2 = np.reshape(data_weight_res3_conv2, (24, 24, 3, 3))
    data_weight_res3_conv2 = data_weight_res3_conv2/pow(2, param_fl_dict["res3_conv2_weight"]-1)
    data_weight_res3_conv2 = torch.from_numpy(data_weight_res3_conv2) # change numpy -> tensor 
    net.state_dict()["res3.conv2.weight"].data.copy_(data_weight_res3_conv2)

    # bias
    data_bias_res3_conv2 = np.genfromtxt('quan_param_path_new_deci/res3_conv2_bias.dat', dtype = 'float32')
    data_bias_res3_conv2 = np.reshape(data_bias_res3_conv2, (24))
    data_bias_res3_conv2 = data_bias_res3_conv2/pow(2, param_fl_dict["res3_conv2_bias"]-1)
    data_bias_res3_conv2 = torch.from_numpy(data_bias_res3_conv2) 
    net.state_dict()["res3.conv2.bias"].data.copy_(data_bias_res3_conv2)


    # res4_conv1 trancated data(hardware)

    # weight
    data_weight_res4_conv1 = np.genfromtxt('quan_param_path_new_deci/res4_conv1_weight.dat', dtype = 'float32')
    data_weight_res4_conv1 = np.reshape(data_weight_res4_conv1, (24, 24, 3, 3))
    data_weight_res4_conv1 = data_weight_res4_conv1/pow(2, param_fl_dict["res4_conv1_weight"]-1)
    data_weight_res4_conv1 = torch.from_numpy(data_weight_res4_conv1) # change numpy -> tensor 
    net.state_dict()["res4.conv1.weight"].data.copy_(data_weight_res4_conv1)

    # bias
    data_bias_res4_conv1 = np.genfromtxt('quan_param_path_new_deci/res4_conv1_bias.dat', dtype = 'float32')
    data_bias_res4_conv1 = np.reshape(data_bias_res4_conv1, (24))
    data_bias_res4_conv1 = data_bias_res4_conv1/pow(2, param_fl_dict["res4_conv1_bias"]-1)
    data_bias_res4_conv1 = torch.from_numpy(data_bias_res4_conv1) 
    net.state_dict()["res4.conv1.bias"].data.copy_(data_bias_res4_conv1)

    # res4_conv2 trancated data(hardware)

    # weight
    data_weight_res4_conv2 = np.genfromtxt('quan_param_path_new_deci/res4_conv2_weight.dat', dtype = 'float32')
    data_weight_res4_conv2 = np.reshape(data_weight_res4_conv2, (24, 24, 3, 3))
    data_weight_res4_conv2 = data_weight_res4_conv2/pow(2, param_fl_dict["res4_conv2_weight"]-1)
    data_weight_res4_conv2 = torch.from_numpy(data_weight_res4_conv2) # change numpy -> tensor 
    net.state_dict()["res4.conv2.weight"].data.copy_(data_weight_res4_conv2)

    # bias
    data_bias_res4_conv2 = np.genfromtxt('quan_param_path_new_deci/res4_conv2_bias.dat', dtype = 'float32')
    data_bias_res4_conv2 = np.reshape(data_bias_res4_conv2, (24))
    data_bias_res4_conv2 = data_bias_res4_conv2/pow(2, param_fl_dict["res4_conv2_bias"]-1)
    data_bias_res4_conv2 = torch.from_numpy(data_bias_res4_conv2) 
    net.state_dict()["res4.conv2.bias"].data.copy_(data_bias_res4_conv2)


    # upsample1_conv1 trancated data(hardware)

    # weight
    data_weight_upsample1_conv = np.genfromtxt('quan_param_path_new_deci/upsamp1_conv_weight.dat',dtype = 'float32')
    data_weight_upsample1_conv = np.reshape(data_weight_upsample1_conv, (96, 24, 3, 3))
    data_weight_upsample1_conv = data_weight_upsample1_conv/pow(2, param_fl_dict["upsamp1_conv_weight"]-1)
    data_weight_upsample1_conv = torch.from_numpy(data_weight_upsample1_conv)
    net.state_dict()["upsamp1.conv.weight"].data.copy_(data_weight_upsample1_conv)

    #bias 

    data_bias_upsample1_conv = np.genfromtxt('quan_param_path_new_deci/upsamp1_conv_bias.dat',dtype = 'float32')
    data_bias_upsample1_conv = np.reshape(data_bias_upsample1_conv, (96))    
    data_bias_upsample1_conv = data_bias_upsample1_conv/pow(2, param_fl_dict["upsamp1_conv_bias"]-1)
    data_bias_upsample1_conv = torch.from_numpy(data_bias_upsample1_conv) 
    net.state_dict()["upsamp1.conv.bias"].data.copy_(data_bias_upsample1_conv)


    # upsample2_conv trancated data(hardware)

    # weight
    data_weight_upsample2_conv = np.genfromtxt('quan_param_path_new_deci/upsamp2_conv_weight.dat',dtype = 'float32')
    data_weight_upsample2_conv = np.reshape(data_weight_upsample2_conv, (96, 24, 3, 3))
    data_weight_upsample2_conv = data_weight_upsample2_conv/pow(2, param_fl_dict["upsamp2_conv_weight"]-1)
    data_weight_upsample2_conv = torch.from_numpy(data_weight_upsample2_conv)
    net.state_dict()["upsamp2.conv.weight"].data.copy_(data_weight_upsample2_conv)

    # bias 

    data_bias_upsample2_conv = np.genfromtxt('quan_param_path_new_deci/upsamp2_conv_bias.dat',dtype = 'float32')
    data_bias_upsample2_conv = np.reshape(data_bias_upsample2_conv, (96))    
    data_bias_upsample2_conv = data_bias_upsample2_conv/pow(2, param_fl_dict["upsamp2_conv_bias"]-1) 
    data_bias_upsample2_conv = torch.from_numpy(data_bias_upsample2_conv) 
    net.state_dict()["upsamp2.conv.bias"].data.copy_(data_bias_upsample2_conv)

    # conv2 trancated data (hardware)

    # weight
    data_weight_conv2 = np.genfromtxt('quan_param_path_new_deci/conv2_weight.dat', dtype = 'float32')
    data_weight_conv2 = np.reshape(data_weight_conv2, (3, 24, 3, 3))
    data_weight_conv2 = data_weight_conv2/pow(2, param_fl_dict["conv2_weight"]-1) # borrow one place for activation #+1
    data_weight_conv2 = torch.from_numpy(data_weight_conv2) # change numpy -> tensor 
    net.state_dict()["conv2.weight"].data.copy_(data_weight_conv2)

    # bias
    data_bias_conv2 = np.genfromtxt('quan_param_path_new_deci/conv2_bias.dat', dtype = 'float32')
    data_bias_conv2 = np.reshape(data_bias_conv2, (3))
    data_bias_conv2 = data_bias_conv2/pow(2, param_fl_dict["conv2_bias"]-1) # borrow one place for activation #+1
    data_bias_conv2 = torch.from_numpy(data_bias_conv2) 
    net.state_dict()["conv2.bias"].data.copy_(data_bias_conv2)

    #########################################################################################################

list_data_PSNR = []

if args.cuda:
    net = net.cuda()

for iter_onepiece in range(1,test_num+1):#(1,27): #1~11 : normal; 12~16 : Misplace; 17~21(index is 20~24) : hunter;
    ## the normal part
    if iter_onepiece < 10:
        output_file_name = 'result/' + str(args.screen_num) + '/' + base + '_onepiece_test_000' + str(iter_onepiece) + '.png'
        input_file_name = 'image_test/LR_onepiece_test_000' + str(iter_onepiece) + '.png'
        compare_file_name = 'image_compare/LR_onepiece_test_000' + str(iter_onepiece) + '.png'
        # input_file_name = 'image_test/temp01.png'
        ref_file_name = 'ref/HR_onepiece_test_000' + str(iter_onepiece) + '.png'
    elif iter_onepiece == 10: 
        output_file_name = 'result/' + str(args.screen_num) + '/' + base + '_onepiece_test_00' + str(iter_onepiece) + '.png'
        input_file_name = 'image_test/LR_onepiece_test_00' + str(iter_onepiece) + '.png'
        compare_file_name = 'image_compare/LR_onepiece_test_00' + str(iter_onepiece) + '.png'
        ref_file_name = 'ref/HR_onepiece_test_00' + str(iter_onepiece) + '.png'
    elif iter_onepiece == 11: 
        output_file_name = 'result/' + str(args.screen_num) + '/' + base + '_onepiece_test_00' + str(iter_onepiece) + '.png'
        input_file_name = 'image_test/LR_onepiece_test_00' + str(iter_onepiece) + '.png'
        compare_file_name = 'image_compare/LR_onepiece_test_00' + str(iter_onepiece) + '.png'
        ref_file_name = 'ref/HR_onepiece_test_00' + str(iter_onepiece) + '.png'
    elif (iter_onepiece > 11)&(iter_onepiece < 100):
        output_file_name = 'result/' + str(args.screen_num) + '/' + base + '_onepiece_test_00' + str(iter_onepiece) + '.png'
        input_file_name = 'image_test/LR_onepiece_test_00' + str(iter_onepiece) + '.png'
        compare_file_name = 'image_compare/LR_onepiece_test_00' + str(iter_onepiece) + '.png'
        ref_file_name = 'ref/HR_onepiece_test_00' + str(iter_onepiece) + '.png'
    else: 
        output_file_name = 'result/' + str(args.screen_num) + '/' + base + '_onepiece_test_0' + str(iter_onepiece) + '.png'
        input_file_name = 'image_test/LR_onepiece_test_00' + str(iter_onepiece) + '.png'
        compare_file_name = 'image_compare/LR_onepiece_test_00' + str(iter_onepiece) + '.png'
        ref_file_name = 'ref/HR_onepiece_test_00' + str(iter_onepiece) + '.png'

    #===== load onepieceSRNet model =====
    print('===> Loading model on iter ' + str(iter_onepiece))

    
    #===== Load input image =====
    temp = []

    imgIn = sio.imread(input_file_name)
    temp = imgIn


    #########################################################################################################################
    #                                            SAVE    DATA                                                               #
    #########################################################################################################################

    #####################################divided by bank###############################################

    # ###### prepare ######
    # temp1 = torch.from_numpy(temp)

    # array_R = temp1[:180,:320,0]
    # array_G = temp1[:180,:320,1]
    # array_B = temp1[:180,:320,2]


    # # ###### for padding ######
    # padding = nn.ZeroPad2d(1)
    # array_R = padding(array_R)
    # array_G = padding(array_G)
    # array_B = padding(array_B)
    # ########################

    # ######## in order to make every bank into same size ########

    # zero_tensor_col = torch.zeros([184, 2], dtype=torch.uint8)
    # zero_tensor_row = torch.zeros([2, 322], dtype=torch.uint8)
    # array_R = torch.cat((array_R, zero_tensor_row), 0) ## cat-> 0 for row; 1 for col
    # array_R = torch.cat((array_R, zero_tensor_col), 1) ## 
    # array_G = torch.cat((array_G, zero_tensor_row), 0) 
    # array_G = torch.cat((array_G, zero_tensor_col), 1) 
    # array_B = torch.cat((array_B, zero_tensor_row), 0) ## cat-> 0 for row; 1 for col
    # array_B = torch.cat((array_B, zero_tensor_col), 1) ## 


    ##################################input write############################################

    # with open("input_bank0.dat", "w") as fb1:
    #     with open("input_bank1.dat", "w") as fb2:
    #         with open("input_bank2.dat", "w") as fb3:
    #             with open("input_bank3.dat", "w") as fb4:
    #                 for y_index in range(92):
    #                     for x_index in range(162):
    #                         for ch in range(24): ## after ch3  temp=0
    #                             for j in range(2):
    #                                 for i in range(2):
    #                                     if(ch==0): # R
    #                                         decimalToBinaryINT(int(array_R[y_index*2+j][x_index*2+i]))
    #                                     elif(ch==1): # G
    #                                         decimalToBinaryINT(int(array_G[y_index*2+j][x_index*2+i]))
    #                                     elif(ch==2): #B
    #                                         decimalToBinaryINT(int(array_B[y_index*2+j][x_index*2+i]))
    #                                     else: # zero condition
    #                                         temp_bin = [0,0,0,0,0,0,0,0,0,0,0]

    #                                     if((x_index%2==0)&(y_index%2==0)): # orange bank
    #                                         if((ch==23)&(i==1)&(j==1)):
    #                                             fb1.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) +"\n")
    #                                             temp_bin = [-1]
    #                                         else:
    #                                             fb1.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) +"_")
    #                                             temp_bin = [-1]
    #                                     elif((x_index%2==1)&(y_index%2==0)): # yellow bank
    #                                         if((ch==23)&(i==1)&(j==1)):
    #                                             fb2.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) + "\n")
    #                                             temp_bin = [-1]
    #                                         else:
    #                                             fb2.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) +"_")
    #                                             temp_bin = [-1]
    #                                     elif((x_index%2==0)&(y_index%2==1)): # Blue bank
    #                                         if((ch==23)&(i==1)&(j==1)):
    #                                             fb3.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) +"\n")
    #                                             temp_bin = [-1]
    #                                         else:
    #                                             fb3.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) +"_")
    #                                             temp_bin = [-1]
    #                                     else: # Green bank 
    #                                         if((ch==23)&(i==1)&(j==1)):
    #                                             fb4.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) +"\n")
    #                                             temp_bin = [-1]
    #                                         else:
    #                                             fb4.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) +"_") 
    #                                             temp_bin = [-1]               
    #             fb4.close()
    #         fb3.close()
    #     fb2.close()
    # fb1.close()
    ###################################################################################################


    ##########################################write conv1_output################################################
    # with open("input_bank0.dat", "w") as fb1:
    #     with open("input_bank1.dat", "w") as fb2:
    #         with open("input_bank2.dat", "w") as fb3:
    #             with open("input_bank3.dat", "w") as fb4:
    #                 for y_index in range(92):
    #                     for x_index in range(162):
    #                         for ch in range(24): ## after ch3  temp=0
    #                             for j in range(2):
    #                                 for i in range(2):
    #                                     if(ch==0): # R
    #                                         decimalToBinaryINT(int(array_R[y_index*2+j][x_index*2+i]))
    #                                     elif(ch==1): # G
    #                                         decimalToBinaryINT(int(array_G[y_index*2+j][x_index*2+i]))
    #                                     elif(ch==2): #B
    #                                         decimalToBinaryINT(int(array_B[y_index*2+j][x_index*2+i]))
    #                                     else: # zero condition
    #                                         temp_bin = [0,0,0,0,0,0,0,0,0,0,0]

    #                                     if((x_index%2==0)&(y_index%2==0)): # orange bank
    #                                         if((ch==23)&(i==1)&(j==1)):
    #                                             fb1.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) +"\n")
    #                                             temp_bin = [-1]
    #                                         else:
    #                                             fb1.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) +"_")
    #                                             temp_bin = [-1]
    #                                     elif((x_index%2==1)&(y_index%2==0)): # yellow bank
    #                                         if((ch==23)&(i==1)&(j==1)):
    #                                             fb2.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) + "\n")
    #                                             temp_bin = [-1]
    #                                         else:
    #                                             fb2.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) +"_")
    #                                             temp_bin = [-1]
    #                                     elif((x_index%2==0)&(y_index%2==1)): # Blue bank
    #                                         if((ch==23)&(i==1)&(j==1)):
    #                                             fb3.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) +"\n")
    #                                             temp_bin = [-1]
    #                                         else:
    #                                             fb3.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) +"_")
    #                                             temp_bin = [-1]
    #                                     else: # Green bank 
    #                                         if((ch==23)&(i==1)&(j==1)):
    #                                             fb4.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) +"\n")
    #                                             temp_bin = [-1]
    #                                         else:
    #                                             fb4.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) +"_") 
    #                                             temp_bin = [-1]               
    #             fb4.close()
    #         fb3.close()
    #     fb2.close()
    # fb1.close()
    

    ##########################################VARIFY###################################################
    # with open("input_bank1_var.dat", "w") as fb1:
    #     for y_index in range(91):
    #         for x_index in range(161):
    #             for ch in range(24): ## after ch3  temp=0
    #                 for j in range(2):
    #                     for i in range(2):
    #                         if(ch==0): # R
    #                             temp_var = int(array_R[y_index*2+j][x_index*2+i])
    #                         elif(ch==1): # G
    #                             temp_var = int(array_G[y_index*2+j][x_index*2+i])
    #                         elif(ch==2): #B
    #                             temp_var = int(array_B[y_index*2+j][x_index*2+i])
    #                         else: # zero condition
    #                             temp_var = 0

    #                         if((x_index%2==0)&(y_index%2==0)): # orange bank
    #                             if((ch==23)&(i==1)&(j==1)):
    #                                 fb1.write(str(temp_var) +"\n")
    #                                 #temp_bin = [-1]
    #                             else:
    #                                 fb1.write(str(temp_var) +"_")
    
    # fb1.close()

    ###################################################################################################

    #####################################with padding (322*182) 3/16###################################
    # with open("input_R.dat", "w") as fR:
    #     for i in range(182):
    #         for j in range(322):
    #             if((i==0) |(j==0)|(i==181)|(j==321)):
    #                 fR.write("00000000000" + " ")
    #             else:
    #                 decimalToBinaryINT(temp[i-1][j-1][0])
    #                 fR.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) + " ")
    #                 temp_bin = [-1]
    #             fR.write("\n")
            
    # fR.close()

    # with open("input_G.dat", "w") as fG:
    #     for i in range(182):
    #         for j in range(322):
    #             if((i==0) |(j==0)|(i==181)|(j==321)):
    #                 fG.write("00000000000" + " ")
    #             else:
    #                 decimalToBinaryINT(temp[i-1][j-1][1])
    #                 fG.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) + " ")
    #                 temp_bin = [-1]
    #             fG.write("\n")
            
    # fG.close()

    # with open("input_B.dat", "w") as fB:
    #     for i in range(182):
    #         for j in range(322):
    #             if((i==0) |(j==0)|(i==181)|(j==321)):
    #                 fB.write("00000000000" + " ")
    #             else:
    #                 decimalToBinaryINT(temp[i-1][j-1][2])
    #                 fB.write(str(temp_bin[0])+str(temp_bin[1])+str(temp_bin[2])+str(temp_bin[3])+str(temp_bin[4])+str(temp_bin[5])+str(temp_bin[6])+str(temp_bin[7])+str(temp_bin[8])+str(temp_bin[9])+str(temp_bin[10]) + " ")
    #                 temp_bin = [-1]
    #             fB.write("\n")
            
    # fB.close()
    ##############################################################################################

    ####################################No padding version########################################
    # with open("input_R.dat", "w") as fR:
    #     for i in range(180):
    #         for j in range(320):
    #             fR.write(str(temp[i][j][0]) + "_")
    #             if(j==319):
    #                 fR.write("\n")
    # fR.close()

    # with open("input_G.dat", "w") as fG:
    #     for i in range(180):
    #         for j in range(320):
    #             fG.write(str(temp[i][j][1]) + "_")
    #             if(j==319):
    #                 fG.write("\n")
    # fG.close()

    # with open("input_B.dat", "w") as fB:
    #     for i in range(180):
    #         for j in range(320):
    #             fB.write(str(temp[i][j][2]) + "_")
    #             if(j==319):
    #                 fB.write("\n")
    # fB.close()
    ##############################################################################################


    #########################################################################################################################
    #                                            SAVE    WEIGHT  & BIAS                                                     #
    #########################################################################################################################
    # select one set per time
    # kernal_property = [rows, cols] # for txt file

    # # 1 Notice that truncated need extra modification
    # list_datafilename = ["conv1"]
    # list_data_access = ["conv1"]
    # kernal_property = [24, 3] 


    # # 2
    # list_datafilename = ["res1_conv1", "res1_conv2", "res2_conv1", "res2_conv2", "res3_conv1", "res3_conv2", "res4_conv1", "res4_conv2"] #conv1&2-> need to adjust
    # list_data_access = ["res1.conv1", "res1.conv2", "res2.conv1", "res2.conv2", "res3.conv1", "res3.conv2", "res4.conv1", "res4.conv2"] #conv1&2-> need to adjust
    # kernal_property = [24, 24]

    
    # # 3
    # list_datafilename = ["upsamp1_conv", "upsamp2_conv"]
    # list_data_access = ["upsamp1.conv", "upsamp2.conv"]
    # kernal_property = [96, 24]

    # # 4
    # list_datafilename = ["conv2"]
    # list_data_access = ["conv2"]
    # kernal_property = [3,24]
            

    # print(net.state_dict()[].shape)

    ########################### print the weight  into .dat file in binary form (8 digits)#####################################
    # for d in range(len(list_datafilename)):
    #     with open((list_datafilename[d] + "_weight.dat"), "w") as f1:
    #         for i in range(kernal_property[0]):
    #             for j in range(kernal_property[1]): ## apend to 24 channel
    #                 # if(j<3):
    #                 temp1 = list(net.state_dict()[list_data_access[d]+".weight"][i][j].data.tolist())     
    #                 # else: ## conv1
    #                 #     temp1 = [[0,0,0],[0,0,0],[0,0,0]]

    #                 for k in range(3):
    #                     for l in range(3):
    #                         temp_binary = decimalToBinary(temp1[k][l], 8)

    #                         # o.ooooooo
    #                         temp_binary = str(temp_bin[0]) + str(temp_bin[1]) + str(temp_bin[2]) + str(temp_bin[3]) + str(temp_bin[4]) + str(temp_bin[5]) + str(temp_bin[6]) + str(temp_bin[7])   
                            
    #                         if((j==23)&(k==2) &(l==2)):
    #                             f1.write(temp_binary)
    #                         else:
    #                             f1.write(temp_binary+"_")

    #                         # reset the global scope list
    #                         temp_bin = [0]

    #             f1.write("\n")     
    #     f1.close()


    # # ####################################################################################################################

    # # ############################# print the bias  into .dat file in binary form  #################################
    #     with open(list_datafilename[d]+"_bias.dat", "w") as f2:
    #         temp2 = list(net.state_dict()[list_data_access[d]+".bias"].data.tolist())
    #         for i in range(kernal_property[0]):
    #             decimalToBinary(temp2[i], 8)

    #             # temp_bin = [0,0,0,0,0,0,0,0]     
    #             temp_binary = str(temp_bin[0]) + str(temp_bin[1]) + str(temp_bin[2]) + str(temp_bin[3]) + str(temp_bin[4]) + str(temp_bin[5]) + str(temp_bin[6]) + str(temp_bin[7])
    #             f2.write(temp_binary)
    #             temp_bin = [0]
    #             f2.write("\n")
    #     f2.close()
    # # ##############################################################################################################


    # ################## print the weight  into .dat file in truncated form (8bits version)##################################
    #     with open(list_datafilename[d]+"_weight_truncated.dat", "w") as ft3:
    #         for i in range(kernal_property[0]): 
    #             for j in range(kernal_property[1]): # layer

    #                 temp1 = list(net.state_dict()[list_data_access[d]+".weight"][i][j].data.tolist())     

    #                 for k in range(3):
    #                     for l in range(3):
    #                         temp_bin_par = [0,0,0,0,0,0,0,0]
    #                         decimalToBinary(temp1[k][l], 8)
    #                         for m in range(8):
    #                             temp_bin_par[m] = temp_bin[m]
    #                         a = Twocom2INT(temp_bin_par)
    #                         ft3.write(str(a) +" ")
    #                         temp_bin = [0]
                            
    #                         #reset the global scope list

    #             ft3.write("\n")     

    #     ft3.close()
    # #############################################################################################################

    # ########################## print the weight  into .dat file in float form  ##################################
    #     with open(list_datafilename[d]+"_weight_float.dat", "w") as f3:
    #         for i in range(kernal_property[0]):
    #             for j in range(kernal_property[1]): 
    #                 # if(j<3):
    #                 temp1 = list(net.state_dict()[list_data_access[d]+".weight"][i][j].data.tolist())     
    #                 # else: ## conv1
    #                 #     temp1 = [[0,0,0],[0,0,0],[0,0,0]]

    #                 #temp1 = list(net.state_dict()[list_data_access[d]+".weight"][i][j].data.tolist())       
    #                 for k in range(3):
    #                     for l in range(3):
    #                         if(k==3 & l==3):
    #                             f3.write(str(temp1[k][l]+ " \n"))
    #                         else:
    #                             f3.write(str(temp1[k][l])+" ")
                            
    #                         # reset the global scope list
    #                         temp_bin = [0]
    #                 f3.write("\n")


                
    #     f3.close()
    # # ############################################################################################################

    # ######################### print the bias  into .dat file in float form ###################################
    #     with open(list_datafilename[d]+"_bias_float.dat", "w") as f4:
    #         temp2 = list(net.state_dict()[list_data_access[d]+".bias"].data.tolist())
    #         for i in range(kernal_property[0]):
    #             f4.write(str(temp2[i]) +"\n")
    #             temp_bin = [0]
    #     f4.close()
    # ##############################################################################################################


    # ############################# print the bias_truncated ############### #################################
    #     with open(list_datafilename[d]+"_bias_truncated.dat", "w") as ff2:
    #         temp1 = list(net.state_dict()[list_data_access[d]+".bias"].data.tolist())
    #         for i in range(kernal_property[0]):
    #             temp_bin_par = [0,0,0,0,0,0,0,0]    
    #             decimalToBinary(temp1[i], 8) 
    #             for m in range(8):
    #                 temp_bin_par[m] = temp_bin[m]
    #             a = Twocom2INT(temp_bin_par)
                
    #             ff2.write(str(a) + " ")
    #             temp_bin = [0]
    #             ff2.write("\n")
    #     ff2.close()
    ##########################################################################################################

    imgIn = imgIn.transpose((2,0,1)).astype(float)
    imgIn = imgIn.reshape(1,imgIn.shape[0],imgIn.shape[1],imgIn.shape[2])
    imgIn = torch.Tensor(imgIn)
    

    #===== Test procedures =====
    varIn = Variable(imgIn)
    if args.cuda:
        varIn = varIn.cuda()

    prediction = net(varIn)
    prediction = prediction.data.cpu().numpy().squeeze().transpose((1,2,0))

    scipy.misc.toimage(prediction, cmin=0.0, cmax=255.0).save(output_file_name)

    #===== Ground-truth comparison =====
    #if args.compare_image is not None:
    imgTar = sio.imread(ref_file_name)
    #prediction = sio.imread(output_file_name)  # read the trancated image
    prediction = sio.imread(compare_file_name)  # read the trancated image
    psnr = calc_psnr(prediction, imgTar, max_val=255.0)
    print('===> PSNR: %.3f dB'%(psnr))
    
    #
    list_data_PSNR.append(psnr)


   # print(net.state_dict()["conv1.bias"])
    # print(net.state_dict()["res1.conv1.weight"].size())
    # print(net.state_dict()["res1.conv2.weight"].size())
    # print(net.state_dict()["upsamp1.conv.weight"].size())
    # print(net.state_dict()["upsamp2.conv.weight"].size())
    # print(net.state_dict()["conv2.weight"].size())
    # print("conv1.weight: ")
    # print(net.state_dict()["conv1.weight"])

####################################save the data in the excel###########################################

workbook = xlsxwriter.Workbook('excel/' + 'PSNR' + '.xlsx') 
worksheet = workbook.add_worksheet() 

row = 0
col = 0
aver=0
for iter in range(0, test_num):
    worksheet.write(row, col, list_data_PSNR[iter])
    row +=1
    aver += list_data_PSNR[iter]
print(aver/test_num)
workbook.close()
#########################################################################################################
