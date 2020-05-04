from __future__ import print_function

import torch
# version1 add the function which can generate the excel automatically 2/26/2020
#import xlsxwriter
# 
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
import numpy as np

import skimage
import skimage.io as sio
import scipy.misc
import argparse

from calc_psnr import calc_psnr


import sys
sys.setrecursionlimit(1000000)

# ########################################3/27/2020##################################################
# data_weight_res1_conv1 =  np.genfromtxt('res1_weight_truncated.dat', dtype = 'float32')

# data_weight_res1_conv1 = np.reshape(data_weight_res1_conv1, (24, 24, 3, 3))



# data_weight_conv1=torch.from_numpy(data_weight_conv1) # change numpy -> tensor 
# data_weight_res1_conv1=torch.from_numpy(data_weight_res1_conv1)


# net.state_dict()["body.0.conv1.weight"].data.copy_(data_weight_res1_conv1)

# data_bias = torch.zeros([24], dtype=torch.float32)
# net.state_dict()["body.0.conv1.bias"].data.copy_(data_bias)

##################################################################################################

############################### import .dat file( truncated weight&bias)######################################

# data_weight_conv1 = np.genfromtxt('conv1_weight_truncated.dat', dtype = 'float32')
# data_weight_res1_conv1 =  np.genfromtxt('res1_weight_truncated.dat', dtype = 'float32')

# data_weight_conv1 = np.reshape(data_weight_conv1, (24, 3, 3, 3))
# data_weight_res1_conv1 = np.reshape(data_weight_res1_conv1, (24, 24, 3, 3))
##############################################################################################################



#===== Arguments =====

# Testing settings
parser = argparse.ArgumentParser(description='NTHU EE - project - onepiece SRNet')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--combo', type=bool, required=True, help='use combo?') 
parser.add_argument('--screen_num', type=int, default=0 , help='Which screen do you want to use?') 
args = parser.parse_args()

print(args)

#################################### decimal (Integar part)-> binary funciton####################################
temp_bin = [-1]
def decimalToBinaryINT(n):
    if(len(temp_bin)==12):
        return temp_bin[:11]

    if(n%2==1):
        temp_bin.insert(0, 1)
        n = n // 2
        decimalToBinaryINT(n)
    else: 
        temp_bin.insert(0, 0)
        n = n // 2
        decimalToBinaryINT(n)

#################################################################################################################

#################################### 2's complement -> int funciton ####################################

def Twocom2INT(n):
    if(n[0]==1):
        result = - pow(2,(len(n)-1))
        for i in range(len(n)-1):
            result = result + int(n[i+1]) * pow(2, ((len(n)-2)-i))
        return result
    else:
        result = 0
        for i in range(len(n)-1):
            result = result + int(n[i+1]) * pow(2, ((len(n)-2)-i))
        return result


#################################################################################################################

## list = [0,0,0,1,0,1]
## a = Twocom2INT(list)
## a = a/32
## print(a)


#################################### decimal (小數點 part)-> binary funciton(6 digits)####################################
temp_bin = [0]
def decimalToBinary(n):
    if(len(temp_bin)==6):
        # convert negative data to 2's com
        if((temp_bin[0]==1)&(temp_bin[1]==0)&(temp_bin[2]==0)&(temp_bin[3]==0)&(temp_bin[4]==0)&(temp_bin[5]==0)):
            temp_bin[0]=0
        
        if(temp_bin[0]==1):

            # out setting
            # temp_bin[1]=1
            # temp_bin[2]=1

            temp_float = 32-(temp_bin[1] * 16 + temp_bin[2] * 8 + temp_bin[3] * 4 + temp_bin[4] * 2 + temp_bin[5] )

            for i in range(1,6):
                if(temp_float%2==1):
                    temp_bin[6-i] = 1
                    temp_float = temp_float // 2
                else:
                    temp_bin[6-i] = 0
                    temp_float = temp_float // 2 

        return temp_bin

    if(n<0):
        temp_bin[0]=1

    if(abs(n*2)>=1):
        temp_bin.append(1)
        n = abs(2*n) -1
        decimalToBinary(n)
    else: 
        temp_bin.append(0)
        n = abs(2*n)
        decimalToBinary(n)

###################################################################################################################

if args.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if args.combo!=True: ## new
    #===== load onepieceSRNet model =====
    print('===> Loading model')

    net = torch.load(args.model)

    if args.cuda:
        net = net.cuda()


    #===== Load input image =====
    imgIn = sio.imread(args.input_image)
    imgIn = imgIn.transpose((2,0,1)).astype(float)
    imgIn = imgIn.reshape(1,imgIn.shape[0],imgIn.shape[1],imgIn.shape[2])
    imgIn = torch.Tensor(imgIn)
    

    #===== Test procedures =====
    varIn = Variable(imgIn)
    if args.cuda:
        varIn = varIn.cuda()

    prediction = net(varIn)
    prediction = prediction.data.cpu().numpy().squeeze().transpose((1,2,0))

    scipy.misc.toimage(prediction, cmin=0.0, cmax=255.0).save(args.output_filename)

    #===== Ground-truth comparison =====
    if args.compare_image is not None:
        imgTar = sio.imread(args.compare_image)
        prediction = sio.imread(args.output_filename)  # read the trancated image
        psnr = calc_psnr(prediction, imgTar, max_val=255.0)
        print('===> PSNR: %.3f dB'%(psnr))

else: 
    base = ""
    output_file_name = ""
    input_file_name = ""
    ref_file_name = ""
    base += args.model[23:28] # FXXBX

    net = torch.load(args.model)


    #########################Load the weight from .dat file and turn them into weight#########################

    ############################################################################################
    #3/21 np can store the right data, 
    #however, the torch cannot will adjust the value automatically (ex: 0.9375->0.938)
    #so I will store the changed weight here.
    ############################################################################################

    # print(torch.max(net.state_dict()["conv1.bias"]))
    # print(torch.min(net.state_dict()["conv1.bias"]))
     
    # print(torch.max(net.state_dict()["body.0.body.0.bias"]))
    # print(torch.min(net.state_dict()["body.0.body.0.bias"]))

    # print(torch.max(net.state_dict()["body.0.body.2.bias"]))
    # print(torch.min(net.state_dict()["body.0.body.2.bias"]))

    # print(torch.max(net.state_dict()["body.1.body.0.bias"]))
    # print(torch.min(net.state_dict()["body.1.body.0.bias"]))

    # print(torch.max(net.state_dict()["body.1.body.2.bias"]))
    # print(torch.min(net.state_dict()["body.1.body.2.bias"]))

    # print(torch.max(net.state_dict()["body.2.body.0.bias"]))
    # print(torch.min(net.state_dict()["body.2.body.0.bias"]))

    # print(torch.max(net.state_dict()["body.2.body.2.bias"]))
    # print(torch.min(net.state_dict()["body.2.body.2.bias"]))

    # print(torch.max(net.state_dict()["body.3.body.0.bias"]))
    # print(torch.min(net.state_dict()["body.3.body.0.bias"]))

    # print(torch.max(net.state_dict()["body.3.body.2.bias"]))
    # print(torch.min(net.state_dict()["body.3.body.2.bias"]))

    # print(torch.max(net.state_dict()["upsamp1.conv.bias"]))
    # print(torch.min(net.state_dict()["upsamp1.conv.bias"]))

    # print(torch.max(net.state_dict()["upsamp2.conv.bias"]))
    # print(torch.min(net.state_dict()["upsamp2.conv.bias"]))

    # print(torch.max(net.state_dict()["conv2.bias"]))
    # print(torch.min(net.state_dict()["conv2.bias"]))


    # data_weight_conv1=torch.from_numpy(data_weight_conv1) # change numpy -> tensor 
    # data_weight_res1_conv1=torch.from_numpy(data_weight_res1_conv1)
    


    # net.state_dict()["conv1.weight"].data.copy_(data_weight_conv1)
    # net.state_dict()["body.0.body.0.weight"].data.copy_(data_weight_res1_conv1)


    ############################################################################################
    # change the bias to zero tensor
    # Because I don't divide the weight for the conv1, so I will divided the result.
    ############################################################################################

    # data_bias = torch.zeros([24], dtype=torch.float32)

    # net.state_dict()["conv1.bias"].data.copy_(data_bias)

    # net.state_dict()["body.0.body.0.bias"].data.copy_(data_bias)

    #####################################debug#############################################
    ## print(data[0][0][0][1])
    ## data=data.type(torch.DoubleTensor)
    ## print(data.dtype)
    ## print(net.state_dict()["conv1.weight"].dtype)
    ##########################################################################################################


    list_data_PSNR = []

    if args.cuda:
        net = net.cuda()

    for iter_onepiece in range(1,2):#(1,27): #1~11 : normal; 12~16 : Misplace; 17~21(index is 20~24) : hunter;
        ## the normal part
        if iter_onepiece < 10:
            output_file_name = 'result/' + str(args.screen_num) + '/' + base + '_onepiece_test_000' + str(iter_onepiece) + '.png'
            input_file_name = 'image_test/LR_onepiece_test_000' + str(iter_onepiece) + '.png'
            ref_file_name = 'ref/HR_onepiece_test_000' + str(iter_onepiece) + '.png'
        elif iter_onepiece == 10: 
            output_file_name = 'result/' + str(args.screen_num) + '/' + base + '_onepiece_test_00' + str(iter_onepiece) + '.png'
            input_file_name = 'image_test/LR_onepiece_test_00' + str(iter_onepiece) + '.png'
            ref_file_name = 'ref/HR_onepiece_test_00' + str(iter_onepiece) + '.png'
        elif iter_onepiece == 11: 
            output_file_name = 'result/' + str(args.screen_num) + '/' + base + '_onepiece_test_00' + str(iter_onepiece) + '.png'
            input_file_name = 'image_test/LR_onepiece_test_00' + str(iter_onepiece) + '.png'
            ref_file_name = 'ref/HR_onepiece_test_00' + str(iter_onepiece) + '.png'
        ## the landscape and more people part (12~16)
        elif (iter_onepiece > 11)&(iter_onepiece < 17):
            output_file_name = 'result/' + str(args.screen_num) + '/' + base + '_onepiece_test_00' + str(iter_onepiece) + '.png'
            input_file_name = 'image_test/LR_onepiece_test_00' + str(iter_onepiece) + '.png'
            ref_file_name = 'ref/HR_onepiece_test_00' + str(iter_onepiece) + '.png'
        ## the Misplace part (17~19)
        elif iter_onepiece == 17:
            output_file_name = 'result/' + str(args.screen_num) + '/' + base + '_onepiece_test_00' + str(iter_onepiece) + '.png'
            input_file_name = 'image_test/LR_onepiece_test_0008.png'
            ref_file_name = 'ref/HR_onepiece_test_0005.png'
            print('Misplaced PART: ')
        elif iter_onepiece == 18:
            output_file_name = 'result/' + str(args.screen_num) + '/' + base + '_onepiece_test_00' + str(iter_onepiece) + '.png'
            input_file_name = 'image_test/LR_onepiece_test_0005.png'
            ref_file_name = 'ref/HR_onepiece_test_0007.png'
        elif iter_onepiece == 19:
            output_file_name = 'result/' + str(args.screen_num) + '/' + base + '_onepiece_test_00' + str(iter_onepiece) + '.png'
            input_file_name = 'image_test/LR_onepiece_test_0004.png'
            ref_file_name = 'ref/HR_onepiece_test_0009.png'
        # hunter part (20~24)
        elif (iter_onepiece > 19)&(iter_onepiece < 27):
            if iter_onepiece== 20 :
                print('HUNTER PART: ')
            output_file_name = 'result/' + str(args.screen_num) + '/' + base + '_hunter_test_00' + str(iter_onepiece) + '.png'
            input_file_name = 'image_test/LR_hunter_test_00' + str(iter_onepiece) + '.png'
            ref_file_name = 'ref/HR_hunter_test_00' + str(iter_onepiece) + '.png'
        else: 
            print('There are some errors if you see this text.')

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

        ####### prepare ######
        temp1 = torch.from_numpy(temp)

        array_R = temp1[:180,:320,0]
        array_G = temp1[:180,:320,1]
        array_B = temp1[:180,:320,2]


        ###### for padding ######
        padding = nn.ZeroPad2d(1)
        array_R = padding(array_R)
        array_G = padding(array_G)
        array_B = padding(array_B)
        ########################

        ######## in order to make every bank into same size ########

        zero_tensor_col = torch.zeros([184, 2], dtype=torch.uint8)
        zero_tensor_row = torch.zeros([2, 322], dtype=torch.uint8)
        array_R = torch.cat((array_R, zero_tensor_row), 0) ## cat-> 0 for row; 1 for col
        array_R = torch.cat((array_R, zero_tensor_col), 1) ## 
        array_G = torch.cat((array_G, zero_tensor_row), 0) 
        array_G = torch.cat((array_G, zero_tensor_col), 1) 
        array_B = torch.cat((array_B, zero_tensor_row), 0) ## cat-> 0 for row; 1 for col
        array_B = torch.cat((array_B, zero_tensor_col), 1) ## 


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

        ############################# print the weight  into .dat file in binary form  #####################################
        # with open("conv1_weight.dat", "w") as f1:
        #     for i in range(24):
        #         for j in range(24): ## apend to 24 channel
        #             if(j<3):
        #                 temp1 = list(net.state_dict()["conv1.weight"][i][j].data.tolist())     
        #             else:
        #                 temp1 = [[0,0,0],[0,0,0],[0,0,0]]

        #             for k in range(3):
        #                 for l in range(3):
        #                     temp_binary = decimalToBinary(temp1[k][l])

        #                     # o.xxoooxx
                            
        #                     temp_binary = str(temp_bin[0]) + str(temp_bin[1]) + str(temp_bin[2]) + str(temp_bin[3]) + str(temp_bin[4]) + str(temp_bin[5]) 
                            
                            

        #                     # make sure the truncation is reasonalbe
        #                     # f1.write(str(temp1[k][l])+ "vs." + temp_binary + "\n")

        #                     if((j==23)&(k==2) &(l==2)):
        #                         f1.write(temp_binary)
        #                     else:
        #                         f1.write(temp_binary+"_")

        #                     # reset the global scope list
        #                     temp_bin = [0]


        #         f1.write("\n")     

        # f1.close()


        ####################################################################################################################

        ############################# print the bias  into .dat file in binary form  #################################
        # with open("conv1_bias.dat", "w") as f2:
        #     temp2 = list(net.state_dict()["conv1.bias"].data.tolist())
        #     for i in range(24):
        #         decimalToBinary(temp2[i])
        #         # make sure the truncation is reasonalbe
        #         # f2.write(str(temp2[i]) + " vs. ")
        #         temp_binary = str(temp_bin[0]) + str(temp_bin[3]) + str(temp_bin[4]) + str(temp_bin[5])
        #         f2.write(temp_binary)
        #         temp_bin = [0]
        #         f2.write("\n")
        # f2.close()
        ##############################################################################################################

        ########################## print the weight  into .dat file in truncated form  ##################################
        # with open("res1_weight_truncated.dat", "w") as f3:
        #     for i in range(24): 
        #         for j in range(3): # layer
                    
        #             temp1 = list(net.state_dict()["conv1.weight"][i][j].data.tolist())       
        #             for k in range(3):
        #                 for l in range(3):
        #                     temp_bin_par = [0,0,0,0,0,0]
        #                     decimalToBinary(temp1[k][l])
        #                     for m in range(6):
        #                         temp_bin_par[m] = temp_bin[m]
        #                     a = Twocom2INT(temp_bin_par)
        #                     f3.write(str(a) +" ")
        #                     temp_bin = [0]
                            
        #                     # reset the global scope list

        #         f3.write("\n")     

        # f3.close()
        ##############################################################################################################

        ########################## print the weight  into .dat file in float form  ##################################
        # with open("conv1_weight_float.dat", "w") as f3:
        #     for i in range(24):
        #         for j in range(3):
        #             # temp1 = list(net.state_dict()["conv1.weight"][i][j].data.tolist())  
        #             temp1 = list(net.state_dict()["conv1.weight"][i][j].data.tolist())       
        #             for k in range(3):
        #                 for l in range(3):
        #                     if(k==3 & l==3):
        #                         f3.write(str(temp1[k][l]+ " \n"))
        #                     else:
        #                         f3.write(str(temp1[k][l])+" ")
                            
        #                     # reset the global scope list
        #                     temp_bin = [0]
        #             f3.write("\n")


        #         #f3.write("\n")     
        # f3.close()
        #############################################################################################################

        ########################## print the weight  into .dat file in float form ###################################
        # with open("conv1_bias_float.dat", "w") as f4:
        #     temp2 = list(net.state_dict()["conv1.bias"].data.tolist())
        #     for i in range(24):
        #         f4.write(str(temp2[i]) +"\n")
        #         temp_bin = [0]
        # f4.close()
        ##############################################################################################################
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
        prediction = sio.imread(output_file_name)  # read the trancated image
        psnr = calc_psnr(prediction, imgTar, max_val=255.0)
        print('===> PSNR: %.3f dB'%(psnr))
        #
        list_data_PSNR.append(psnr)
        #
        





####################################save the data in the excel###########################################
# workbook = xlsxwriter.Workbook('excel/' + str(args.screen_num) + '/' + args.model[23:28] + '.xlsx') 
# worksheet = workbook.add_worksheet() 

# row = 0
# col = 0

# for iter in range(0, 24):
#     worksheet.write(row, col, list_data_PSNR[iter])
#     row +=1
#     if (iter==15)|(iter==18):
#         row +=1

# workbook.close()
#########################################################################################################