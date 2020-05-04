from __future__ import print_function

import torch
# version1 add the function which can generate the excel automatically 2/26/2020
import xlsxwriter
# 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import skimage
import skimage.io as sio
import scipy.misc
import argparse

from calc_psnr import calc_psnr

#===== Arguments =====

# Testing settings
parser = argparse.ArgumentParser(description='NTHU EE - project - onepiece SRNet')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--combo', type=bool, required=True, help='use combo?') 
args = parser.parse_args()

print(args)

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

    list_data_PSNR = []

    if args.cuda:
        net = net.cuda()

    for iter_onepiece in range(1,4): #1~11 : normal; 12~16 : Misplace; 17~21(index is 20~24) : hunter;
        ## the normal part
        if iter_onepiece < 10:
            output_file_name = 'result_spec/'+ 'x' +  '/' + base + '_onepiece_test_000' + str(iter_onepiece) + '.jpg'
            input_file_name = 'image_test_spec/LR_onepiece_test_000' + str(iter_onepiece) + '.jpg'
            ref_file_name = 'ref_spec/HR_onepiece_test_000' + str(iter_onepiece) + '.jpg'
        else: 
            print('There are some errors if you see this text.')

        #===== load onepieceSRNet model =====
        print('===> Loading model on iter ' + str(iter_onepiece))
        
        
        #===== Load input image =====

        imgIn = sio.imread(input_file_name)
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
        # #if args.compare_image is not None:
        # imgTar = sio.imread(ref_file_name)
        # prediction = sio.imread(output_file_name)  # read the trancated image
        # psnr = calc_psnr(prediction, imgTar, max_val=255.0)
        # #print('===> PSNR: %.3f dB'%(psnr))
        # #
        # list_data_PSNR.append(psnr)
        # #

 #===== save the data in the excel ====
# workbook = xlsxwriter.Workbook('excel/' + 'x' + '/' + args.model[23:28] + '.xlsx') 
# worksheet = workbook.add_worksheet() 

# row = 0
# col = 0

# for iter in range(0, 24):
#     worksheet.write(row, col, list_data_PSNR[iter])
#     row +=1
#     if (iter==15)|(iter==18):
#         row +=1

# workbook.close()
    