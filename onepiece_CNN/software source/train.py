from __future__ import print_function

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse

from model import onepieceSRNet
from dataset_train import datasetTrain
from dataset_val import datasetVal

# import sys
# sys.setrecursionlimit(1000000)

# temp_bin = [0]
#===== Arguments =====

# Training settings
parser = argparse.ArgumentParser(description='NTHU EE - CP HW3 - onepieceSRNet')
parser.add_argument('--patchSize', type=int, default=128, help='HR image cropping (patch) size for training')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--epochSize', type=int, default=250, help='number of batches as one epoch (for validating once)')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--nFeat', type=int, default=16, help='channel number of feature maps')
parser.add_argument('--nResBlock', type=int, default=2, help='number of residual blocks')
parser.add_argument('--nTrain', type=int, default=2, help='number of training images')
parser.add_argument('--nVal', type=int, default=1, help='number of validation images')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use, if Your OS is window, please set to 0')
parser.add_argument('--seed', type=int, default=715, help='random seed to use. Default=715')
parser.add_argument('--printEvery', type=int, default=50, help='number of batches to print average loss ')
parser.add_argument('--screen_num', type=int, default=0, help='Which screen do you want to use?') ##new
args = parser.parse_args()

print(args)

if args.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#===== Datasets =====
print('===> Loading datasets')
train_set = datasetTrain(args)
train_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)
val_set = datasetVal(args)
val_data_loader = DataLoader(dataset=val_set, num_workers=args.threads, batch_size=1, shuffle=False)

#===== onepieceSRNet model =====
print('===> Building model')
net = onepieceSRNet(nFeat=args.nFeat, nResBlock=args.nResBlock)

if args.cuda:
    net = net.cuda()

#===== Loss function and optimizer =====
criterion = torch.nn.L1Loss()

if args.cuda:
    criterion = criterion.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

#===== Training and validation procedures =====
def train(epoch):
    net.train()
    epoch_loss = 0
    for iteration, batch in enumerate(train_data_loader):
        varIn, varTar = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            varIn = varIn.cuda()
            varTar = varTar.cuda()

        optimizer.zero_grad()
        loss = criterion(net(varIn), varTar)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        if (iteration+1)%args.printEvery == 0:
            print("===> Epoch[{}]({}/{}): Avg. Loss: {:.4f}".format(epoch, iteration+1, len(train_data_loader), epoch_loss/args.printEvery))
            epoch_loss = 0
        


from math import log10

def validate():
    net.eval()
    avg_psnr = 0
    mse_criterion = torch.nn.MSELoss()
    for batch in val_data_loader:
        varIn, varTar = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            varIn = varIn.cuda()
            varTar = varTar.cuda()

        prediction = net(varIn)
        mse = mse_criterion(prediction, varTar)
        psnr = 10 * log10(255*255/mse.data)
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(val_data_loader)))


def checkpoint(epoch):
    model_out_path = "model_pretrained/" + str(args.screen_num) + "/net_F{}B{}_epoch_{}.pth".format(args.nFeat, args.nResBlock, epoch)
    torch.save(net, model_out_path) # save all net
    print("Checkpoint saved to {}".format(model_out_path))



#===== Main procedure =====
for epoch in range(1, args.nEpochs + 1):
    train(epoch)
    validate()
    checkpoint(epoch)


############################for .data########################
# def decimalToBinary(n):
#     if(len(temp_bin)==6):
#         # convert negative data to 2's com
#         if((temp_bin[0]==1) &(temp_bin[3]==0)&(temp_bin[4]==0)&(temp_bin[5]==0)):
#             temp_bin[0]=0
        
#         if(temp_bin[0]==1):
#             # out setting
#             temp_bin[1]=1
#             temp_bin[2]=1

#             temp_float = 8-(temp_bin[3] * 4 + temp_bin[4] * 2 + temp_bin[5] )

#             for i in range(1,4):
#                 if(temp_float%2==1):
#                     temp_bin[6-i] = 1
#                     temp_float = temp_float // 2
#                 else:
#                     temp_bin[6-i] = 0
#                     temp_float = temp_float // 2 

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
################################################################
    
    


    ## save data


    # temp1 = [[],[],[]] # for conv1's weight
    # temp2 = [] # for conv1's bias
    



    ############################# print the weight  into .dat file in binary form  #############################
    # with open("conv1_weight.dat", "w") as f1:
    #     for i in range(24):
    #         for j in range(3):
    #             temp1 = list(net.state_dict()["conv1.weight"][i][j].data.tolist())     
                
    #             for k in range(3):
    #                 for l in range(3):
    #                     decimalToBinary(temp1[k][l])

    #                     # o.xxoooxx
    #                     temp_binary = str(temp_bin[0]) + str(temp_bin[3]) + str(temp_bin[4]) + str(temp_bin[5]) 
                        

    #                     # make sure the truncation is reasonalbe
    #                     # f1.write(str(temp1[k][l])+ "vs." + temp_binary + "\n")

    #                     if(k==2 & l==2 & j==2):
    #                         f1.write(temp_binary)
    #                     else:
    #                         f1.write(temp_binary+"_")

    #                     # reset the global scope list
    #                     temp_bin = [0]


    #         f1.write("\n")     

    # f1.close()
    ##############################################################################################################
    


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


 
    ##################### print the weight  into .dat file in float form  #####################
    # with open("conv1_weight_float.dat", "w") as f3:
    #     for i in range(24):
    #         for j in range(3):
    #             temp1 = list(net.state_dict()["conv1.weight"][i][j].data.tolist())     
                
    #             for k in range(3):
    #                 for l in range(3):
    #                     f3.write(str(temp1[k][l])+"\n")
                        
    #                     # reset the global scope list
    #                     temp_bin = [0]


    #         #f3.write("\n")     

    # f3.close()
    
    ##################### print the weight  into .dat file in float form  #####################
    # with open("conv1_bias_float.dat", "w") as f4:
    #     temp2 = list(net.state_dict()["conv1.bias"].data.tolist())
    #     for i in range(24):
    #         f4.write(str(temp2[i]) +"\n")
    #         temp_bin = [0]
    # f4.close()