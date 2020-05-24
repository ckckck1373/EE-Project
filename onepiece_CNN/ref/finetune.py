import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import csv
import torch.optim as optim
import argparse
import random
import copy
from datetime import datetime
from lenet import *
from data_setting import *
from arg_setting import *

def test_accuracy(model, testloader):
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = model(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
    print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct.cpu().numpy() / total))

def dynamic_quan(num_group, bit_width):
    '''Quantize a group of numbers into [min,max] in bit_num bits of expression'''
    num_min = torch.abs(num_group.min())
    num_max = torch.abs(num_group.max())

    if(num_max >= num_min):
        max_val_abs = num_max
    else:
        max_val_abs = num_min

    if max_val_abs <= 0.0001:
        max_val_abs = torch.tensor(0.0001).cuda()

    int_bit = torch.ceil(torch.log2(max_val_abs) + 1)
    fractional_length = bit_width - int_bit
    # use empirical ways to find best SQNR
    new_fra_length = find_best_sqnr(num_group, bit_width, fractional_length)

    interval = 2 ** (-1 * new_fra_length)
    half_interval = interval / 2
    max_val = (2 ** (bit_width - 1) - 1) * interval
    min_val = - (2 ** (bit_width - new_fra_length - 1))

    quan_group = torch.floor((num_group + half_interval) / interval)
    quan_group = quan_group * interval

    quan_group[quan_group >= max_val] = max_val
    quan_group[quan_group <= min_val] = min_val
    num_group.copy_(quan_group)
    return new_fra_length.item()

def find_best_sqnr(num_group, bit_width, fractional_length):
    best_sqnr = 0
    best_i = 0
    P_num = torch.sum(num_group ** 2)

    for i in range(5):
        interval = 2 ** (-1 * (fractional_length + i))
        half_interval = interval / 2
        max_val = (2 ** (bit_width - 1) - 1) * interval
        min_val = - (2 ** (bit_width - (fractional_length + i) - 1))

        quan_group = torch.floor((num_group + half_interval) / interval)
        quan_group = quan_group * interval

        quan_group[quan_group >= max_val] = max_val
        quan_group[quan_group <= min_val] = min_val

        diff = num_group - quan_group

        P_err = torch.sum(diff ** 2)

        sqnr = P_num / P_err

        if sqnr > best_sqnr:
            best_sqnr = sqnr
            best_i = i

    return fractional_length + best_i

def stochastic_sampling(num_group, bit_width):
    '''Quantize a group of numbers into [min,max] in bit_num bits of expression'''
    num_min = torch.abs(num_group.min())
    num_max = torch.abs(num_group.max())

    if(num_max >= num_min):
        max_val_abs = num_max
    else:
        max_val_abs = num_min

    int_bit = torch.ceil(torch.log2(max_val_abs) + 1)
    fractional_length = bit_width - int_bit

    new_fra_length = find_best_sqnr(num_group, bit_width, fractional_length)

    interval = 2 ** (-1 * new_fra_length)
    half_interval = interval / 2
    max_val = (2 ** (bit_width - 1) - 1) * interval
    min_val = - (2 ** (bit_width - fractional_length - 1))

    quan_group = (num_group) / interval
    rand_num = torch.rand_like(quan_group)
    rand_num.cuda()

    quan_group = torch.floor(quan_group+rand_num)
    quan_group = quan_group * interval

    quan_group[quan_group >= max_val] = max_val
    quan_group[quan_group <= min_val] = min_val
    num_group.copy_(quan_group)

def fine_tune_quan(model, trainloader, epoch_times):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epoch_times):  # loop over the dataset multiple times
        optimizer = optim.SGD(model.parameters(), lr=learning_rate(0.1, epoch), weight_decay=0.002)
        running_loss = 0.0
        model.train()
        for i, (data, labels) in enumerate(trainloader):
            # get the inputs
            data, labels = Variable(data.cuda()), Variable(labels.cuda())
                    
            now_dict = copy.deepcopy(model.state_dict())
            quan_dict = copy.deepcopy(model.state_dict())

            # stochastic sampling 
            for target_name, layer_param in quan_dict.items():
                stochastic_sampling(layer_param, args.bw_param)

            model.load_state_dict(quan_dict)

            # initial the gradient
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(data)
            loss = criterion(output, labels)

            # Get the gradient w'
            loss.backward()
            quan_param = model.parameters()

            model.load_state_dict(now_dict)
            #update to full precision weights
            for old, new in zip(model.parameters(),quan_param):
                old.grad = copy.deepcopy(new.grad)

            optimizer.step()

            # print statistics
            running_loss += loss.data.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        before_qu_state_dict = copy.deepcopy(model.state_dict())
        new_state_dict = copy.deepcopy(model.state_dict())

        for target_name, layer_param in new_state_dict.items():
            dynamic_quan(layer_param, args.bw_param)
                
        model.eval()

        print("After fine-tune and after quantize, accuracy")
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
            outputs = model.module.quan_forward(images, fl_dict)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
            accuracy = (100 * correct.cpu().numpy() / total)
        print('Accuracy of the quantized network on the 10000 test images: %.3f %%' % accuracy)

        global best_acc_quan

        if accuracy > best_acc_quan:
            savepoint_name = "model_pretrained/lenet_mnist_finetuned.tar"
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' % accuracy)
            torch.save({
                'state': before_qu_state_dict ,
                'arch': 'resnet_cifar',
                }, savepoint_name)
            best_acc_quan = accuracy

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 15):
        optim_factor = 0.001
    elif(epoch > 10):
        optim_factor = 0.01
    elif(epoch > 5):
        optim_factor = 0.1
    else:
        optim_factor = 1
    return init*optim_factor

model = Net()
criterion = nn.CrossEntropyLoss()
# GPU mode
if torch.cuda.is_available():
    model.cuda()
    model = torch.nn.DataParallel(model)

best_acc_quan = 0

# load the pretrained model
print('===> Loading the pretrained model')
savepoint = torch.load(args.model)
ori_state_dict = savepoint['state']
model.load_state_dict(ori_state_dict)
# merge the conv layer and bn layer

max_dict = {}
fl_dict = {}
# First, collect the activation from 100 training image, then get the maximum value and record it 
for index, data in enumerate(trainloader):
    images, labels = data
    images, labels = Variable(images.cuda()), Variable(labels.cuda())
    # Because i set the batch size to 100, i only need one set of training images
    if(index == 1):
        data_box = model.module.collect_data(images)
        for layer_name, param in data_box.items():
            max_dict[layer_name] = np.max(param.data.cpu().numpy())

# according to different policy to get base fractional length for activation
print("the following is the information for fractional part")
for layer_name in max_dict:
    fl_dict[layer_name] = int(args.bw_activation) - np.ceil(np.log2(max_dict[layer_name]))
    print(layer_name + " need %d bits" % fl_dict[layer_name])

print('=====================================')
print('===> Quantizing the parameter to %d bits and the activation to %d bits' % (args.bw_param, args.bw_activation))
quan_state_dict = copy.deepcopy(model.state_dict())

for target_name, layer_param in quan_state_dict.items():
    dynamic_quan(layer_param, args.bw_param)

model.load_state_dict(quan_state_dict)

print("original accuracy(quantize all)")
correct = 0
total = 0
for data in testloader:
    images, labels = data
    images, labels = Variable(images.cuda()), Variable(labels.cuda())
    outputs = model.module.quan_forward(images, fl_dict)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.data).sum()
    accuracy = (100 * correct.cpu().numpy() / total)
print('Accuracy of the quantized network on test images: %.3f %%' % accuracy)

epoch_times = 20

model.load_state_dict(ori_state_dict)

fine_tune_quan(model, trainloader, epoch_times)
print("The best acc for fixed-point is: %f" % best_acc_quan)