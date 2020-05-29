import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from arg_setting import *

def find_best_sqnr(num_group, bit_width, frational_length):
    best_sqnr = 0
    best_i = 0
    P_num = torch.sum(num_group ** 2)

    for i in range(5):
        interval = 2 ** (-1 * (frational_length + i))
        half_interval = interval / 2
        max_val = (2 ** (bit_width - 1) - 1) * interval
        min_val = - (2 ** (bit_width - (frational_length + i) - 1))

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

    return frational_length + best_i

def pixel_unshuffle(input, upscale_factor):
    """
    Rearranges elements in a Tensor of shape (C, rH, rW) to (*, (r^2)*C, H, W)
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = input.contiguous().view(batch_size, channels, out_height, upscale_factor, out_width, upscale_factor)
    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

class PixelUnShuffle(nn.Module):
    """
    Rearranges elements in a Tensor of shape (C, rH, rW) to (*, r^2C, H, W)
    """
    def __init__(self, upscale_factor=2):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=0)
        self.unshuffle = PixelUnShuffle(upscale_factor=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024, 500)
        self.fc2 = nn.Linear(500, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv2(self.relu(self.conv1(self.unshuffle(x)))))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def collect_data(self, x):
        data_box = {}
        data_box['input'] = x
        data_box['unshuffle'] = self.unshuffle(x)
        data_box['conv1'] = self.relu(self.conv1(data_box['unshuffle']))
        data_box['conv2'] = self.relu(self.conv2(data_box['conv1']))
        data_box['conv3'] = self.pool(self.relu(self.conv3(data_box['conv2'])))
        k = data_box['conv3'].view(-1, 64 * 4 * 4)
        data_box['fc1'] = self.relu(self.fc1(k))
        data_box['fc2'] = self.fc2(data_box['fc1'])
        return data_box

    def quan_forward(self, x, fl_dict):
        x = self.quantize_activation(x, args.bw_activation, fl_dict['input'])
        x = self.unshuffle(x)
        x = self.quantize_activation(x, args.bw_activation, fl_dict['unshuffle'])
        x = self.relu(self.conv1(x))
        x = self.quantize_activation(x, args.bw_activation, fl_dict['conv1'])
        x = self.relu(self.conv2(x))
        x = self.quantize_activation(x, args.bw_activation, fl_dict['conv2'])
        x = self.pool(self.relu(self.conv3(x)))
        x = self.quantize_activation(x, args.bw_activation, fl_dict['conv3'])
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.quantize_activation(x, args.bw_activation, fl_dict['fc1'])
        x = self.fc2(x)
        x = self.quantize_activation(x, args.bw_activation, fl_dict['fc2'])
        return  x

    def quan_input(self, x, fl_dict):
        print('quan_input')
        x = self.quantize_activation(x, args.bw_activation, fl_dict['input'])
        return x

    def quan_unshuffle(self, x, fl_dict):
        print('quan_unshuffle')
        x = self.quantize_activation(x, args.bw_activation, fl_dict['input'])
        x = self.unshuffle(x)
        x = self.quantize_activation(x, args.bw_activation, fl_dict['unshuffle'])
        return x

    def quan_conv1(self, x, fl_dict):
        print('quan_conv1')
        x = self.quantize_activation(x, args.bw_activation, fl_dict['input'])
        x = self.unshuffle(x)
        x = self.quantize_activation(x, args.bw_activation, fl_dict['unshuffle'])
        x = self.relu(self.conv1(x))
        x = self.quantize_activation(x, args.bw_activation, fl_dict['conv1'])
        return x

    def quan_conv2(self, x, fl_dict):
        print('quan_conv2')
        x = self.quantize_activation(x, args.bw_activation, fl_dict['input'])
        x = self.unshuffle(x)
        x = self.quantize_activation(x, args.bw_activation, fl_dict['unshuffle'])
        x = self.relu(self.conv1(x))
        x = self.quantize_activation(x, args.bw_activation, fl_dict['conv1'])
        x = self.relu(self.conv2(x))
        x = self.quantize_activation(x, args.bw_activation, fl_dict['conv2'])
        return x

    def quan_conv3(self, x, fl_dict):
        print('quan_conv3')
        x = self.quantize_activation(x, args.bw_activation, fl_dict['input'])
        x = self.unshuffle(x)
        x = self.quantize_activation(x, args.bw_activation, fl_dict['unshuffle'])
        x = self.relu(self.conv1(x))
        x = self.quantize_activation(x, args.bw_activation, fl_dict['conv1'])
        x = self.relu(self.conv2(x))
        x = self.quantize_activation(x, args.bw_activation, fl_dict['conv2'])
        x = self.relu(self.conv3(x))
        x = self.quantize_activation(x, args.bw_activation, fl_dict['conv3'])
        return x

    def quan_pool(self, x, fl_dict):
        print('quan_pool')
        x = self.quantize_activation(x, args.bw_activation, fl_dict['input'])
        x = self.unshuffle(x)
        x = self.quantize_activation(x, args.bw_activation, fl_dict['unshuffle'])
        x = self.relu(self.conv1(x))
        x = self.quantize_activation(x, args.bw_activation, fl_dict['conv1'])
        x = self.relu(self.conv2(x))
        x = self.quantize_activation(x, args.bw_activation, fl_dict['conv2'])
        x = self.pool(self.relu(self.conv3(x)))
        x = self.quantize_activation(x, args.bw_activation, fl_dict['conv3'])
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
'''
class Quan_Net(Net):
    def __init__(self):
        super(Quan_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv1_1 = nn.Conv2d(1, 20, kernel_size=3)
        self.conv1_2 = nn.Conv2d(20, 20, kernel_size=3)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def quan_forward(self, x, fl_dict):
        x = self.quantize_activation(x, args.bw_activation, fl_dict['input'])
        x = F.relu(self.conv1_1(x))
        x = self.quantize_activation(x, args.bw_activation, fl_dict['conv1_1'])
        x = self.pool(F.relu(self.conv1_2(x)))
        x = self.quantize_activation(x, args.bw_activation, fl_dict['conv1_2'])
        x = self.pool(F.relu(self.conv2(x)))
        x = self.quantize_activation(x, args.bw_activation, fl_dict['conv2'])
        x = x.view(-1, 50 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.quantize_activation(x, args.bw_activation, fl_dict['fc1'])
        x = self.fc2(x)
        x = self.quantize_activation(x, args.bw_activation, fl_dict['fc2'])
        return  x

    def quantize_activation(self, num_group, bit_width, frational_length):
        # print("parameter_bit : ", bit_width, "frational length : ",frational_length)
        new_fra_length = find_best_sqnr(num_group, bit_width, frational_length)

        interval = 2 ** (-1 * new_fra_length)
        half_interval = interval / 2
        max_val = (2 ** (bit_width - 1) - 1) * interval
        min_val = - (2 ** (bit_width - new_fra_length - 1))

        quan_data = torch.floor((num_group.data + half_interval) / interval)
        quan_data = quan_data * interval

        quan_data[quan_data >= max_val] = max_val
        quan_data[quan_data <= min_val] = min_val
        num_group.data = quan_data.cuda()
        return num_group
'''