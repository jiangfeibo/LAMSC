from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pickle

def AWGN_channel(x, snr):  # used to simulate additive white gaussian noise channel
    [batch_size, length] = x.shape
    x_power = torch.sum(torch.abs(x)) / (batch_size * length)
    n_power = x_power / (10 ** (snr / 10.0))
    noise = torch.rand(batch_size, length, device="cuda") *n_power
    return x + noise

class channel_net(nn.Module):
    def __init__(self, in_dims=800, mid_dims=128, snr=25):
        super(channel_net, self).__init__()
        self.enc_fc = nn.Linear(in_dims, mid_dims)
        self.dec_fc = nn.Linear(mid_dims, in_dims)
        self.snr = snr

    def forward(self, x):
        ch_code = self.enc_fc(x)
        ch_code_with_n = AWGN_channel(ch_code,self.snr)
        x = self.dec_fc(ch_code_with_n)
        return ch_code,ch_code_with_n,x

class MutualInfoSystem(nn.Module):  # mutual information used to maximize channel capacity
    def __init__(self):
        super(MutualInfoSystem, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, inputs):
        output = F.relu(self.fc1(inputs))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        return output

def sample_batch(batch_size, sample_mode, x, y):  # used to sample data for mutual info system
    length = x.shape[0]
    if sample_mode == 'joint':
        index = np.random.choice(range(length), size=batch_size, replace=False)
        batch_x = x[index, :]
        batch_y = y[index, :]
    elif sample_mode == 'marginal':
        joint_index = np.random.choice(range(length), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(length), size=batch_size, replace=False)
        batch_x = x[joint_index, :]
        batch_y = y[marginal_index, :]
    batch = torch.cat((batch_x, batch_y), 1)
    return batch

from torchsummary import summary

if __name__ == '__main__':
    net = channel_net()
    summary(net,device="cpu")
