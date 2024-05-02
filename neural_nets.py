
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
import numpy as np
from torch.autograd import Variable

class SCNet(nn.Module):
    def __init__(self,input_dim=3, ASC=False):
        super(SCNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=5,bias=False)
        self.pool = nn.MaxPool2d((2, 2),return_indices=True)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=5,bias=False)
        self.use_ASC = ASC
        self.Mask = MaskNet(32)
        self.convt1= nn.ConvTranspose2d(32, 128, kernel_size=5)
        self.convt2 = nn.ConvTranspose2d(128, input_dim, kernel_size=5)
        self.uppool = nn.MaxUnpool2d(2, 2)

    def forward(self, x = None, latent = None):
        if latent == None:
            x = F.leaky_relu(self.conv1(x))
            x, self.indices1 = self.pool(x)
            x = F.leaky_relu(self.conv2(x))
            x, self.indices2 = self.pool(x)
            self.x_shape = x.shape
            if self.use_ASC:
                x = self.Mask(x)
            latent = x.view(x.size(0), -1)
            return latent
        else:
            x = latent.view(self.x_shape)
            x = self.uppool(x,self.indices2)
            x = F.leaky_relu(self.convt1(x))
            x = self.uppool(x,self.indices1)
            x = F.tanh(self.convt2(x))
            return x

class MaskNet(nn.Module):
    def __init__(self,input_dim=32):
        super(MaskNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(128, input_dim, kernel_size=3,padding=1)

    def forward(self, x):
        y = self.conv1(x)
        y = F.relu(y)
        mask = self.conv2(y) + torch.abs(x)
        mask = torch.sign(mask)
        mask = F.relu(mask)
        x = torch.mul(x, mask)
        # print(x.shape)
        # index = torch.where(x!=0)
        # retain_x = x[index]
        # print("压缩语义bit:", retain_x.element_size() * retain_x.nelement())

        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_dims):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_dims, 128, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(128, in_dims, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class AttentionNet(nn.Module):
    def __init__(self, in_dims=3*3, kernel_size=7):
        super(AttentionNet, self).__init__()
        self.ca = ChannelAttention(in_dims)
        self.sa = SpatialAttention(kernel_size)
        self.out1 = nn.Linear(36864,128)
        self.out2 = nn.Linear(128,3)


    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.out1(x))
        x = F.sigmoid(self.out2(x))
        return x


if __name__ == '__main__':
    net = SCNet()
    net.to("cuda")
    summary(net,(3,64,64),device="cuda")

    net = AttentionNet()
    net.to("cuda")
    summary(net, (3*3, 64, 64), device="cuda")


