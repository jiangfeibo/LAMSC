from torchsummary import summary
import torch
from torch import nn
from neural_nets import SCNet
from channel_nets import channel_net

# merge semantic model and channel model as a SC model
class base_net(nn.Module):
    def __init__(self, isc_model, channel_model):
        super(base_net, self).__init__()
        self.isc_model = isc_model # images SC model
        self.ch_model = channel_model # channel model

    def forward(self,x):
        s_code = self.isc_model(x)
        c_code, c_code_, s_code_ = self.ch_model(s_code)
        im_decoding = self.isc_model(latent = s_code_)
        # return raw and recovered channel code, raw and recovered semantic code, and the recovered image array
        return c_code, c_code_, s_code, s_code_, im_decoding

if __name__ == '__main__':
    SC_model = SCNet()
    channel_model = channel_net(in_dims=5408,snr=25)
    model = base_net(SC_model, channel_model).to("cuda")
    summary(model,(3,64,64),device="cuda")
