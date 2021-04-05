import torch
import logging
from torch import nn

# RCAN, 注意力机制, 2018


class RCAN(nn.Module):
    def __init__(self, in_channel, scale=2, num_filter=64, num_residual_group=10, num_residual=20):
        super(RCAN, self).__init__()
        logging.debug('in_channel: '+str(in_channel))
        logging.debug('scale: '+str(scale))
        logging.debug('num_filter: '+str(num_filter))
        logging.debug('num_residual_group: '+str(num_residual_group))
        logging.debug('num_residual: '+str(num_residual))
        self.num_residual_group = num_residual_group
        self.input_conv = nn.Conv2d(in_channel, num_filter, 3, padding=1)
        seq = []
        for _ in range(num_residual_group):
            seq.append(RCAB(num_residual, num_filter))
        self.rir = nn.Sequential(*seq)
        self.rir_conv = nn.Conv2d(num_filter, num_filter, 3, padding=1)
        self.upsample = ESPCN(num_filter, scale)
        self.output_conv = nn.Conv2d(num_filter, in_channel, 3, padding=1)

    def forward(self, x):
        x = self.input_conv(x)
        skip = x
        for _ in range(self.num_residual_group):
            x = self.rir(x)
        x = skip+self.rir_conv(x)
        x = self.upsample(x)
        return self.output_conv(x)


class CA(nn.Module):
    def __init__(self, num_filter):
        super(CA, self).__init__()
        self.HGP = nn.AdaptiveAvgPool2d((1,1))
        self.wD = nn.Conv2d(num_filter, 4, 1)
        self.wU = nn.Conv2d(4, num_filter, 1)
        self.f = nn.Sigmoid()

    def forward(self, x):
        skip = x
        x = self.HGP(x)
        x = torch.relu_(self.wD(x))
        x = self.wU(x)
        x = self.f(x)
        return skip*x


class RCAB(nn.Module):
    def __init__(self, num_residual, num_filter):
        super(RCAB, self).__init__()
        self.num_residual = num_residual
        seq = []
        for _ in range(num_residual):
            seq.append(nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(num_filter, num_filter, 3, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(num_filter, num_filter, 3, padding=1)
                ),
                CA(num_filter)
            ))
        self.residuals = nn.Sequential(*seq)

    def forward(self, x):
        for i in range(self.num_residual):
            skip = x
            x = skip+self.residuals[i](x)
        return x


class ESPCN(nn.Module):
    def __init__(self, in_channel, scale=2):
        super(ESPCN, self).__init__()
        # 卷积尺寸计算 O=（I-K+2P）/S+1
        self.add_module('n1 conv', nn.Conv2d(in_channel, 64, 5, padding=2))
        self.add_module('tanh 1', nn.Tanh())
        self.add_module('n2 conv', nn.Conv2d(64, 32, 3, padding=1))
        self.add_module('tanh 2', nn.Tanh())
        self.add_module('n3 conv', nn.Conv2d(32, in_channel*scale*scale, 3, padding=1))
        self.add_module('pixel shuf', nn.PixelShuffle(scale))

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x
