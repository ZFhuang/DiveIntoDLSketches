import torch
from torch import nn

# ESPCN, 实时SRCNN, 2016


class ESPCN(nn.Module):
    def __init__(self, ratio=2):
        super(ESPCN, self).__init__()
        # 卷积尺寸计算 O=（I-K+2P）/S+1
        self.add_module('n1 conv', nn.Conv2d(1, 64, 5, padding=2))
        self.add_module('tanh 1', nn.Tanh())
        self.add_module('n2 conv', nn.Conv2d(64, 32, 3, padding=1))
        self.add_module('tanh 2', nn.Tanh())
        self.add_module('n3 conv', nn.Conv2d(32, 1*ratio*ratio, 3, padding=1))
        self.add_module('pixel shuf', nn.PixelShuffle(ratio))

    def forward(self, img):
        for module in self._modules.values():
            img = module(img)
        return img
