import torch
import logging
from torch import nn
from torch.nn.modules.activation import LeakyReLU, PReLU

# EDSR, 改良的SRResNet, 2017


class EDSR(nn.Module):
    def __init__(self, scale=4, in_channel=3, num_filter=256, num_resiblk=32, resi_scale=0.1):
        super(EDSR, self).__init__()
        # 去掉了bn层, 且残差以外不再有relu
        self.num_filter = num_filter
        self.num_resiblk = num_resiblk
        self.resi_scale=resi_scale
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channel, num_filter, 9, padding=4),
        )
        seq = []
        for _ in range(num_resiblk):
            seq.append(nn.Sequential(
                nn.Conv2d(num_filter, num_filter, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(num_filter, num_filter, 3, padding=1),
            ))
        self.residual_blocks = nn.Sequential(*seq)
        self.resi_out = nn.Sequential(
            nn.Conv2d(num_filter, num_filter, 3, padding=1),
        )
        # 上采样
        seq = []
        for _ in range(scale//2):
            seq.append(nn.Sequential(
                nn.Conv2d(num_filter, num_filter*4, 3, padding=1),
                nn.PixelShuffle(2),
            ))
        self.upsample = nn.Sequential(*seq)
        self.output_conv = nn.Conv2d(num_filter, in_channel, 3, padding=1)

    def forward(self, x):
        x = self.input_conv(x)
        # 内外两种残差连接
        skip = x
        resi_skip = x
        for i in range(self.num_resiblk):
            x = self.residual_blocks[i](x)*self.resi_scale+resi_skip
            resi_skip = x
        x = self.resi_out(x)+skip
        x = self.upsample(x)
        return self.output_conv(x)