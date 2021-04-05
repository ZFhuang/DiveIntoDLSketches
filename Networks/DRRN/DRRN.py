import torch
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d

# DRCN, 改进的递归残差网络, 2017


class DRRN(nn.Module):
    # 文中建议的各层组合是令 d=(1+2*num_resid_units)*num_recur_blocks+1 为 20左右
    def __init__(self, num_recur_blocks=1, num_resid_units=10, num_filter=128, filter_size=3):
        super(DRRN, self).__init__()
        # 多个递归块连接
        seq = []
        for i in range(num_recur_blocks):
            if i == 0:
                # 第一个递归块
                seq.append(RecursiveBlock(
                    num_resid_units, 1, num_filter, filter_size))
            else:
                seq.append(RecursiveBlock(num_resid_units,
                                          num_filter, num_filter, filter_size))
        self.residual_blocks = nn.Sequential(*seq)
        # 最终的出口卷积
        self.last_conv = nn.Conv2d(
            num_filter, 1, filter_size, padding=filter_size//2)

    def forward(self, img):
        skip = img
        img = self.residual_blocks(img)
        img = self.last_conv(img)
        # 总残差
        img = skip+img
        return img


class RecursiveBlock(nn.Module):
    # 类似DRCN的递归残差结构, 在RecursiveBlock内部的多个ResidualBlock权值是共享的
    def __init__(self, num_resid_units=3, input_channel=128, output_channel=128, filter_size=3):
        super(RecursiveBlock, self).__init__()
        self.num_resid_units = num_resid_units
        # 递归块的入口卷积
        self.input_conv = nn.Conv2d(
            input_channel, output_channel, filter_size, padding=filter_size//2)
        self.residual_unit = nn.Sequential(
            # 两个conv组, 都有一套激活和加权
            nn.Conv2d(output_channel, output_channel, filter_size,
                      padding=filter_size//2),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(True),
            nn.Conv2d(output_channel, output_channel, 1),

            nn.Conv2d(output_channel, output_channel, filter_size,
                      padding=filter_size//2),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(True),
            nn.Conv2d(output_channel, output_channel, 1)
        )

    def forward(self, x_b):
        x_b = self.input_conv(x_b)
        skip = x_b
        # 多次残差, 重复利用同一个递归块
        for i in range(self.num_resid_units):
            x_b = self.residual_unit(x_b)
            x_b = skip+x_b
        return x_b
