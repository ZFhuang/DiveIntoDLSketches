import torch
import logging
from torch import nn

# SRDenseNet, 稠密连接超分辨, 2017


class SRDenseNet(nn.Module):
    def __init__(self, scale=4, dense_input_chan=16, growth_rate=16, bottleneck_channel=256, num_dense_conv=8, num_dense=8):
        super(SRDenseNet, self).__init__()
        logging.debug('scale: '+str(scale))
        logging.debug('dense_input_chan: '+str(dense_input_chan))
        logging.debug('growth_rate: '+str(growth_rate))
        logging.debug('bottleneck_channel: '+str(bottleneck_channel))
        logging.debug('num_dense_conv: '+str(num_dense_conv))
        logging.debug('num_dense: '+str(num_dense))
        self.dense_input_chan = dense_input_chan
        self.growth_rate = growth_rate
        self.num_dense_conv = num_dense_conv
        self.num_dense = num_dense

        # 输入层, 通道数转为dense_input_chan
        self.input = nn.Sequential(
            nn.Conv2d(1, dense_input_chan, 3, padding=1),
            nn.ReLU(True)
        )

        # 稠密层, 由多个稠密块组成, 有skip连接, 输出通道num_dense*num_dense_conv*growth_rate+dense_input_chan
        seq = []
        for i in range(num_dense):
            seq.append(DenseBlock((i*num_dense_conv*growth_rate) +
                                  dense_input_chan, growth_rate, num_dense_conv))
        self.dense_blocks = nn.Sequential(*seq)

        # 缩小输出时候的维度的瓶颈层, 输出通道bottleneck_channel
        self.bottleneck = bottleneck_layer(
            num_dense*num_dense_conv*growth_rate+dense_input_chan, bottleneck_channel)

        # 用于上采样的反卷积层, 通道保持bottleneck_channel
        seq = []
        for _ in range(scale//2):
            seq.append(nn.ConvTranspose2d(bottleneck_channel,
                                          bottleneck_channel, 3, stride=2, padding=1))
        self.deconv = nn.Sequential(*seq)

        # 输出层, 输出通道1
        self.output = nn.Conv2d(bottleneck_channel, 1, 3, padding=1)

    def forward(self, x):
        x = self.input(x)
        dense_skip = x
        for i in range(self.num_dense):
            x = self.dense_blocks[i](x)
            # 稠密残差连接
            dense_skip = torch.cat(
                (dense_skip, x[:, (i*self.num_dense_conv*self.growth_rate)+self.dense_input_chan:, :, :]), dim=1)
        x = self.bottleneck(dense_skip)
        x = self.deconv(x)
        x = self.output(x)
        return x


def bottleneck_layer(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1),
    )


def conv_layer(in_channel, out_channel):
    return nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, 3, padding=1),
    )


class DenseBlock(nn.Module):
    def __init__(self, in_channel=16, growth_rate=16, num_convs=8):
        super(DenseBlock, self).__init__()
        self.num_convs = num_convs
        seq = []
        for _ in range(num_convs):
            # 不断连接并增加着的特征图
            seq.append(conv_layer(in_channel, growth_rate))
            in_channel = in_channel+growth_rate
        self.convs = nn.Sequential(*seq)

    def forward(self, x):
        for i in range(self.num_convs):
            # 拼接之前得到的特征图
            y = self.convs[i](x)
            x = torch.cat((x, y), dim=1)
        return x
