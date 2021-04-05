import torch
import logging
from torch import nn

# DBPN, 带反馈结构的超分辨, 2018


class Dense_DBPN(nn.Module):
    def __init__(self, in_channel=1, scale=2, num_pair=2,  num_filter=18, grow_rate=18):
        super(Dense_DBPN, self).__init__()
        logging.debug('in_channel: '+str(in_channel))
        logging.debug('scale: '+str(scale))
        logging.debug('num_pair: '+str(num_pair))
        logging.debug('num_filter: '+str(num_filter))
        logging.debug('grow_rate: '+str(grow_rate))
        self.num_pair = num_pair
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1),
            nn.Conv2d(64, num_filter, 1, padding=0)
        )

        # pair+1个上采样块, 需要数量比下采样多一个
        seq = []
        seq.append(Up_projection(num_filter, grow_rate, scale))
        for i in range(num_pair):
            seq.append(Up_projection((i+1)*grow_rate, grow_rate, scale))
        self.up_proj = nn.Sequential(*seq)

        # pair个下采样块
        seq = []
        for i in range(num_pair):
            seq.append(Down_projection(
                (i+1)*grow_rate, grow_rate, scale))
        self.down_proj = nn.Sequential(*seq)

        self.reconstruction = nn.Conv2d(
            (num_pair+1)*grow_rate, 1, 3, padding=1)

    def forward(self, x):
        x = self.feature_extraction(x)
        up_skip = self.up_proj[0](x)
        down_skip = []
        for i in range(self.num_pair):
            if i == 0:
                down_skip = self.down_proj[i](up_skip)
            else:
                down_skip = torch.cat(
                    (self.down_proj[i](up_skip), down_skip), dim=1)
            up_skip = torch.cat((self.up_proj[i+1](down_skip), up_skip), dim=1)
        x = self.reconstruction(up_skip)
        return x


class Up_projection(nn.Module):
    # 每导入一次块就x2
    def __init__(self, in_channel, out_channel=28, scale=2, deconv_size=8):
        super(Up_projection, self).__init__()
        self.input_conv = nn.Conv2d(in_channel, out_channel, 1, padding=0)
        self.deconv_base = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel,
                               deconv_size, stride=scale, padding=deconv_size//2-1),
            nn.PReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 6, stride=scale, padding=2),
            nn.PReLU()
        )
        self.deconv_fea = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel,
                               deconv_size, stride=scale, padding=deconv_size//2-1),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.input_conv(x)
        skip_fea = x
        x = self.deconv_base(x)
        skip_base = x
        x = self.conv(x)
        x = self.deconv_fea(x-skip_fea)
        return x+skip_base


class Down_projection(nn.Module):
    # 每导入一次块就x2
    def __init__(self, in_channel, out_channel=28, scale=2, deconv_size=8):
        super(Down_projection, self).__init__()
        self.input_conv = nn.Conv2d(in_channel, out_channel, 1, padding=0)
        self.conv_base = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 6, stride=2, padding=2),
            nn.PReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel,
                               deconv_size, stride=scale, padding=deconv_size//2-1),
            nn.PReLU()
        )
        self.conv_fea = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 6, stride=2, padding=2),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.input_conv(x)
        skip_fea = x
        x = self.conv_base(x)
        skip_base = x
        x = self.deconv(x)
        x = self.conv_fea(x-skip_fea)
        return x+skip_base
