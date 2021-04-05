import torch
import logging
from torch import nn
from torch.nn.modules.activation import LeakyReLU, PReLU

# SRGAN, 生成对抗超分辨, 2017


class SRGAN_generator(nn.Module):
    # 也称SRResNet, 用来生成图像
    def __init__(self, scale=4, in_channel=3, num_filter=64, num_resiblk=16):
        logging.debug('scale: '+str(scale))
        logging.debug('in_channel: '+str(in_channel))
        logging.debug('num_filter: '+str(num_filter))
        logging.debug('num_resiblk: '+str(num_resiblk))
        super(SRGAN_generator, self).__init__()
        self.num_filter = num_filter
        self.num_resiblk = num_resiblk
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channel, num_filter, 9, padding=4),
            nn.PReLU()
        )
        # 大量的残差块
        seq = []
        for _ in range(num_resiblk):
            seq.append(nn.Sequential(
                nn.Conv2d(num_filter, num_filter, 3, padding=1),
                nn.BatchNorm2d(num_filter),
                nn.PReLU(),
                nn.Conv2d(num_filter, num_filter, 3, padding=1),
                nn.BatchNorm2d(num_filter),
            ))
        self.residual_blocks = nn.Sequential(*seq)
        self.resi_out = nn.Sequential(
            nn.Conv2d(num_filter, num_filter, 3, padding=1),
            nn.BatchNorm2d(num_filter),
        )
        # 上采样
        seq = []
        for _ in range(scale//2):
            seq.append(nn.Sequential(
                nn.Conv2d(num_filter, num_filter*4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ))
        self.upsample = nn.Sequential(*seq)
        self.output_conv = nn.Conv2d(num_filter, in_channel, 3, padding=1)

    def forward(self, x):
        x = self.input_conv(x)
        # 内外两种残差连接
        skip = x
        resi_skip = x
        for i in range(self.num_resiblk):
            x = self.residual_blocks[i](x)+resi_skip
            resi_skip = x
        x = self.resi_out(x)+skip
        x = self.upsample(x)
        return self.output_conv(x)


class SRGAN_discriminator(nn.Module):
    # 也称VGG19, 用来判别当前图像是否是真实的
    def __init__(self, in_channel=3):
        logging.debug('in_channel: '+str(in_channel))
        super(SRGAN_discriminator, self).__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        # 大量卷积层来提取特征
        self.convs = nn.Sequential(
            conv_layer(64, 64, 2),
            conv_layer(64, 128, 1),
            conv_layer(128, 128, 2),
            conv_layer(128, 256, 1),
            conv_layer(256, 256, 2),
            conv_layer(256, 512, 1),
            conv_layer(512, 512, 2)
        )
        self.output_conv = nn.Sequential(
            # 这里通过池化和卷积将高维数据变为单一的正负输出
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, 1, padding=0)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_conv(x)
        x = self.convs(x)
        x = self.output_conv(x)
        # 注意分类网络最后的激活
        return torch.sigmoid(x.view(batch_size))


def conv_layer(in_channel, out_channel, stride):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(inplace=True)
    )


class Adversarial_loss(nn.Module):
    # 损失函数, 是两种损失的结合
    def __init__(self, disc_alpha=1e-3):
        super(Adversarial_loss, self).__init__()
        self.alpha = disc_alpha
        self.mse_loss = nn.MSELoss()

    def forward(self, X, Y, loss_disc):
        # 图像本身loss是MSE
        loss = self.mse_loss(X,Y)
        # 判别器loss
        _loss_disc=loss_disc.detach()
        _loss_disc=torch.sum(-torch.log(_loss_disc))
        # 结合
        loss = loss+self.alpha*_loss_disc
        return loss
