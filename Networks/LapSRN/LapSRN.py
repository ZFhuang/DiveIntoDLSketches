import torch
from torch import nn

# LapSRN, 拉普拉斯金字塔结构, 2017


class LapSRN(nn.Module):
    def __init__(self, fea_chan=64, scale=2, conv_num=3):
        super(LapSRN, self).__init__()
        self.level_num = int(scale/2)
        self.share_ski_upsample = nn.ConvTranspose2d(
            1, 1, 4, stride=scale, padding=1)
        self.input_conv = nn.Conv2d(1, fea_chan, 3, padding=1)
        seq = []
        for _ in range(conv_num):
            seq.append(nn.Conv2d(fea_chan, fea_chan, 3, padding=1))
            seq.append(nn.LeakyReLU(0.2, True))
        self.share_embedding = nn.Sequential(*seq)
        self.share_fea_upsample = nn.ConvTranspose2d(
            fea_chan, fea_chan, 4, stride=scale, padding=1)
        self.share_output_conv = nn.Conv2d(fea_chan, 1, 3, padding=1)

    def forward(self, img):
        tmp = self.input_conv(img)
        for _ in range(self.level_num):
            skip = self.share_ski_upsample(img)
            img = self.share_embedding(tmp)
            img = self.share_fea_upsample(img)
            tmp = img
            img = self.share_output_conv(img)
            img = img+skip
        return img


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
