import torch
from torch import nn
from torch.nn.modules.activation import ReLU

# RED, 对称残差网络, 2016


class RED(nn.Module):
    def __init__(self,ratio=2, num_feature=32, num_con_decon_mod=5, filter_size=3, skip_step=2):
        super(RED, self).__init__()
        if num_con_decon_mod//skip_step < 1:
            print('Size ERROR!')
            return
        self.num_con_decon_mod = num_con_decon_mod
        self.skip_step = skip_step
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, num_feature, 3, padding=1),
            nn.ReLU(True)
            )
        # 提取特征, 要保持大小不变
        conv_seq = []
        for i in range(0, num_con_decon_mod):
            conv_seq.append(nn.Sequential(
                nn.Conv2d(num_feature, num_feature,
                          filter_size, padding=filter_size//2),
                nn.ReLU(True)
            ))
        self.convs= nn.Sequential(*conv_seq)
        # 反卷积返还特征, 要保持大小不变
        deconv_seq = []
        for i in range(0, num_con_decon_mod):
            deconv_seq.append(nn.Sequential(
                nn.ConvTranspose2d(num_feature, num_feature, filter_size,padding=filter_size//2),
                nn.ReLU(True)
            ))
        self.deconvs=nn.Sequential(*deconv_seq)
        # 真正的放大步骤
        self.output_conv = nn.ConvTranspose2d(num_feature, 1, 3,stride=ratio,padding=1)

    def forward(self, img):
        img = self.input_conv(img)
        skips = []
        # 对称残差连接
        for i in range(0, self.num_con_decon_mod):
            if i%self.skip_step==0:
                skips.append(img)
            img = self.convs[i](img)
        for i in range(0, self.num_con_decon_mod):
            img = self.deconvs[i](img)
            if i%self.skip_step==0:
                img=img+skips.pop()
                # 测试中这里不激活效果更好
                # img=torch.relu(img+skips.pop())
        img=self.output_conv(img)
        return img
