from torch import nn

# ZSSR, 无监督超分辨率, 2018


class ZSSR(nn.Module):
    def __init__(self, in_channel=1, num_layers=8,num_filter=64):
        # 简单的全卷积残差网络
        super(ZSSR, self).__init__()
        self.input_conv=nn.Conv2d(in_channel,num_filter,3,padding=1)
        seq=[]
        for _ in range(num_layers-2):
            seq.append(nn.Conv2d(num_filter,num_filter,3,padding=1))
            seq.append(nn.ReLU(True))
        self.body=nn.Sequential(*seq)
        self.output_conv=nn.Conv2d(num_filter,in_channel,3,padding=1)


    def forward(self, x):
        skip=x
        x=self.input_conv(x)
        x=self.body(x)
        x=skip+x
        return self.output_conv(x)