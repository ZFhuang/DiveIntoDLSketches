import torch
from torch import nn

# FastSRCNN, 速度更快的SRCNN, 2016
class FSRCNN(nn.Module):
    def __init__(self,d,s,m,ratio=2):
        super(FSRCNN, self).__init__()
        # 卷积尺寸计算 O=（I-K+2P）/S+1
        feature_extraction=nn.Conv2d(1,d,5, padding=2)
        shrinking=nn.Conv2d(d,s,1)
        seq=[]
        for i in range(m):
            seq.append(nn.Conv2d(s,s,3,padding=1))
        non_linear=nn.Sequential(*seq)
        expanding=nn.Conv2d(s,d,1,padding=0)
        # 反卷积尺寸计算 O=(I-1)×s+k-2P 
        deconvolution=nn.ConvTranspose2d(d,1,9,stride=ratio,padding=4)
        self.body=nn.Sequential(
            feature_extraction,
            nn.PReLU(),
            shrinking,
            nn.PReLU(),
            non_linear,
            nn.PReLU(),
            expanding,
            nn.PReLU(),
            deconvolution
        )
    
    def forward(self, img):
        return self.body(img)