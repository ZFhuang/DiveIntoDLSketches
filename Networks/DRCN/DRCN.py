import torch
from torch import nn

# DRCN, 深度递归超分辨, 2016


class DRCN(nn.Module):
    def __init__(self, recur_time=16):
        super(DRCN, self).__init__()
        self.recur_time = recur_time
        self.Embedding = nn.Sequential(
            nn.Conv2d(1, 256, 3, padding=1),
            nn.ReLU(True)
        )
        self.Inference = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True)
        )
        self.Reconstruction = nn.Sequential(
            nn.Conv2d(256, 1, 3, padding=1),
            nn.ReLU(True)
        )
        self.WeightSum = nn.Conv2d(recur_time, 1, 1)

    def forward(self, img):
        skip = img
        img = self.Embedding(img)
        output = torch.empty(
            (img.shape[0], self.recur_time, img.shape[2], img.shape[3]),device='cuda')
        # 残差连接
        for i in range(self.recur_time):
            img = self.Inference(img)
            output[:, i, :, :] = (skip+self.Reconstruction(img)).squeeze(1)
        # 加权合并
        output = self.WeightSum(output)
        return output
