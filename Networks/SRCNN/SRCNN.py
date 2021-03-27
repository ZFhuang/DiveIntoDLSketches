import torch
import matplotlib.pyplot as plt
from torch import nn, optim
import cv2
import numpy as np
from PIL import Image

# SRCNN, 最早的超分辨率卷积神经网络, 2014


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # 三个卷积层
        # O=（I-K+2P）/S+1
        # 三层的大小都是不变的, 通道数在改变
        self.conv1 = nn.Conv2d(1, 64, 9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, 1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, 5, padding=2)

    def forward(self, img):
        # 三层的学习率不同
        # 两个激活层
        img = torch.relu(self.conv1(img))
        img = torch.relu(self.conv2(img))
        # 注意最后一层不要激活
        return self.conv3(img)
