import torch
from torch import nn, optim

# SRCNN, 最早的超分辨率卷积神经网络
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # 三个卷积层
        # O=（I-K+2P）/S+1
        # 三层的大小都是不变的, 通道数在改变
        self.conv1=nn.Conv2d(3,64,9, stride=1, padding=4)
        self.conv2=nn.Conv2d(64,32,1, stride=1, padding=0)
        self.conv3=nn.Conv2d(32,3,5, stride=1, padding=2)
    
    def forward(self, img):
        # 两个激活层
        img=torch.relu(self.conv1(img))
        img=torch.relu(self.conv2(img))
        # 注意最后一层的激活比较特殊, sigmoid或tahn, 或者不要激活
        return self.conv3(img)
