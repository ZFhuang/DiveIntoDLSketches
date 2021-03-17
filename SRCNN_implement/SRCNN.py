import torch
from torch import nn, optim

# SRCNN, 最早的超分辨率卷积神经网络
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1=nn.Conv2d(3,64,5, stride=1, padding=2)
        self.conv2=nn.Conv2d(64,32,1, stride=1, padding=0)
        self.conv3=nn.Conv2d(32,3,9, stride=1, padding=4)
    
    def forward(self, img):
        img=torch.relu(self.conv1(img))
        img=torch.relu(self.conv2(img))
        return torch.sigmoid(self.conv3(img))
