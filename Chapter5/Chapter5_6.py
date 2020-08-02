# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import data_process
from d2lzh_pytorch import train

import time
import torch
from torch import nn, optim
import torchvision


"""
这一节介绍了深度卷积神经网络AlexNet
"""


# AlexNet通过在LeNet的基础上进行的一些简单的改变使得深度卷积成为了可能
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv=nn.Sequential(
            # 目标输入的数据是1*224*224的很大的图片，所以一上来就用了很大的卷积核
            # 大的卷积核可以提取出比较大的特征，且快速缩小图片
            # 这里论文对尺寸的描述有问题，论文中输出是96*55*55，实际应是54或53
            nn.Conv2d(1,96,11,4),
            nn.ReLU(),
            # 池化减小大小
            nn.MaxPool2d(3,2),
            # 再来一次
            nn.Conv2d(96,256,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            # 连续三个卷积层大幅增加参数数量
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            # 再池化减小大小
            nn.MaxPool2d(3,2)
        )
        self.fc=nn.Sequential(
            # 然后在外部变平后用全连接层
            nn.Linear(256*5*5,4096),
            nn.ReLU(),
            # 丢弃法，防止过拟合
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,10)
        )
    
    def forward(self, img):
        feature = self.conv(img)
        # 变平并计算
        return self.fc(feature.view(img.shape[0], -1))

# 加上这个限定才能支持多线程读取
if __name__ == '__main__':
    net = AlexNet()
    print(net)
    print('————————————————————————————')

    # 设置计算设备，让计算在GPU上进行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 由于GPU的显存充足，可以适当调高batch_size加快训练
    batch_size = 256
    # 这里修改了load_data_fashion_mnist函数，让其支持在读取时对数据进行尺寸调整
    train_iter, test_iter = data_process.load_data_fashion_mnist(
        batch_size, resize=224)
    lr, num_epochs = 0.001, 5
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    # 1 epoch = 190 sec
    train.train_ch5(net, train_iter, test_iter,
                    batch_size, optim, device, num_epochs)
