# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import data_process
from d2lzh_pytorch import train

import time
import torch
from torch import nn, optim

# 设置计算设备，让计算在GPU上进行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
这一节开始进入神经网络真正的重点：卷积神经网络
"""


# 这里用nn的模板来搭建一个卷积神经网络,这里搭建的是著名的LeNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            # 目标数据是Fashion-MNIST数据集，1*28*28
            # 这里是1输入通道，6输出通道，5*5的卷积核
            # 卷积将高宽变小，但通道变多了，6*24*24(28-5+0+1=24)
            nn.Conv2d(1, 6, 5),
            # LeNet使用的是Sigmoid函数，这个激活函数后来发现有很多缺点，例如计算慢
            nn.Sigmoid(),
            # 池化层大幅减小图像(除二)，6*12*12
            nn.MaxPool2d(2, 2),
            # 继续卷积并增加通道，16*8*8
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            # 再缩小图像,16*4*4
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            # 因此全连接的输入是4*4
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            # 最终目的是分类为10个类
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        return self.fc(feature.view(img.shape[0], -1))


net = LeNet()
print(net)
# 此时模型仍然在CPU上
print(list(net.parameters())[0].device)
print('————————————————————————————')


# 这里开始实际计算，由于卷积神经网络计算量很大，所以要迁移到GPU上进行
batch_size = 256
train_iter, test_iter = data_process.load_data_fashion_mnist(batch_size)
# 在这里修改了一下之前的evaluate_accuracy函数使其支持了GPU计算
# 然后修改了train_ch3函数，得到train_ch5，也是为了支持GPU并彻底支持了神经网络库
lr, num_epochs = 0.001, 5
optim = torch.optim.Adam(net.parameters(), lr=lr)
train.train_ch5(net, train_iter, test_iter,
                batch_size, optim, device, num_epochs)
