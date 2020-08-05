# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import layers
from d2lzh_pytorch import data_process
from d2lzh_pytorch import train

import torch
import time
import torch.nn.functional as F
from torch import nn,optim

"""
这一节介绍了并联多个网络的GoogLeNet, 它的运行效率很高效果也很好，设计精细
"""

# 加上这个限定才能支持多线程读取
if __name__ == "__main__":
    # 设置计算设备，让计算在GPU上进行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GoogLeNet的核心，Inception块，将多个卷积并联起来进行
    class Inception(nn.Module):
        # 这里的c1到c4的这么多参数是各个线路的通道数
        def __init__(self, in_c, c1, c2, c3, c4):
            super(Inception, self).__init__()
            # 由四条线路组成
            self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
            self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
            self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
            self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
            self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
            self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

        def forward(self, x):
            # 增加激活层并连接不同的单元
            p1 = F.relu(self.p1_1(x))
            p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
            p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
            p4 = F.relu(self.p4_2(self.p4_1(x)))
            # 连接各个通道并输出
            return torch.cat((p1, p2, p3, p4), dim=1)

    # 然后我们要组合出GoogLeNet，GoogLeNet结构复杂,由五个子网络模块串联得到
    # 首先是一个有较大卷积层的块提取特征，然后用最大池化缩小图片
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # 然后再使用两个卷积层将通道继续增加，池化缩小图片
    b2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1),
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # 第三个模块串联了两个Inception并池化，谷歌提出说池化层的效果非常好
    b3 = nn.Sequential(
        Inception(192, 64, (96, 128), (16, 32), 32),
        Inception(256, 128, (128, 192), (32, 96), 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # 五个Inception的串联，可以看到通道数大多成倍数关系
    b4 = nn.Sequential(
        Inception(480, 192, (96, 208), (16, 48), 64),
        Inception(512, 160, (112, 224), (24, 64), 64),
        Inception(512, 128, (128, 256), (24, 64), 64),
        Inception(512, 112, (144, 288), (32, 64), 64),
        Inception(528, 256, (160, 320), (32, 128), 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # 第五个模块是最终的输出模块，两个Inception进行最后的提取后，准备给全连接层分类
    b5 = nn.Sequential(
        Inception(832, 256, (160, 320), (32, 128), 128),
        Inception(832, 384, (192, 384), (48, 128), 128),
        # 全局平均池化用来给后面的全连接层连接
        layers.GlobalAvgPool2d()
    )
    # 最后把上面的模块串联到一起
    net = nn.Sequential(
        b1,
        b2,
        b3,
        b4,
        b5,
        # 打平
        layers.FlattenLayer(),
        # 输出
        nn.Linear(1024, 10)
    )

    # 这里先输出一下网络形状检查一下
    X = torch.rand(1, 1, 96, 96)
    for blk in net.children():
        X = blk(X)
        print('output shape: ', X.shape)
    print('————————————————————————————')

    # 训练的处理之前相同，但是为了节省实际这里把图片尺寸缩小了
    # 1 epoch = 229.8sec
    # 由于网络很大可能会爆显存
    batch_size = 128
    train_iter, test_iter = data_process.load_data_fashion_mnist(
        batch_size, resize=96)
    lr, num_epochs = 0.001, 5
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    train.train_ch5(net, train_iter, test_iter,
                    batch_size, optim, device, num_epochs)
