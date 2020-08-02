# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import layers
from d2lzh_pytorch import data_process
from d2lzh_pytorch import train

import torch
import time
from torch import nn,optim

"""
这一节介绍了串联多个网络的"网络中的网络NiN"
"""

# 加上这个限定才能支持多线程读取
if __name__ == "__main__":
    # 设置计算设备，让计算在GPU上进行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 前几节的网络都是多个卷积层和池化组成的块然后串联几个全连接层组成的块形成的结构
    # NiN的特点在于其串联了多个"卷积+全连接"的子网络，由于全连接层需要打平数据
    # 所以这里全连接层用前面说到的1*1卷积来代替，从而让空间信息能自然传递到后面的层
    # 下面就是"卷积+全连接"的子网络NiN块的生成函数，网络中类似AlexNet来使用它
    def NiN_block(in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            # 一个和设定参数有关的核心卷积层
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            # 用两个kernel_size=1的卷积来代替全连接层
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )
        return block

    # 实例化这个网络
    net = nn.Sequential(
        # 一开始利用NiN块进行逐步缩小的卷积
        NiN_block(1, 96, kernel_size=11, stride=4, padding=0),
        nn.MaxPool2d(kernel_size=3, stride=2),
        NiN_block(96, 256, kernel_size=5, stride=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        NiN_block(256, 384, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Dropout(0.5),
        # 结尾时进行最后一次卷积，大幅减少通道数
        NiN_block(384, 10, kernel_size=3, stride=1, padding=1),
        # 然后利用全局池化大幅减少参数数量，直接合并整个面，将4维打为实际上的2维
        layers.GlobalAvgPool2d(),
        # 最后用一个扁平层作为输出
        layers.FlattenLayer()
    )

    # 和之前一样，测试一下网络结构
    X = torch.rand(1, 1, 224, 224)
    for name, blk in net.named_children():
        X = blk(X)
        print(name, 'output shape:', X.shape)

    print('————————————————————————————')

    # 训练的处理和AlexNet部分相同，但是这里选用更大的学习率，看一个epoch即可
    # 1 epoch = sec
    batch_size = 128
    train_iter, test_iter = data_process.load_data_fashion_mnist(
        batch_size, resize=224)
    lr, num_epochs = 0.002, 5
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    train.train_ch5(net, train_iter, test_iter,
                    batch_size, optim, device, num_epochs)
