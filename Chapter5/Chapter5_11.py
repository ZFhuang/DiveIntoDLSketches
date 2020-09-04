# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import data_process, train, layers

import time
import torch
from torch import nn, optim
import torch.nn.functional as F

"""
这一节介绍了深度神经网络的大突破：残差神经网络ResNet
"""

# 加上这个限定才能支持多线程读取
if __name__ == "__main__":
    # 设置计算设备，让计算在GPU上进行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 保存一下残差块layers.Residual，重点是在前向传播函数中提供了绕路的选项
    
    # 下面测试一下残差块的运行，没有旁路的话残差块的输入和输出通道必须相等
    blk = layers.Residual(3, 3)
    X = torch.rand((4, 3, 6, 6))
    print(blk(X).shape)
    # 有旁路卷积的话就可以调整输入输出的尺寸，因为旁路有对准输入输出的效果
    # 旁路卷积也可以不开启，那样的话残差块的输入输出必须相等
    blk = layers.Residual(3, 6, use_1x1_conv=True, stride=2)
    print(blk(X).shape)
    print('————————————————————————————')

    # 首先和前面类似，定义一个动态返回残差块层的生成函数
    def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
        # 对于第一个残差块要定论输入通道和输出通道相同
        if first_block:
            assert in_channels == out_channels
        blk = []
        # 然后用循环来决定这个残差块层要包含多少个残差块
        for i in range(num_residuals):
            # 这里要根据当前是第几个残差块来决定通道大小的问题
            if i == 0 and not first_block:
                blk.append(layers.Residual(in_channels, out_channels,
                                    use_1x1_conv=True, stride=2))
            else:
                blk.append(layers.Residual(out_channels, out_channels))
        # 和VGG的时候类似，返回序列化后的残差块链
        return nn.Sequential(*blk)

    # 我们可以用Sequential简单地构造这次的网络，但是这样得到的网络命名不清晰
    net = nn.Sequential(
        # 前几层和GoogLeNet一样
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        # 然后加入残差块组，记得要标记第一组残差块且输入输出要相同
        resnet_block(64, 64, 2, first_block=True),
        resnet_block(64, 128, 2),
        resnet_block(128, 256, 2),
        resnet_block(256, 512, 2),
        # 再加入全局池化层和全连接层收尾
        layers.GlobalAvgPool2d(),
        layers.FlattenLayer(),
        nn.Linear(512, 10)
    )

    # 下面用分步add的方法构造，看起来比较丑但是输出的时候名称清晰
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", layers.GlobalAvgPool2d())
    net.add_module("fc", nn.Sequential(
        layers.FlattenLayer(),
        nn.Linear(512, 10)
    ))

    X = torch.rand(1, 1, 224, 224)
    for name, layer in net.named_children():
        X = layer(X)
        print(name, ' output shape: ', X.shape)
    print('————————————————————————————')

    # 最后类似之前，进行一下测试，这里也减小了图片大小
    # 1epoch = 189.7sec
    batch_size = 256
    train_iter, test_iter = data_process.load_data_fashion_mnist(
        batch_size, resize=96)
    lr, num_epochs = 0.001, 5
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    train.train_ch5(net, train_iter, test_iter,
                    batch_size, optim, device, num_epochs)
