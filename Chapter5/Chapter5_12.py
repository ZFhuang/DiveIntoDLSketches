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
这一节介绍了ResNet的改进版本，DenseNet，它在残差旁路的处理上用相连替代了相加
"""

# 加上这个限定才能支持多线程读取
if __name__ == "__main__":
    # 设置计算设备，让计算在GPU上进行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DenseNet(稠密网络)的特点是在残差旁路的处理上用相连替代了相加
    # 因而网络可能在旁路的影响下快速增大，网络设计了连接用的稠密快和减小网络的过渡层
    # 首先我们来实现稠密块结构，稠密块由多个卷积块组成，是对前面残差块的改进
    def conv_block(in_channels, out_channels):
        # 这里的卷积结构是对ResNet卷积结构的改进，简化为归一，激活，卷积的结构
        blk = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        return blk

    class DenseBlock(nn.Module):
        # 稠密块由多个卷积块组成，每个块的输出通道数都一样
        def __init__(self, num_convs, in_channels, out_channels):
            super(DenseBlock, self).__init__()
            net = []
            for i in range(num_convs):
                # 但是这里用循环连接多个卷积块时，卷积块的输入被连接起来
                # 其实就是为了接收残差网络中旁路的X数据
                in_c = in_channels+i*out_channels
                # 新尺寸的块，接收上一层的数据和这一层刚输出的数据
                net.append(conv_block(in_c, out_channels))
            self.net = nn.ModuleList(net)
            # 计算最后的输出通道数，实际上就是原输入和n层输出之和
            self.out_channels = in_channels+num_convs*out_channels

        def forward(self, X):
            # 前向传播的时候不断将最新的输出与至今为止的输出拼接在一起输入给下一层
            for blk in self.net:
                Y = blk(X)
                X = torch.cat((X, Y), dim=1)
            # 输出会不停和上一层的输出拼接，因此输出尺寸会增大
            return X

    # 同样测试一下网络块的尺寸是否正确，这里由于稠密块的输出是不断增长的
    # 2个卷积层，输入通道=3，输出通道=10由于会被倍乘因此也被称为增长率
    blk = DenseBlock(2, 3, 10)
    X = torch.rand(4, 3, 8, 8)
    Y = blk(X)
    # [4, 23, 8, 8].这里输出23=(第一层:3+10(成为下一层输入))->(第二层:13+10)=>23
    print(Y.shape)
    print('————————————————————————————')

    # 从上面清楚知道稠密层会大幅增加通道数，因此过渡层用1*1卷积来减小通道数并缩图
    def transition_block(in_channels, out_channels):
        blk = nn.Sequential(
            # 批量归一稳定数据并激活
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            # 1*1卷积减少通道数
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            # 平均池化用stride来减半图片的大小
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        # 最终得到小很多的输入
        return blk

    blk = transition_block(23, 10)
    # [4, 10, 4, 4]，输出通道被卷积为10，图片宽高都被池化减半了
    print(blk(Y).shape)
    print('————————————————————————————')

    # 真正来实例化DenseNet模型，和上节一样用复杂的写法换来清晰的结构输出
    # 首先是和ResNet一致的基础特征提取
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    # 接下来用稠密层和过渡层来替换ResNet的残差层
    # num_channels是动态的输入通道数，growth_rate是输出通道数
    num_channels, growth_rate = 64, 32
    # 这里是各个稠密块这里设置的卷积层数
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    # 用循环得到各个稠密块应有的层数
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        # 初始化对应稠密块并加入网络
        DB = DenseBlock(num_convs, num_channels, growth_rate)
        net.add_module("DenseBlock_%d" % i, DB)
        # 更新新的输入通道数
        num_channels = DB.out_channels
        # 每加入一次稠密块就加上一层过渡层，这里用参数让过渡层把输出的通道数减半
        if i != len(num_convs_in_dense_blocks)-1:
            net.add_module("TransitionBlock_%d" %
                           i, transition_block(num_channels, num_channels//2))
            # 更新新的输入通道数
            num_channels = num_channels//2

    # 最后一次归一化和激活后用全局池化和全连接层收尾，输入尺寸都是动态变化的
    net.add_module("BatchNorm", nn.BatchNorm2d(num_channels))
    net.add_module("Relu", nn.ReLU())
    net.add_module("GlobalAvgPool", layers.GlobalAvgPool2d())
    net.add_module("FullConnnect", nn.Sequential(
        layers.FlattenLayer(),
        nn.Linear(num_channels, 10)
    ))

    # 同样测试网络各层的尺寸形状
    X = torch.rand(1, 1, 96, 96)
    for name, layer in net.named_children():
        X = layer(X)
        print(name, ' output shape: ', X.shape)
    print('————————————————————————————')

    # 最后和之前一样进行一下测试，这里也减小了图片大小，这个网络对显存要求很大
    # 1epoch = 138.2sec
    batch_size = 256
    train_iter, test_iter = data_process.load_data_fashion_mnist(
        batch_size, resize=96)
    lr, num_epochs = 0.001, 5
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    train.train_ch5(net, train_iter, test_iter,
                    batch_size, optim, device, num_epochs)
