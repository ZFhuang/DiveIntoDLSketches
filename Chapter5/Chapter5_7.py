# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import data_process
from d2lzh_pytorch import train
from d2lzh_pytorch import layers

import torch
from torch import nn, optim

"""
这一节介绍了使用重复元素的网络VGG
"""

# 加上这个限定才能支持多线程读取
if __name__ == "__main__":
    # 设置计算设备，让计算在GPU上进行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # VGG的特色就是VGG块，VGG块时数个填充1，窗口3*3的卷积层加上一个步幅2的2*2最大池化
    # VGG网络利用VGG块堆积小卷积的特性保证有相同感受野的情况下提升了网络的深度
    # 下面这里先定义一下VGG块的生成函数，因为下面会重复用到VGG块
    def vgg_block(num_convs, in_channels, out_channels):
        block = []
        # 重复多次的3*3卷积
        for i in range(num_convs):
            # 这里要对应好通道数，且利用padding=1使得输出尺寸不变
            if i == 0:
                block.append(nn.Conv2d(in_channels, out_channels,
                                       kernel_size=3, padding=1))
            else:
                block.append(nn.Conv2d(out_channels, out_channels,
                                       kernel_size=3, padding=1))
            block.append(nn.ReLU())
        # 2*2的最大池化层
        block.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # 最后用单星号将这个由Module组成的列表拆包给下面的Sequential
        # 顺便，双星号是打包的意思，可以接收单星号的拆包操作
        return nn.Sequential(*block)

    # 然后来实现网络本身，网络是动态搭建的，由于经典款式包含了8个卷积层3个全连接层
    # 因此经常被称为VGG11
    def VGG_11(conv_arch, fc_features, fc_hidden_units=4096):
        # 先初始化框架
        net = nn.Sequential()
        # 根据输入的列表形参数动态加载新的vgg块，块内参数都是列表中的内容
        # 这里我们外部加入经典的8个卷积
        for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
            net.add_module("vgg_block_"+str(i+1),
                           vgg_block(num_convs, in_channels, out_channels))
        # 然后加入和AlexNet相同的全连接部分，3个全连接层，中间加有ReLU激活和丢弃层
        net.add_module("fc", nn.Sequential(
            layers.FlattenLayer(),
            nn.Linear(fc_features, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, 10)
        ))
        return net

    # VGG的卷积部分的内部构造列表，第一个参数是卷积层的重复次数
    conv_arch = (
        (1, 1, 64),
        (1, 64, 128),
        (2, 128, 256),
        (2, 256, 512),
        (2, 512, 512)
    )
    # 卷积连接到全连接层时的尺寸，需要手动计算出来
    # 每次应用一个VGG块，尺寸减半一次，因此尺寸此时为224/32=7，512则是最后的通道数
    fc_features = 512*7*7
    # 这里控制了每个全连接层内部的隐藏层连接数量，数量任意，越大越复杂
    fc_hidden_units = 4096

    # 然后这里来实例化VGG网络并显示
    net = VGG_11(conv_arch, fc_features, fc_hidden_units)
    # 构造一个1*224*224的样本来观察输出的情况,并且这里要用前面提到的嵌套输出
    X = torch.rand(1, 1, 224, 224)
    # 因为VGG本身很庞大复杂，这样的输出比较直观
    for name, block in net.named_children():
        X = block(X)
        print(name, 'ouput shape: ', X.shape)
    print('————————————————————————————')

    # 实际读取数据进行测试，由于这个网络太复杂，这里构造一个卷积小一些的网络来测试
    # 比率，全部参数都进行一定的缩放来缩小网络
    ratio = 8
    conv_arch = (
        (1, 1, 64//ratio),
        (1, 64//ratio, 128//ratio),
        (2, 128//ratio, 256//ratio),
        (2, 256//ratio, 512//ratio),
        (2, 512//ratio, 512//ratio)
    )
    fc_features = 512*7*7//ratio
    fc_hidden_units = 4096//ratio
    # 实例化新的网络并打印看看
    net = VGG_11(conv_arch, fc_features, fc_hidden_units)
    # 构造一个1*224*224的样本来观察输出的情况,并且这里要用前面提到的嵌套输出
    X = torch.rand(1, 1, 224, 224)
    # 因为VGG本身很庞大复杂，这样的输出比较直观
    for name, block in net.named_children():
        X = block(X)
        print(name, 'ouput shape: ', X.shape)
    print('————————————————————————————')

    # 训练的处理和AlexNet部分相同，即使缩小网络后耗时依然很长，看一个epoch即可
    batch_size = 256
    train_iter, test_iter = data_process.load_data_fashion_mnist(
        batch_size, resize=224)
    lr, num_epochs = 0.001, 5
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    train.train_ch5(net, train_iter, test_iter,
                    batch_size, optim, device, num_epochs)
