# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import torch
from torch import nn
from torch.nn import init
import numpy as np

import sys
sys.path.append(r".")
from d2lzh_pytorch import data_process
from d2lzh_pytorch import layers
from d2lzh_pytorch import train


"""
这一节也是试一下用神经网络模块来简洁实现多层感知机
"""

# 多层感知机与之前softmax的区别就是多了个隐藏层
num_features, num_outputs, num_hiddens = 28*28, 10, 256
# 网络从上到下逐级深入
net = nn.Sequential(
    # 先用一层扁平层把输入的参数打平
    # 扁平层主要意义是代替卷积层，相当于直接感受所有特征
    layers.FlattenLayer(),
    # 然后用线性的全连接层连接到隐藏层
    nn.Linear(num_features, num_hiddens),
    # 然后用ReLU激活
    nn.ReLU(),
    # 再连接到全连接的输出层
    nn.Linear(num_hiddens, num_outputs),
)
# 对net中的每个参数都进行正态分布初始化
for p in net.parameters():
    init.normal_(p, 0, 0.01)
# 设置与读取
batch_size = 256
train_iter, test_iter = data_process.load_data_fashion_mnist(batch_size)
loss = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(),0.5)
# 进行训练
num_epochs = 5
train.train_ch3(net, train_iter, test_iter, loss,
                num_epochs, batch_size, None, None, optim)
