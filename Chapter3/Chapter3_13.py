# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import plot
from d2lzh_pytorch import data_process
from d2lzh_pytorch import train
from d2lzh_pytorch import layers

import torch
import numpy as np
import torch.nn as nn

"""
这一节介绍了另一种解决过拟合问题的方法"丢弃法"
"""

# 从零开始实现丢弃法
# 丢弃法的特点是在训练的时候按照一定概率把各个结点的权重清零或增大


# 这里首先来实现丢弃函数
def dropout(X, drop_prob):
    # 设置权重属性，确认丢弃概率合法
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1-drop_prob
    # 保留概率为0时提前返回以免导致下面的错误
    if keep_prob == 0:
        return torch.zeros_like(X)
    # 用正态随机得到初始值，按照是否小于保留概率来二值化，然后浮点化方便下面计算
    mask = (torch.randn(X.shape) < keep_prob).float()
    # 返回矩阵点对点乘完成丢弃，然后除保留概率得到扩张部分的值
    return mask*X/keep_prob


# # 测试一下丢弃函数
# X = torch.arange(16).view(2, 8)
# print(dropout(X, 0))
# print(dropout(X, 0.5))
# print(dropout(X, 1))

# 然后为了实战效果，这里再试试用Fashion-MNIST数据集,首先来设置参数
# 这一次的网络将有两个隐藏层，正常情况下很可能过拟合，但是因为有丢弃法所以不会
num_features, num_outputs, num_hiddens1, num_hiddens2 = 28*28, 10, 256, 256
W1 = torch.tensor(np.random.normal(0, 0.01, size=(
    num_features, num_hiddens1)), dtype=torch.float, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, size=(
    num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, size=(
    num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)
params = [W1, b1, W2, b2, W3, b3]
drop_prob1, drop_prob2 = 0.2, 0.5


# 然后定义模型，注意这次有两个隐藏层，为了结果的稳定性在测试的时候不进行丢弃
def net(X, is_training=True):
    X = X.view(-1, num_features)
    # 输入层连接隐藏层1
    H1 = (torch.matmul(X, W1)+b1).relu()
    # 判断是否在训练，丢弃只发生在隐藏层
    if is_training:
        H1 = dropout(H1, drop_prob1)
    # 隐藏层1连接隐藏层2
    H2 = (torch.matmul(H1, W2)+b2).relu()
    if is_training:
        H2 = dropout(H2, drop_prob2)
    # 隐藏层2连接输出层
    return torch.matmul(H2, W3)+b3


# 修改了一下之前的正确率计算函数，然后进行测试
num_epochs, lr, batch_size = 5, 100.0, 256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter = data_process.load_data_fashion_mnist(batch_size)
# train.train_ch3(net, train_iter, test_iter, loss,
#                 num_epochs, batch_size, params, lr)

# 然后一样来尝试一下用网络模块来实现丢弃法,丢弃法就是增加一个nn.Dropout()层
net = nn.Sequential(
    layers.FlattenLayer(),
    nn.Linear(num_features, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(num_hiddens2, num_outputs)
)

# 训练并测试
optim = torch.optim.SGD(net.parameters(), lr=0.5)
train.train_ch3(net, train_iter, test_iter, loss,
                num_epochs, batch_size, None, None, optim)
