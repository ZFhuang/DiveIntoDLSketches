# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import data_process
from d2lzh_pytorch import layers
from d2lzh_pytorch import softmax
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import torch
from torch.nn import init


"""
这一节介绍了如何用torch的神经网络模组来实现3.6的softmax回归
"""


# 初始化参数和读取数据
batch_size = 256
train_iter, test_iter = data_process.load_data_fashion_mnist(batch_size)
num_features = 28*28
num_outputs = 10


class LinearNet(nn.Module):
    # 初始化网络，仍然只用一个单层的线性回归即可
    def __init__(self, num_features, num_outputs):
        super(LinearNet, self).__init__()
        # 对比3.3的线性回归网络，最大的差别就是这里有了多个输出点
        self.linear = nn.Linear(num_features, num_outputs)
    
    # 前向传播
    def forward(self, x):
        # 线性平坦化层，将X变为一维的状态
        # 这部分被单独定义为一个类FlattenLayer了
        y = self.linear(x.view(x.shape[0], -1))
        return y


# 从类初始化网络
net = LinearNet(num_features, num_outputs)

# 3.3提到的序列化组合网络方法，利用到了提取出来的平坦层类FlattenLayer
net = nn.Sequential(OrderedDict([
    ('flatten', layers.FlattenLayer()),
    ('linear', nn.Linear(num_features, num_outputs))
]))

# 初始化网络的参数,仍然是正态初始化和常数初始化
init.normal_(net.linear.weight, 0, 0.01)
init.constant_(net.linear.bias, 0)

# 损失函数指针的设置，使用交叉熵损失
loss = nn.CrossEntropyLoss()

# 优化器的设置，使用之前使用的SGD，网络参数和学习率输入到优化器中
learning_rate = 0.03
optim = torch.optim.SGD(net.parameters(), learning_rate)

# 开始训练模型,使用3.6编写的训练模板
num_epochs = 5
# 注意这里由于有了优化器所以不需要在这里输入网络参数和学习率了
softmax.train_ch3(net, train_iter, test_iter, loss,
                  num_epochs, batch_size, None, None, optim)
