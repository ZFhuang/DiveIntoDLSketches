# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import train, plot,data_process

import numpy as np
import torch
import time
from torch import nn,optim

"""
这一节详细介绍小批量随机梯度下降
"""

# 先从NASA的数据集进行读取，保存了get_data_ch7()函数
# 前面的返回值是特征，后面的是标签
features, labels = data_process.get_data_ch7()
print(features.shape)
print('————————————————————————————')


# 定义一个暂时的梯度下降函数，获取网络的每个参数，利用超参数'学习率'和梯度来改变参数
# state是新的超参数，暂时不用用到
def sgd(params, state, hyperparams):
    for p in params:
        p.data -= hyperparams['lr']*p.grad.data

# 然后编写并保存本章的训练函数train_ch7


# 再包装一个训练sgd的接口函数并调用
def train_sgd(lr, batch_size, num_epochs=2):
    train.train_ch7(sgd, None, {'lr': lr}, features,
                    labels, batch_size, num_epochs)


# 批大小和数据集大小相同时，是传统梯度下降，速度慢但方向稳定
train_sgd(1, 1500, 6)
# 批大小只有1时，是随机梯度下降，方向不稳定，且耗时较长，但下降快且省内存
train_sgd(0.01, 1)
# 批大小是中间数值时，在运行速度与内存与周期下降速度中折衷选取,这里达到了又快又准
train_sgd(0.05, 10)
print('————————————————————————————')

# 使用库的简单实现，效果与之前类似
train.train_pytorch_ch7(optim.SGD, {"lr": 0.05}, features, labels, 10)
print('————————————————————————————')
