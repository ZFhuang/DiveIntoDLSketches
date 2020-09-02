# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import train, plot,data_process

import torch
import math

"""
这一节介绍了优化学习率只减不增缺陷的优化器RMSprop
"""


# 上一节的AdaGrad的缺点是学习率只减不增，这导致了在训练后期Ada可能因为学习率太小无法跳
# 出局部最小值从而难以找到有用的解，RMSprop简单对Ada加了一个参数就解决了问题，下面照
# 例进行一下测试
def rmsprop_2d(x1, x2, s1, s2):
    # 关键是增加了参数gamma，这个参数使得梯度很小的时候s2不一定大于s1，从而让学习率可
    # 能发生回升从而可能跳出局部最优
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


eta, gamma = 0.4, 0.9
plot.show_trace_2d(f_2d, train.train_2d(rmsprop_2d))
print('————————————————————————————')

# 下面是从零开始的完整实现
features, labels = data_process.get_data_ch7()


# 初始化参数
def init_rmsprop_states():
    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return (s_w, s_b)


def rmsprop(params, states, hyperparams):
    # 关键在于超参数gamma的引入
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        # 新的s的生成带有了gamma的加权
        s.data = gamma * s.data + (1 - gamma) * (p.grad.data)**2
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)


# 新的训练可以看到即使在训练的末尾线条也可能出现一些波动了
train.train_ch7(rmsprop, init_rmsprop_states(), {'lr': 0.01, 'gamma': 0.9},
                features, labels)
print('————————————————————————————')

# 简洁实现可以直接调用库中的RMSprop优化器，但是这里的gamma变为了alpha超参数
train.train_pytorch_ch7(torch.optim.RMSprop, {
                        'lr': 0.01, 'alpha': 0.9}, features, labels)
print('————————————————————————————')
