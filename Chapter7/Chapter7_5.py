# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import train, plot,data_process

import torch
import math

"""
这一节介绍了优化统一的学习率难以适应复杂的维度信息的情况而生的梯度下降器AdaGrad
"""


# 要解决的核心问题是动量法的学习率是恒定且应用在全局上的，导致了动量法需要选择合适的学习
# 率，对应不同尺度的维度学习率可能并不能很好地调整参数，且常需要用较小的学习率进行学习以
# 保证训练末尾的时候不会因为学习率过大而震荡，但这个调整又会使得一开始的收敛变慢。因此
# AdaGrad设置了会自适应减小的学习率来保证开始和结尾速度合适，且对每个参数都独立设置学习
# 率保证能适应调整不同尺度的参数


def adagrad_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2*x1, 4*x2, 1e-6
    s1 += g1**2
    s2 += g2**2
    x1 -= eta/math.sqrt(s1+eps)*g1
    x2 -= eta/math.sqrt(s2+eps)*g2
    return x1, x2, s1, s2


def f_2d(x1, x2):
    return 0.1*x1**2+2*x2**2


# 先模拟下AdaGrad的效果，可以看到收敛的步距是在缩小的
# 而且即使学习率很大也正确且快速地逼近了目标
eta = 2
# plot.show_trace_2d(f_2d, train.train_2d(adagrad_2d))
print('————————————————————————————')

# 这里开始从零开始实现AdaGrad优化器，重点是维护控制学习率的变量
# 这里使用大一些的数据来测试
features, labels = data_process.get_data_ch7()


# 初始化参数
def init_adagrad_states():
    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return (s_w, s_b)


def adagrad(params, states, hyperparams):
    eps = 1e-6
    # 对每个参数对应的学习率按公式设置然后应用到参数上
    for p, s in zip(params, states):
        # 按照梯度大小提高s的值，也就是减小学习率，使用了平方使得s只增不减
        s.data += (p.grad.data**2)
        # 用类似学习率的用法来使用s，s作为学习率的权重
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)


# 对比上节的动量法可以看到这次下降很快而且收敛后也很稳定
train.train_ch7(adagrad, init_adagrad_states(), {'lr': 0.1}, features, labels)
print('————————————————————————————')

# 简洁实现只要使用库函数即可
train.train_pytorch_ch7(torch.optim.Adagrad, {'lr': 0.1}, features, labels)
print('————————————————————————————')
