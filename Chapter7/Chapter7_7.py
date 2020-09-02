# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import train, plot,data_process

import torch
import math

"""
这一节介绍了另一种优化学习率只减不增缺陷的优化器AdaDelta，其特点是没有学习率这一超参数
"""

# AdaDelta在RMSprop的基础上额外维护了状态变量delta，这个变量用来替代超参数学习率，让梯
# 度对参数的影响更加自由，让学习率从初始化的时候就是利用梯度学习而来

features, labels = data_process.get_data_ch7()


def init_adadelta_states():
    s_w, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(
        1, dtype=torch.float32)
    # 需要额外初始化参数delta，用来调整临时梯度的值
    delta_w, delta_b = torch.zeros(
        (features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    return ((s_w, delta_w), (s_b, delta_b))


def adadelta(params, states, hyperparams):
    # 这里的rho相当于AdaGrad的gamma
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        s[:] = rho * s + (1 - rho) * (p.grad.data**2)
        # 对临时梯度使用了更复杂的更新方法，关键是使用了(delta + eps)来代替之前所需的
        # 学习率
        g = p.grad.data * torch.sqrt((delta + eps) / (s + eps))
        p.data -= g
        # delta在使用后也得进行更新,更新方法类似于对中间值s的更新
        delta[:] = rho * delta + (1 - rho) * g * g


train.train_ch7(adadelta, init_adadelta_states(),
                {'rho': 0.9}, features, labels)
print('————————————————————————————')

# 简洁实现依然是使用库已经写好的Adadelta,库的版本一开始的时候loss就比较低了
train.train_pytorch_ch7(torch.optim.Adadelta, {'rho': 0.9}, features, labels)
