# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import train, plot, data_process

import torch

"""
这一节介绍了结合了动量法和RMSprop的优化器——Adam法，且进行了偏差修正
"""

# 从零开始实现Adam法，重点是按照书里的公式设置速度v和可变学习率s，且做好偏差修正
features, labels = data_process.get_data_ch7()


# 初始化参数
def init_adam_states():
    v_w, v_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(
        1, dtype=torch.float32)
    s_w, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(
        1, dtype=torch.float32)
    return ((v_w, s_w), (v_b, s_b))


def adam(params, states, hyperparams):
    # 有两个beta来控制过去信息和梯度变换所占的的权重
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        # 更新v和s
        v[:] = beta1 * v + (1 - beta1) * p.grad.data
        s[:] = beta2 * s + (1 - beta2) * p.grad.data**2
        # 使用记录下来的超参数时间步t来进行偏差修正，这是因为由于加权平均和极限的原因，
        # 只截取一部分值进行估算既准确又不准确，一开始初始化的辅助参数0可能会导致很大
        # 的偏差，尽管在计算链长的时候前面的值会被近似忽略，但是在数据刚开始计算的时候
        # 影响可能很大，而且难以用多次加权来处理
        # 对时间步的偏差修正大幅减少了前期数据对总体数据的影响，因而能得到更准确的移动
        # 加权平均的结果
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        # 应用到参数上
        p.data -= hyperparams['lr'] * v_bias_corr / \
            (torch.sqrt(s_bias_corr) + eps)
    # 更新超参数时间步
    hyperparams['t'] += 1


# 训练测试，现在的优化算法由于RMSprop部分使得下降比较稳定，又由于动量法部分下降很快
train.train_ch7(adam, init_adam_states(), {
                'lr': 0.01, 't': 1}, features, labels)
print('————————————————————————————')

# 简洁实现
train.train_pytorch_ch7(torch.optim.Adam, {'lr': 0.01}, features, labels)
print('————————————————————————————')
