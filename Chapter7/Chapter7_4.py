# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import train, plot,data_process

import torch

"""
这一节介绍了能够自适应调整梯度下降方向的动量法,动量法利用了之前参数变化的过程作为辅助
"""

# 学习率，控制变化的速度
eta = 0.4


def f_2d(x1, x2):
    return 0.1*x1**2+2*x2**2


# f_2d的求导，作为简化的基本梯度下降来表示
def gd_2d(x1, x2, s1, s2):
    return (x1-eta*0.2*x1, x2-eta*4*x2, 0, 0)


# 将需要显示的目标函数f_2d绘制出来，然后利用下降函数进行训练并绘制轨迹
# 这里可以看到折线波动着收敛，但是在横向的移动上速度很慢，且一旦学习率大于0.5就无法收敛
# plot.show_trace_2d(f_2d, train.train_2d(gd_2d))
print('————————————————————————————')


# 动量法可以解决这个问题，让其在竖直方向上不再是那么粗暴的折线且让水平方向更快逼近结果
# 动量法的特点是拥有一个内部的变量"速度v"，它会根据当前的参数值不断做出调整，然后又影
# 响对参数x1x2的调整项从而影响了接下来的参数的调整，其实际上是利用之前时间步中的参数状
# 态来影响后面的参数，作为一个权重改变参数变化的速度和方向
def momentum_2d(x1, x2, v1, v2):
    # 动量的超参数gamma==0时，相当于小批随机梯度下降，式子和gd_2d相同
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2


# 动量法不但加速了收敛，稳定了计算，还提高了学习率设置的容错率
eta, gamma = 0.4, 0.5
# plot.show_trace_2d(f_2d, train.train_2d(momentum_2d))
print('————————————————————————————')

# 下面开始从零开始实现动量法，关键是维护一个和参数形状相同的速度变量
features, labels = data_process.get_data_ch7()


# 初始化各个内置的速度变量并返回
def init_momentum_states():
    v_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    v_b = torch.zeros(1, dtype=torch.float32)
    return (v_w, v_b)


# 速度也作为一个超参数传入训练中，改变了数据更新的方法，思路和前面一致
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        # 更新速度
        v.data = hyperparams['momentum'] * \
            v.data + hyperparams['lr'] * p.grad.data
        # 更新参数
        p.data -= v.data


# # 综合上面，超参数动量的用途是改变速度的更新速度，动量越大速度受到之前的影响就越大
# train.train_ch7(sgd_momentum, init_momentum_states(), {
#                 'lr': 0.02, 'momentum': 0.5}, features, labels)
# # 这里将动量改为0.9使速度更不容易被梯度改变，直观上让收敛的曲线角度变化更不容易改变，
# # 这会使得前期变化很快，后期不容易稳定
# train.train_ch7(sgd_momentum, init_momentum_states(), {
#                 'lr': 0.02, 'momentum': 0.9}, features, labels)
# # 由于速度的改变关键是学习率*梯度，因此如果减小学习率则会使得后期由于强大的动量影响，
# # 微小的梯度改变不容易影响速度，但相应的也减慢了刚开始的收敛速度
# train.train_ch7(sgd_momentum, init_momentum_states(), {
#                 'lr': 0.004, 'momentum': 0.9}, features, labels)
print('————————————————————————————')

# 动量法在库中可以很容易被调用，只要对优化器指定名为动量的超参数即可
train.train_pytorch_ch7(
    torch.optim.SGD, {'lr': 0.004, 'momentum': 0.9}, features, labels)
