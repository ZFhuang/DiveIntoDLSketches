# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import train, plot

import numpy as np
import torch
import math

"""
这一节重新开始详细介绍和实验梯度下降相关的算法
"""


# 写一个一维的梯度下降函数进行测试，这里假定目标函数是x**2，因此导数是2*x
# 这里的eta是一个比较小的值，也就是学习率，代表了往梯度方向移动的步伐大小
def gd(eta):
    # 设置初始值
    x = 10
    results = [x]
    for i in range(10):
        # 梯度下降，
        x -= eta*2*x
        # 将下降过程中的x值记录下来
        results.append(x)
    print('epoch 10, x:', x)
    return results


res = gd(0.2)


# 绘制出x的下降轨迹
def show_trace(res):
    # 设置绘制框的大小
    n = max(abs(min(res)), abs(max(res)), 10)
    # 画线时的尺度
    f_line = np.arange(-n, n, 0.1)
    plot.set_figsize()
    # 绘制x**2的线
    plot.plt.plot(f_line, [x * x for x in f_line])
    # 绘制结果线
    plot.plt.plot(res, [x * x for x in res], '-o')
    plot.plt.xlabel('x')
    plot.plt.ylabel('f(x)')
    plot.plt.show()


show_trace(res)
print('————————————————————————————')

# 控制梯度下降速率的eta就是学习率，一般需要开始时由人工设定,不同的学习率有不同效果
# 太小的学习率会下降得很慢
show_trace(gd(0.05))
# 太大的学习率会震荡
show_trace(gd(1.1))
print('————————————————————————————')

# 而对于梯度下降的多维模式，需要用到偏导数来计算
# 由于直接求偏导就是往各个轴向的变化率，用方向导数来指定所有可能方向上的变换率
# 由于导数*单位方向u，就是导数的模*梯度与方向的夹角的cos，因此cos取-1最好
# 最终下降公式也是x<-x-eta*delta(f(x)),这里二维梯度下降和下降过程的绘制函数都保存了
# 二维梯度下降train_2d(trainer)，绘制下降曲线show_trace_2d(f, results)
eta = 0.1


def f_2d(x1, x2):  # ⽬标函数
    return x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0)


plot.show_trace_2d(f_2d, train.train_2d(gd_2d))
print('————————————————————————————')

# 随机梯度下降的测试，由于随机梯度下降一般采样一个局部来计算梯度估计总体，因此梯度不准
# 这里用均值为0的正态分布随机噪声来模拟随机梯度下降的错误梯度
# 尽管路线会出现抖动，但是仍然能到达目标，而且迭代速度快了很多


def sgd_2d(x1, x2, s1, s2):
    return (x1 - eta * (2 * x1 + np.random.normal(0.1)), x2 -
            eta * (4 * x2 + np.random.normal(0.1)), 0, 0)


plot.show_trace_2d(f_2d, train.train_2d(sgd_2d))
print('————————————————————————————')
