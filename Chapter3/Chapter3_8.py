# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import plot
import numpy as np
import torch
import matplotlib.pylab as plt

"""
这一节介绍了多层感知机的基本概念和三个常见的激活函数，都是tensor自带的函数
"""

# ReLU函数(rectified linear unit)，修复线性单元，(0,+∞)
# 先写一个可以绘制xy值的绘图函数xyplot然后使用
# 这里的x值是从-8到8以0.1为间隔排列出来的tensor
# 这里通过调用relu函数直接得到x经过relu处理后的值并作为y使用
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# relu函数也就是只取正值，负值取0的函数, ReLu(x)=max(x,0)
y = x.relu()
# plot.xyplot(x,y,'relu')
# 然后试着对这个relu求导并展示，由于x函数y=x,斜率为1，其余情况为0，因此导数有明显断层
y.sum().backward()
# plot.xyplot(x,x.grad,'grad of relu')

# Sigmoid函数，(0,1)
# 以前很常用，现在被ReLU取代，特点是负值趋向0，正值趋向1，0附近接近线性变换，中点1/2
y = x.sigmoid()
# plot.xyplot(x,y,'sigmoid')
# Sigmoid的导数形状类似正态分布，顶点为(0,0.25)
x.grad.zero_()
y.sum().backward()
# plot.xyplot(x,x.grad,'grad of sigmoid')

# Tanh函数，(-1,1)
# 是形状类似sigmoid的函数，但是这个函数0的时候结果是0，基于原点对称
y = x.tanh()
# plot.xyplot(x,y,'tanh')
# 其导数也类似sigmoid的结果，但是这次顶点是(0,1)
x.grad.zero_()
y.sum().backward()
# plot.xyplot(x,y,'grad of tanh')
