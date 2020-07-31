# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
from torch import nn
import torch
import sys
sys.path.append(r".")
from d2lzh_pytorch import layers


"""
这一节介绍了神经网络的二维卷积
"""

# 在layers中写好测试用的卷积层函数，然后下面是效果
X = torch.tensor([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])
K = torch.tensor([
    [0, 1],
    [2, 3]
])
print(layers.corr2d(X, K))
print('————————————————————————————')


# 上面实现的是二维卷积的函数，这里基于这个函数将其实现为一个卷积层
# 卷积核的大小就是卷积层的名字
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        # 随机初始化权重核偏差
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # 前向传播就是卷积并加上偏差
        return layers.corr2d(x, self.weight)+self.bias


# 卷积的经典用途就是边缘检测核，这里首先初始化一个黑白分明的目标矩阵,左右1，中间0
X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)
print('————————————————————————————')


# 然后[1,-1]是最简单的边缘检测核，他将横向相同的部分置0，不同的部分检测为1和-1
K = torch.tensor([[1, -1]])
Y = layers.corr2d(X, K)
# 因此卷积操作可以处理局部相关的空间
print(Y)
print('————————————————————————————')


# 由于上面有了边缘检测核，这里再试试如何利用这些数据学习出这个核
# 首先目标要学习的就是这个卷积层的参数，核大小设置核上面的检测核相同
conv2d = Conv2D(kernel_size=(1, 2))
step = 30
lr = 0.01
for i in range(step):
    # 预测(前向传播)
    Y_hat = conv2d(X)
    # 平方误差
    l = ((Y_hat-Y)**2).sum()
    # 反向传播
    l.backward()
    # 手动进行梯度下降
    conv2d.weight.data -= lr*conv2d.weight.grad
    conv2d.bias.data -= lr*conv2d.bias.grad
    # 记得梯度清零
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    # 每5次迭代输出一次效果
    if (i+1) % 5 == 0:
        print('Step %d, loss %.3f' % (i+1, l.item()))

print('weight:', conv2d.weight.data)
print('bias:', conv2d.bias.data)
print('————————————————————————————')
