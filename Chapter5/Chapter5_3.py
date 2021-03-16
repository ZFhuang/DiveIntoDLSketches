# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import layers

import torch
from torch import nn

"""
这一节介绍了多通道的卷积
"""


# 首先是多个输入通道的情况，这种情况下卷积核和数据需要同样的通道数
# 各层之间分别卷积后再求和得到新的单通道输出
def corr2d_multi_in(X, K):
    res = layers.corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        # 此处各层分别卷积并求和，这种写法兼容了单通道
        res += layers.corr2d(X[i, :, :], K[i, :, :])
    return res


X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
print(corr2d_multi_in(X, K))
print('————————————————————————————')


# 多通道输入且输出的情况，此时由于我们有三维的输入，因此需要k个三维的卷积核，也就是4维
# 这里每个卷积核按照上面的方法进行一次卷积，然后多个输出通道的结果连接起来
# 连接的时候注意越往深层越是索引排在前面
def corr2d_multi_in_out(X, K):
    # 此处将每个输出通道的核都进行了一次卷积并组合起来，由于K是四维，所以每个k是三维
    return torch.stack([corr2d_multi_in(X, k) for k in K])


K = torch.stack([K, K+1, K+2])
print(K.shape)
print(corr2d_multi_in_out(X, K))
print('————————————————————————————')


# 1*1*k卷积层，其作用与全连接层等价，对每个元素多个通道进行一次操作然后累加输出
def corr2d_multi_in_out_1x1(X, K):
    # 第一个参数是输入通道
    c_input, h, w = X.shape
    # 第一个参数是输出通道
    c_out = K.shape[0]
    # 将数据打平为向量组
    X = X.view(c_input, h*w)
    # 将核打平为向量组
    K = K.view(c_out, c_input)
    # 利用矩阵相乘来完成向量组相乘从而实现网络的计算
    Y = torch.mm(K, X)
    # 返回的时候再调整结果为合适的尺寸
    return Y.view(c_out, h, w)


# 一个3*3的三通道输入
X = torch.rand(3, 3, 3)
# 一个1*1的3通道输入2通道输出的核
K = torch.rand(2, 3, 1, 1)
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
# 这里输出很接近0，说明1*1卷积效果与全连接是相同的
print((Y1-Y2).norm().item())
print('————————————————————————————')
