# coding=utf-8

import torch
from torch import nn

"""
这一节介绍了使用神经网络时会发生的填充问题和步幅问题
"""

# 这里定义了一个卷积层，前两个参数"通道"在下一节才会提到，重点是核大小和填充参数
# 这里的padding指的是在各个边缘都添加一行/列
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)


def comp_conv2d(conv2d, X):
    # 给当前矩阵增加两维(批量和通道)，是空扩展，本质上还是只有当前的值
    X = X.view((1, 1)+X.shape)
    Y = conv2d(X)
    # 应用卷积后的结果省略掉头两维:批量和通道
    return Y.view(Y.shape[2:])


X = torch.rand(8, 8)
# 卷积后的边长n-k+2p+1=8-3+1*2+1=8
print(comp_conv2d(conv2d, X).shape)
print('————————————————————————————')


# 卷积层的核和填充可以是不正方的
conv2d = nn.Conv2d(in_channels=1, out_channels=1,
                   kernel_size=(5, 3), padding=(2, 1))
# 卷积后的行长n-k+2p+1=8-5+2*2+1=8
# 卷积后的列长n-k+2p+1=8-3+1*2+1=8
print(comp_conv2d(conv2d, X).shape)
print('————————————————————————————')


# 增加了步幅参数stride
conv2d = nn.Conv2d(in_channels=1, out_channels=1,
                   kernel_size=3, padding=1, stride=2)
# 卷积后的边长(n-k+2p+s)/s=(8-3+1*2+2)/2=4
print(comp_conv2d(conv2d, X).shape)
print('————————————————————————————')


# 最复杂综合的卷积
conv2d = nn.Conv2d(in_channels=1, out_channels=1,
                   kernel_size=(3,5), padding=(0,1), stride=(3,4))
# 除不完的情况，默认向下取整
# 卷积后的行长(n-k+2p+s)/s=(8-3+0+3)/3=取2
# 卷积后的列长(n-k+2p+s)/s=(8-5+1*2+4)/4=取2
print(comp_conv2d(conv2d, X).shape)

# 这部分网上有另一个公式n=[(n-k+2p)/s]+1,向下取整
