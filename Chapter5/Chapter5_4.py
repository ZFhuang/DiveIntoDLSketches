# coding=utf-8

import torch
from torch import nn

"""
这一节介绍了神经网络的池化层，池化本质类似于采样操作
"""


# 这里首先实现了一个池化函数pool2d
def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    # 预分配池化后结果矩阵Y，矩阵的尺寸和最基础的卷积公式一致
    Y = torch.zeros(X.shape[0]-p_h+1, X.shape[1]-p_w+1)
    # 池化核的中心和卷积核相同，都以左上角元素为中心坐标
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 扫描所有范围内元素，根据池化模式来选择所需的结果存入
            if mode == 'max':
                Y[i, j] = X[i:i+p_h, j:j+p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i+p_h, j:j+p_w].mean()
    return Y


X = torch.arange(9, dtype=torch.float).view(3, 3)
# 这里尝试使用平均池化
print(pool2d(X, (2, 2), 'avg'))
print('————————————————————————————')


# 和卷积一样，池化也有自己的填充和步幅设置
# 这里先建立一个顺序分配的矩阵作为测试
X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
print(X)
# 然后这次调用nn自带的池化层函数来进行,默认情况下填充为0，步幅和池化核大小相等
pool2d = nn.MaxPool2d(3)
# 因此这里输出元素只有一个10
print(pool2d(X))
# 但是当然我们可以自己指定想要的填充和步幅，操作和卷积层一样
pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
# 因此这里输出元素只有一个10
print(pool2d(X))
print('————————————————————————————')


# 在处理多通道数据的时候，池化层不该被输入和输出的通道数，因为它对各个通道池化后不合并
X=torch.cat((X,X+1),dim=1)
print(X.shape)
pool2d=nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d(X).shape)
print('————————————————————————————')
