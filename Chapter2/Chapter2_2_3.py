# coding=utf-8

# 用这个来导入PyTorch包
import torch

"""
这一节介绍了tensor的广播broadcasting机制
"""

# 当两个形状不同的tensor进行按元素计算时可能会触发广播机制
# 广播会适当复制元素让两个tensor的形状相同然后再计算
# 两个 Tensors 只有在下列情况都符合下才能进行 broadcasting 操作：
# 每个 tensor 至少有一维
# 遍历所有的维度，从尾部维度开始，每个对应的维度大小要么相同，要么其中一个是1，要么其中一个不存在
# arange函数会得到[i,j)的排列
x = torch.arange(1, 61).view(5, 3, 4, 1)
print(x)
print('————————————')
# 可广播，因为从后往前看维度，维度非零，且对应x的维度中要么相同要么其中一个是1或者不存在
y = torch.empty(3, 1, 2)
# 不可广播，因为第一个维度2与x的第二个维度3不匹配
y = torch.empty(2, 1, 2)
# 可广播，可以不存在多个维度，只要从后往前直到不存在为止都符合上面规则即可
y = torch.empty(1)
print(y)
print('————————————')
# 广播的时候，若维度不同，则维度较小的维度补1让维度相同，然后各个维度选取数字大的作为最终维度
# 没有元素的部分被填0
print(x+y)