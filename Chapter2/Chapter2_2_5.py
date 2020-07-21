# coding=utf-8

# 用这个来导入PyTorch包
import torch
# 用这个来导入Numpy包
import numpy

"""
这一节介绍了tensor和numpy互相转换的操作
"""

# numpy是python非常常用的一个数学计算包
# tensor转numpy数组
a = torch.ones(2, 5)
b = a.numpy()
print(a)
print(b)
print('————————————')

# numpy数组转tensor，用这个方法转换后的数组与原数据共享内存
c = torch.from_numpy(b)
# torch.tensor()则会发生拷贝
d = torch.tensor(b)
print(c)
print(d)
print('————————————')
