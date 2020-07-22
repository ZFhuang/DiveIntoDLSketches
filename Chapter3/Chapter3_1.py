# coding=utf-8

import torch

"""
这一节介绍了线性回归的基本概念和神经网络图
"""

x = torch.ones(100)
y = torch.ones(100)
z = torch.zeros(100)
# 两个向量相加一种方法是用循环逐元素相加,这种做法繁琐且速度极慢
for i in range(100):
    z = x[i]+y[i]

# 另一种做法是直接做向量加法,这种做法速度快很多，所以在计算的时候最好将计算向量化
z = x+y

# 多运用广播机制可以简化代码的编写(见2.2)
a = torch.ones(10)
b = 2
print(a+b)
print('————————————')
