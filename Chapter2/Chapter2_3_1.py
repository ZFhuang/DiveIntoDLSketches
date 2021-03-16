# coding=utf-8

import torch

"""
这一节介绍了如何在PyTorch中求梯度
"""

# 我们在机器学习中，需要一层一层地对一个tensor不断施加操作
# 操作累计的梯度前向传播然后完成后再反向传播回来完成梯度计算
# 首先要对需要计算的tensor设置requires_grad为true，设置后它表示这个tensor的需要计算梯度
x = torch.rand(5, 5, requires_grad=True)
print(x)

# tensor一般是经过某种函数运算得到的，如果一个tensor是某个Function类的计算结果
# 那么其grad_fn属性会返回其应用的函数对象，None代表还未应用过，直接创建的tensor就是None
print(x.grad_fn)
print('————————————')

# 下面这里由于x是y+x生成的结果，所以requires_grad会记录AddBackward
y = torch.rand(5, 5, requires_grad=True)
x = y+x
print(x.grad_fn)
print('————————————')
