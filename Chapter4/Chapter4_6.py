# coding=utf-8

import torch
from torch import nn

"""
这一节介绍了如何在GPU上进行机器学习计算
"""

# 书中所说的查看GPU信息在Windows环境下一般通过进入到NVIDIA Corporation\NVSMI
# 目录中然后调用nvidia-smi即可显示
# 然后再Pytorch中我们通过调用下面的函数来查看当前GPU的信息

# cuda是否可用
print(torch.cuda.is_available())
# GPU数量
print(torch.cuda.device_count())
# 当前GPU的索引号和信息
print(torch.cuda.current_stream())
# 当前GPU的名称
print(torch.cuda.get_device_name())
print('————————————————————————————')


# 默认情况下Tensor是在CPU和内存中计算的，我们需要将其迁移到GPU和显存上来加速
x = torch.tensor([1, 2, 3])
print(x)
# cuda函数将tensor迁移到GPU上
x = x.cuda(0)
print(x)
print(x.device)
print('————————————————————————————')


# 也可以在一开始就指定好将tensor创建到GPU上,这样速度比迁移快很多
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
x = torch.tensor([3, 2, 1], device=device)
# 或者
x = torch.tensor([3, 2, 1]).to(device)
print(x)
# 对GPU上的元素的计算结果仍然在GPU上
y = x**2
print(y)
# 下面这里会报错：不能把GPU的元素和CPU的元素混用计算
# z=y+x.cpu()
print('————————————————————————————')


# 我们一样可以将网络移动到GPU上，也要保证模型输入的tensor也在GPU上才能计算
net = nn.Linear(3, 1)
net.cuda()
print(list(net.parameters())[0].device)
x = torch.rand(2, 3).cuda()
print(net(x))
print('————————————————————————————')
