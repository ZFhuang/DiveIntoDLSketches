# coding=utf-8

import torch

"""
这一节介绍了如何将tensor在CPU和GPU间移动
"""

# 主要是用to函数来进行移动
# 在移动前需要先判断一下cuda是否可用
# 通常情况下我们创建的tensor是处于CPU上的
if torch.cuda.is_available():
    x = torch.ones(5, 3, device="cuda")
    y = x.to(torch.device("cpu"))

print(x)
print(y)
print('————————————')
