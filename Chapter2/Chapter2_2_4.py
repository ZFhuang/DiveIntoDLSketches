# coding=utf-8

# 用这个来导入PyTorch包
import torch

"""
这一节介绍了运算的内存消耗问题
"""

# view，索引之类的操作不会改变对象所指向的内存
x = torch.tensor([4, 3])
y = torch.tensor([3, 4])
print(y)
# 判断是否指向同一块内存最简单的方法就是使用id函数，id相同的对象内存指向相同
id_old = id(y)
print('————————————')
# 像这个加法会将结果存到新内存中并返回给y，也就是此时的y和之前已经不是同一个y了
y = x+y
# 结果会是False
print(id_old == id(y))
print('————————————')

# 如果想指向相同的内存，第一种方法是将结果存到索引中
y[:] = x+y
# 另一种方法是用不同的加法操作，例如自增，原址版本，修改add输出为止
y += x
y.add_(x)
torch.add(x, y, out=y)
