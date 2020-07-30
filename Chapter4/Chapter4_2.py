# coding=utf-8

import torch
from torch import nn
from torch.nn import init

"""
这一节介绍了模型参数的访问，初始化和共享是如何运用的
"""

# Sequential会完成参数的默认初始化
net = nn.Sequential(
    nn.Linear(4, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)
print(net)
print('————————————————————————————')


X = torch.rand(2, 4)
Y = net(X).sum()
# 我们可以通过Module类的parameters或named_parameters来访问网络的参数(迭代器)
# named_parameters还会返回参数的名字
# 这里显示是生成器(迭代器)
print(type(net.named_parameters()))
# 这里会打印出所有参数的名字和大小，带有索引前缀
for name, param in net.named_parameters():
    print(name, param.size())
print('————————————————————————————')


# 对于Sequential的网络，可以用下标定位其某一层，然后打印层内参数的内容，此时没有前缀
for name, param in net[0].named_parameters():
    print(name, param.size())
print('————————————————————————————')


class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # 如果⼀一个Tensor是Parameter,那么它会⾃自动被添加到模型的参数列列表⾥
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        # 因此weight2没有加入参数列表中
        self.weight2 = torch.rand(20, 20)

    def forward(self, x):
        pass


n = MyModel()
for name, param in n.named_parameters():
    # 因此这里不会打印出weight2
    print(name)
print('————————————————————————————')


# 参数也可以读取其data，grad之类
weight_0 = list(net[0].parameters())[0]
print(weight_0.data)
print(weight_0.grad)
Y.backward()
print(weight_0.grad)
print('————————————————————————————')


# Pytorch中的Module的参数都采取了较为合理的初始化策略，但是有时候我们需要改变策略
# 此时可以如下使用init函数来改变初始化
for name, param in net.named_parameters():
    if 'weight' in name:
        # 正态分布初始化
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)
    if 'bias' in name:
        # 常数初始化
        init.constant_(param, val=0)
        print(name, param.data)
print('————————————————————————————')


# 我们还可以自定义参数初始化的方法
def init_weight_(tensor):
    # 首先不要计算梯度，with保护上下文
    with torch.no_grad():
        # 首先均匀分布[-10,10]
        tensor.uniform_(-10, 10)
        # 然后这行代码的意思是将绝对值大于5的值保留，剩下的值置0，类似于掩码操作
        tensor *= (tensor.abs() >= 5).float()


for name, param in net.named_parameters():
    if 'weight' in name:
        # 自定义初始化
        init_weight_(param)
        print(name, param.data)
print('————————————————————————————')


# 然后如果想要再多个层中共享模型的参数，之前4.1有过重复调用同一层的操作
# 还有一种情况是如果我们把通过Module传进Sequential，此时参数也会是共享的
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
# 打印的时候会发现两层只对应一层参数
for name, param in net.named_parameters():
    if 'weight' in name:
        init.constant_(param, val=3)
        print(name, param.data)
# 而且查看ID也会发现两层对应同一块内存
print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))
print('————————————————————————————')


# 进行计算的话会发现反向传播的时候梯度会被计算两次，一次是3，两次是6
x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad)
