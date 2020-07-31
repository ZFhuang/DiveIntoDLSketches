# coding=utf-8

import torch
from torch import nn

"""
这一节介绍了除了使用现有的层外，如何自定义神经网络的层
"""


class CenteredLayer(nn.Module):
    # 首先定义不含模型参数的自定义层,网络层也是继承自Module类的，通过forward函数计算
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        # 这层是一个简单的归一化层
        return x-x.mean()


layer = CenteredLayer()
print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)))
# 也可以和之前类似将自定义层和别的层串起来使用
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
y = net(torch.rand(4, 8))
# 由于进行均值归一化层，所以值很接近0
print(y.mean().sum())
print('————————————————————————————')

# 还可以自定义含有可学习的模型参数的层，前面说过Parameter类是tensor的子类
# 因此如果又一个tensor是Parameter的话，那么它就会被自动加入到模型的参数列表中
# 因此将自定义的参数定义为Parameter是关键，一种方法是4.2提到的直接定义Parameter
# 另外的方法就是用ParameterList和ParameterDict


class MyListDense(nn.Module):
    def __init__(self):
        super(MyListDense, self).__init__()
        # 这里使用了ParameterList，接收一个Parameter列表，下面就是构造了三个元素
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        # 还可以用append继续添加新的参数
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        # 经过ParameterList的参数可以被索引，这里通过索引然后矩阵乘法来完成前向传播
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x


net = MyListDense()
print(net)
print('————————————————————————————')


class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        # 这里使用了ParameterDict，接收一个Parameter字典，然后规则就和字典一致了
        # 例如可以用update新增参数，用keys得到键值，用items返回键值对
        # 要记得字典是由大括号括起来的，列表则是中括号,且键值对用冒号连接
        self.params = nn.ParameterDict({
            # key 和 value
            'linear1': nn.Parameter(torch.randn(4, 4)),
            'linear2': nn.Parameter(torch.randn(4, 1))
        })
        # 这里用update()继续添加新的参数
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))})

    def forward(self, x, choice='linear1'):
        # 这里前向传播时可以选择要进行的层,因此可以通过传入键值不同来进行不同的传播
        return torch.mm(x, self.params[choice])


net = MyDictDense()
print(net)
x = torch.ones(1, 4)
# 这里对输入针对性使用了某一层，但是这里需要手动连接多层
print(net(x, 'linear1'))
print(net(x, 'linear2'))
print(net(x, 'linear3'))
print('————————————————————————————')

# 这里串联了两层
net = nn.Sequential(
    # 字典层默认只应用了第一层
    MyDictDense(),
    # 列表层则完整前向传播了
    MyListDense()
)
print(net)
print(net(x))
