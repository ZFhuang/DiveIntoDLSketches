# coding=utf-8

import torch
from torch import nn

"""
这一节介绍了构造网络模型的细节, 非常关键的一节
"""


# 通过继承nn.Module可以自定义出自己想要的模型，此时生成的网络是固定的
# 这种写法比较常见
class MLP(nn.Module):
    # 重载__init__和forward函数可以创建模型参数和定义前向计算
    # __init__声明了带有参数的层和模型的结构
    def __init__(self, **kwargs):
        # 通过传递参数给父类来初始化模型
        super(MLP, self).__init__(**kwargs)
        # 然后定义了模型的各种变量例如各种计算层，用变量名做key
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)

    # 然后重载forward函数来指定如何进行正向的计算，将前面定义好的层连接起来
    # forward函数会在step的时候被调用, 通过函数的嵌套调用来完成一步的计算
    # 反向传播不需要定义，因为系统会根据自动求梯度自己生成一个
    def forward(self, x):
        return self.output(self.act(self.hidden(x)))


# 然后实例化net来进行计算即可, 传入net的参数会到达forward处
X = torch.rand(2, 784)
net = MLP()
print(net)


# 也可以通过接收一系列参数的方法动态初始化网络
class MySquential(nn.Module):
    from collections import OrderedDict

    def __init__(self, *args):
        super(MySquential, self).__init__()
        # 构造函数中带有一参数指针，如果仅有一个参数且其格式是有序字典OrderedDict
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            # 遍历字典中的项，此时项应该是各层网络的形容，将其加入网络,用序号做key
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            # 否则代表传入的是一些Module，将其直接加入即可
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        # 前向传播时顺序遍历网络中的所有Module并应用到input上直到出结果
        # 这种写法使得网络只能是序列化的
        for module in self._modules.values():
            input = module(input)
        return input


net = MySquential(
    # 此处输入的就是Module参数组形式
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print(net)


# 还可以使用ModuleList来初始化网络，此时通过反复的append来改进网络
# 初始化的时候使用的是列表，自动用序号做key
net = nn.ModuleList([
    nn.Linear(784, 256),
    nn.ReLU()
])
net.append(nn.Linear(256, 10))
print(net)


# 还可以用ModuleDict类接收一个模块字典作为输入，然后也可以类似字典一样添加新项
# 用字符串做key
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU()
})
net['output'] = nn.Linear(256, 10)
print(net)


# 试着利用第一种方法构造一个更加复杂的网络
class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        # 内部定义一组随机值，关闭梯度计算从而不会被训练改变
        self.rand_weight = torch.rand(20, 20, requires_grad=False)
        # 定义一个线性层
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        # 前向传播过程，先应用线性层，然后激活，然后再应用之前的线性层
        X = self.linear(X)
        # 激活层这里让值与前面的随机权重进行了相乘并加一，表示我们可以自定义relu
        X = nn.functional.relu(torch.mm(X, self.rand_weight.data)+1)
        # 这里发生了层的复用，从而参数会被共享
        X = self.linear(X)
        # 通过判断输出值的数量来对结果进行微调
        while X.norm().item() > 1:
            X /= 2
        if X.norm().item() < 0.8:
            X *= 10
        return X.sum()


X = torch.rand(2, 20)
net = FancyMLP()
print(net)


# 然后是尝试了对这些继承于Module的网络进行嵌套使用
class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        # 这里可以看到内部开启了一个新的网络
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())

    def forward(self, x):
        # 每次调用就会自动序列执行子类的forward
        return self.net(x)


# 外部也嵌套了一次网络
net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())

x = torch.rand(2, 40)
print(net)
print(net(x))
