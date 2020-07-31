# coding=utf-8

import torch
from torch import nn

"""
这一节介绍了数据储存和读取的相关内容
"""

# save和load函数可以存储和读取Tensor，列表，字典等等
x = torch.ones(3)
# 存到文件夹目录中
torch.save(x, 'x.pt')
# 读取
x2 = torch.load('x.pt')
print(x2)
print('————————————————————————————')


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


# 对于一个神经网络，我们可以用state_dict函数得到内部可以学习的层的参数
net = MLP()
print(net.state_dict())
# 类似的，优化器的参数也可以经由此得到
optim = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(optim.state_dict())
print('————————————————————————————')


# 保存网络有两种方法：1.仅保留参数；2.保存整个网络
# 建议只保存参数减小空间加快加载,保存和加载的方法和上面类似
# 这里的路径建议以pt或pth作为后缀
PATH = 'net.pth'
X = torch.randn(2, 3)
Y = net(X)
torch.save(net.state_dict(), PATH)
# 注意这里读取的时候需要先初始化一个相同的网络然后再读取
net2 = MLP()
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)
print(Y == Y2)
print('————————————————————————————')
