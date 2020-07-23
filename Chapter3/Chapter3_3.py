# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import *
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import torch
from collections import OrderedDict
from torch.nn import init
import torch.optim as optim


"""
这一节用库函数简化了3.2的线性回归过程
"""

# 首先生成和3.2一样的样例样本，1000个样本，每个样本2个特征和1个标签
num_features = 2
num_examples = 1000
features = torch.from_numpy(np.random.normal(
    0, 1, (num_examples, num_features)))
real_w = [2, -3.4]
real_b = 4.2
labels = real_w[0]*features[:, 0]+real_w[1]*features[:, 1]+real_b
e = np.random.normal(0, 0.01, size=labels.size())
labels += e

batch_size = 10
# 用data包来加载数据集
dataset = Data.TensorDataset(features, labels)
# 读取batch的工具,设置shuffle=True让batch读取时是随机的
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
# 这里data_iter的用法同样是用for读取即可
for X, y in data_iter:
    # print(X, y)
    break

# 之前说到线性回归相当于一个单层神经网络，那么这里就来利用torch的神经网络模块来尝试实现
# torch的神经网络模块是torch.nn，nn的核心是Module，既可以表示某个层也可以表示一个网络
# 可以通过继承Module来自定义自己的网络/层类


class LinearNet(nn.Module):
    # 初始化函数，初始化时需要参数特征数量n_features
    def __init__(self, n_features):
        # 调用父类的初始化，传递自己
        super(LinearNet, self).__init__()
        # 只有一层的神经网络，而且是线性回归层，n_features->1
        self.linear = nn.Linear(n_features, 1)

    # 前向传播部分，每个nn.Module都必须有
    def forward(self, x):
        # 接收参数tensor:x，返回前向传播的结果y
        y = self.linear(x)
        return y


# 应用网络并打印网络的结构，会打印出网络中各层的排布
net = LinearNet(num_features)
print(net)

# 也可以通过 Sequential顺序的 生成所需的网络，在构造参数中写入网络计算图的结构
net = nn.Sequential(
    nn.Linear(num_features, 1)
)

# 也可以向现有的 Sequential 加入新的结构层,注意需要标记层的类型linear
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_features, 1))

# 还可以用有序表来初始化
# from collections import OrderedDict
net = nn.Sequential(OrderedDict([('linear', nn.Linear(num_features, 1))]))
print(net)

# net.parameters()可以返回网络模型参数的生成器，也就是可用for来查看网络的参数情况
# 注意前面的这些生成是自带了bias的
for p in net.parameters():
    # 这里会打印出两个特征权重和一个bias
    print(p)

# 注意torch只支持batch作为样本,如果想只输入一个样本,可用input.unsqueeze(0)来给样本增加一维

# 使用网络前，和之前一样需要初始化参数，在nn中的init模块中有多种初始化方法
# 调用正态分布的原址版本，将第0层的权重初始化为(0,0.01)的正态分布
init.normal_(net[0].weight, mean=0, std=0.01)
# 将第0层的bias设置为常数0，也可以直接修改data
init.constant_(net[0].bias, val=0)

# nn模块提供了多种损失函数，这里一样采用均方误差函数MSE并取函数指针
loss = nn.MSELoss()

# nn也提供了很多优化算法，这里一样采用随机梯度下降SGD算法
# 参数是前面从网络中可以得到的网络参数，学习率可以直接写入SGD参数中
# import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)
# 会打印出优化器的设置信息
print(optimizer)
# 我们可以对不同的子网络设置不同的学习率，内层覆盖外层，下面是一个接近的小尝试
optimizer = optim.SGD([
    {'params': net.parameters(), 'lr': 0.01}
], lr=0.03)
print(optimizer)

# 类似之前的做法来迭代训练模型
num_epochs = 3
for e in range(num_epochs):
    for X, y in data_iter:
        # 将X输入到网络中得到预测值y_hat
        # 这里由于一开始设置的不好导致需要进行两次类型转换
        y_hat = net(X.float()).double()
        # 计算误差，调用的是前面的MSE误差函数
        l = loss(y_hat, y.view(-1, 1))
        # 将梯度清零
        optimizer.zero_grad()
        # 误差的反向传播计算
        l.backward()
        # 让优化器自动开始下一个循环
        optimizer.step()
    print('epoch %d, loss: %f' % (e+1, l.item()))

# 比较最终结果,直接提取网络第0层的数据
result = net[0]
# weight成员会输出权重向量
print(real_w, ' ; ', result.weight)
# 偏置向量
print(real_b, ' ; ', result.bias)
