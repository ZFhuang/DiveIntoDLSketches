# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import plot
from d2lzh_pytorch import linear_reg

import torch
import numpy as np
import torch.nn as nn

"""
上一节可见过拟合是比较棘手的问题，这一节介绍了处理过拟合问题的一个方法"权重衰减"
"""

# 同样的先生成样本数据，依据公式y=0.05+sigma(i=1:p)(0.01xi)+ε
# 这里为了让效果变得明显，考虑使用200维的回归但训练样本只有20个，测试却有100个
num_train, num_test, num_features = 20, 100, 200
real_w, real_b = torch.ones(num_features, 1), 0.05
features = torch.randn(num_test+num_train, num_features)
# 用矩阵乘法来生成标签,这是很常用的做法
labels = torch.matmul(features, real_w)+real_b
# 加噪声
labels += torch.tensor((np.random.normal(0, 0.01, size=labels.size())),
                        dtype=float)
train_features, test_features = features[:num_train,
                                         :], features[num_train:, :]
train_labels, test_labels = labels[:num_train], labels[num_train:]


# 下面从零开始实现权重衰减的效果
# 首先还是一样需要初始化模型的参数
def init_params():
    # 会初始化生成正态分布的w和全0的b，然后返回，记得每个参数都要开启梯度
    w = torch.randn((num_features, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# L2惩罚项的公式，求出权重的平方和然后除2
def l2_penalty(w):
    return (w**2).sum()/2


# 训练和测试
batch_size, num_epochs, lr = 1, 100, 0.003
# 用之前写的线性函数即可，然后用平方误差作为代价函数
net, loss = linear_reg.linreg, linear_reg.squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, True)


# 参数lambd是衰减比率，越大则惩罚越强，越难以拟合
def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            # 这里的损失项要加上前面写好的L2惩罚项,注意乘上lambd好控制衰减比率
            l = loss(net(X, w, b), y)+lambd * l2_penalty(w)
            l = l.sum()
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            # 进行梯度下降
            linear_reg.sgd([w, b], lr, batch_size)
        # 记录每次训练的loss
        train_ls.append(loss(net(train_features, w, b),
                             train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b),
                            test_labels).mean().item())
    # 绘制和输出
    print('L2 norm of w:', w.norm().item())
    plot.semilogy(range(1, num_epochs+1), train_ls, 'epochs',
                  'loss', range(1, num_epochs+1), test_ls, ['train', 'test'])


# # lambd为0时，惩罚项完全无效，此时过拟合，会看到训练误差远低于泛化误差
# fit_and_plot(0)
# # 打开权重衰减，会发现效果好了很多，输出的权重也小了很多
# fit_and_plot(3)


# 这里尝试用神经网络模块来简洁实现L2惩罚拟合，这里的惩罚系数为wd
def fit_and_plot_pytorch(wd):
    net = nn.Linear(num_features, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    # weight_decay就是附加在损失函数上的惩罚系数,修改这个超参数会改变衰减比率
    optim_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd)
    # 不对bias进行惩罚
    optim_b = torch.optim.SGD(params=[net.bias], lr=lr)
    # 训练
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            # 具体的训练和之前的做法都一样
            l = loss(net(X), y).mean()
            optim_w.zero_grad()
            optim_b.zero_grad()
            l.backward()
            # 反向传播后就让优化器迭代
            optim_w.step()
            optim_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    plot.semilogy(range(1,num_epochs+1), train_ls, 'epochs', 'loss',
                  range(1,num_epochs+1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net.weight.data.norm().item())


# 同样进行测试和对比，结果和基础状态一致
fit_and_plot_pytorch(0)
fit_and_plot_pytorch(3)
