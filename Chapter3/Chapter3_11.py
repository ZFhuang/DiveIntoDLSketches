# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import data_process
from d2lzh_pytorch import plot

import torch
import numpy as np

"""
这一节用代码模拟了多项式拟合中会出现的欠拟合和过拟合问题
"""

# 和前面做过的类似，自己模拟生成一组样本
# 目标是y=1.2x-3.4x^2+5.6x^3+5+ε
num_train, num_test, real_w, real_b = 100, 100, [1.2, -3.4, 5.6], 5
# 初始化样本，标准正态分布的随机数，相当于一次方形式的特征
features = torch.randn((num_train+num_test, 1))
# cat=concatnate, 将两个tensor合成，参数dim让结果是添加在下面一行
# pow对生成的特征进行次方操作，得到三次方拟合的特征
poly_features = torch.cat(
    (features, torch.pow(features, 2), torch.pow(features, 3)), 1)
# 组合特征得到样本标签
labels = (real_w[0]*poly_features[:, 0]+real_w[1] *
          poly_features[:, 1]+real_w[2]*poly_features[:, 2]+real_b)
# 加上噪声模拟真实数据
labels += torch.tensor(
    np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
# 随意查看几个样本
print(features[:2], poly_features[:2], labels[:2])

# 然后定义一个绘制对数尺度图的函数semilogy并且训练模型后绘制查看
num_epochs, loss = 100, torch.nn.MSELoss()

# 写一个训练并进行绘制的函数fit_and_plot
def fit_and_plot(train_features, test_features, train_labels, test_labels):
    # 写一个只有线性层的网络，打平连接到一个输出
    # 这个过程自动初始化了参数所以不用额外处理了
    net = torch.nn.Linear(train_features.shape[-1], 1)
    # 设置批大小，记得限制最小值
    batch_size = min(10, train_labels.shape[0])
    # 读取和参数设置
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    # 用下划线来标明这个循环的循环值是不需要用到的
    for _ in range(num_epochs):
        # 进行正常的训练和误差计算
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        # 把数据的标签打平变为方便处理的标签组
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        # 计算出预测的误差并和上面的标签组合起来
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
    # 打印结果
    print('final epoch: train loss',
          train_ls[-1], 'test loss', test_ls[-1])
    plot.semilogy(range(1, num_epochs+1), train_ls, 'epochs',
                  'loss', range(1, num_epochs+1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data, '\nbias:', net.bias.data)


# 尝试观察效果的区别
# 用三阶多项式函数进行拟合，模型复杂度正常，训练误差和泛化误差接近且都较低
fit_and_plot(poly_features[:num_train, :], poly_features[num_train:, :],
             labels[:num_train], labels[num_train:])

# 只用一阶多项式拟合，模型复杂度太低，训练误差和泛化误差难以下降，是欠拟合
fit_and_plot(features[:num_train, :], features[num_train:, :],
             labels[:num_train], labels[num_train:])

# 用三阶多项式对较少的数据进行拟合，模型复杂度太高，训练误差远低于泛化误差，是过拟合
fit_and_plot(poly_features[:2, :], poly_features[num_train:, :],
             labels[:2], labels[num_train:])
