# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import data_process
from d2lzh_pytorch import layers
from d2lzh_pytorch import train
from d2lzh_pytorch import plot

import torch
import pandas as pd
import torch.nn as nn
import numpy as np


"""
这一节是这一章的总结和实战kaggle上的房价预测数据集，介绍了机器学习实践的一整套流程
"""

# 打印torch版本和设置好tensor的默认格式简化接下来的代码
print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

"""
读取数据
"""
# 从kaggle下载数据后放到想要的文件夹中，然后用pandas读取数据
# 接下来的数据集操作都需要用到pd作为接口
train_data = pd.read_csv('./Datasets/KaggleHouse/train.csv')
test_data = pd.read_csv('./Datasets/KaggleHouse/test.csv')
# 输出的时候会发现训练集多了一行，那是数据标签
# 然后所有数据的第一行是数据的id，也要舍弃
print(train_data.shape)
print(test_data.shape)
# 根据上面的注释删去部分内容并把测试集与训练集合并
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:-1]))

"""
预处理数据
"""
# 这里要对特征进行标准化，将特征的分布统一起来方便学习
# 首先得到所有特征的下标排列
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# 然后利用一个lambda对下标的每个特征都进行一次标准化
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x-x.mean())/(x.std()))
# 最后由于此时特征的均值已经是0了，所以缺失值可以用0填补
all_features = all_features.fillna(0)
# 然后再将数据集中的离散值转化为指示值，简单的说就是数据集中很多数据其内容是一个标签
# 此时我们通过自动将标签排列出来然后每个数据用01来指示属于哪个标签，从而方便训练
# 这里使用的是独热编码pandas.get_dummies,处理后可以看到特征维度变高了很多
# dummy_na参数是控制是否要把nan也作为一种编码判断
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)
# 最后将数据转换为tensor方便后面的训练
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(
    train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

"""
训练模型
"""
# 这里使用线性回归模型和平方损失函数来训练模型测试
# 平方误差，也就是每个结果的误差平方化然后求和
loss = torch.nn.MSELoss()


# 根据输入参数的数量生成一个网络
def get_net(feature_num):
    # 只用了一个线性层
    net = nn.Linear(feature_num, 1)
    # 正态分布随机初始化参数
    for p in net.parameters():
        nn.init.normal_(p, mean=0, std=0.01)
    return net


# 由于比赛的评价标准是对数均方根误差，所以定义一个对数均方根误差函数
def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设为1，使取对数时更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        # 对数输入的平方误差求平均值乘两倍再开方
        rmse = torch.sqrt(2*loss(clipped_preds.log(), labels.log()).mean())
    return rmse.item()


# 训练函数，用Adam优化算法来处理
def train(net, trian_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(trian_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # 这次在这里使用Adam优化器,weight_decay是惩罚比率
    optim = torch.optim.Adam(params=net.parameters(
    ), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for e in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            # 清空梯度
            optim.zero_grad()
            l.backward()
            optim.step()
        train_ls.append(log_rmse(net, trian_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


"""
K折交叉验证
"""


# 写个3.11中介绍过的K折交叉验证函数，它是用来在真正测试前判断模型的泛化能力的
# 将数据集分为k组, 然后每次取其中一组用于验证, 剩下的用来训练, 这样一共可进行k轮
def get_k_fold_data(k, i, X, y):
    assert k > 1
    # 双斜杠是下取整的除法
    fold_size = X.shape[0]//k
    X_train, y_train = None, None
    # 对K折里的每一折
    for j in range(k):
        # slice的效果是截取范围内的一段，配合下面的下标运算符从数据中截取
        idx = slice(j*fold_size, (j+1)*fold_size)
        # 得到截取范围内的子tensor
        X_part, y_part = X[idx, :], y[idx]
        # K折中的参数i代表要取其中的那一折作为验证集
        if j == i:
            X_valid, y_valid = X_part, y_part
        # 其他折作为训练集使用，根据是否为空决定是初始化还是拼接
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    # 返回得到的训练集和验证集
    return X_train, y_train, X_valid, y_valid


# 确切进行K折验证，实际上就是进行了K次训练和验证，返回K次训练和验证的平均误差
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    # 对于每一种折法
    for i in range(k):
        # 得到k折的验证集和训练集
        data = get_k_fold_data(k, i, X_train, y_train)
        # 用X的特征数来生成一个对应大小的网络
        net = get_net(X_train.shape[1])
        # 将数据应用到上面的训练函数中，*data自动将四个返回值展开输入到参数中
        train_ls, valid_ls = train(
            net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        # -1指取出最后一格元素
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        # 绘图第一折的误差曲线，并输出每一折的训练结果
        if i == 0:
            plot.semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'rmse',
                          range(1, num_epochs+1), valid_ls, ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f' %
              (i, train_ls[-1], valid_ls[-1]))
    # 返回误差总和的平均值，valid部分可以较好地衡量泛化误差
    return train_l_sum/k, valid_l_sum/k


"""
训练，选择模型，提交结果
"""
# 修改这里的超参数，重复这段直到找到一个满意的超参数再到下面测试
# 注意要让验证误差也一起减少而不只是训练误差减少，那样的话是过拟合
k, num_epochs, learning_rate, weight_decay, batch_size = 5, 150, 5, 0, 32
# train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs,
#                           learning_rate, weight_decay, batch_size)
# print('%d-fold validation: avg train rmse %f, avg valid rmse %f' %
#       (k, train_l, valid_l))


# 写一个预测函数并将训练与预测结果打包成可以提交到kaggle的格式
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    # 适配网络
    net = get_net(train_features.shape[1])
    # 进行训练，这里是使用了前面猜测好的超参数进行的
    train_ls, _ = train(net, train_features, train_labels,
                        None, None, num_epochs, lr, weight_decay, batch_size)
    # 绘图和输出
    plot.semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    # detach用于得到当前计算图的变量
    preds = net(test_features).detach().numpy()
    # 保存到提交文档中
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./Datasets/KaggleHouse/submission.csv', index=False)


# 训练
train_and_pred(train_features, test_features, train_labels,
               test_data, num_epochs, learning_rate, weight_decay, batch_size)
