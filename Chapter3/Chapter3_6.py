# coding=utf-8


import torch
import torchvision
import numpy as np
import sys
sys.path.append(r'.')
from d2lzh_pytorch import data_process
from d2lzh_pytorch import softmax
from d2lzh_pytorch import plot


"""
这一节从零开始实现了一个Softmax离散回归
"""

# 这里要用前面的Fashion Minist数据集测试，调用3.5节封装好的函数
batch_size = 256
train_iter, test_iter = data_process.load_data_fashion_mnist(batch_size)

# 初始化模型的参数
num_features = 28*28
num_outputs = 10
W = torch.tensor(np.random.normal(
    0, 0.01, (num_features, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs,dtype=torch.float)
W.requires_grad_(True)
b.requires_grad_(True)

# 对tensor的操作可以指对其中一部分进行，然后保留其他的部分
X=torch.tensor([[1,2,3],[4,5,6]])
# 例如下面这里只求和了各个列
print(X.sum(dim=0, keepdim=True))
# 例如下面这里只求和了各个行
print(X.sum(dim=1, keepdim=True))

# 3.4中介绍到softmax的核心是输出一个向量，表示各个对应下标的可能性值，取最大作为结果
# 每个可能性值都是输出值求指数exp后，与全体输出值求占比得到的
# 这里为了测试先随机了一组标签输出值模拟，然后softmax对其进行指数平均
X = torch.rand((2, 5))
X_prob = softmax.softmax(X)
print(X_prob, X_prob.sum(dim=1))


# 然后这里自定义了一个softmax分类网络模型，也是线性回归的一种
# 注意下面需要先调整好矩阵相乘的矩阵大小，然后再进行计算
def net(X):
    return softmax.softmax(torch.mm(X.view(-1, num_features), W)+b)


# 3.4节介绍的交叉熵损失函数，交叉熵损失也就是标签向量中那个标记为1的标签对应的
# 概率的log值之和求平均后越大越好，越大说明预测越接近总体正确。下面是简单的模拟
# 下面y_hat是2个样本在3个类别的预测概率,和为1，y是这两个样本的标签类别
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# LongTensor是long格式的tensor
y = torch.LongTensor([0, 2])
# view中的-1意思是根据其他维度的值推断这一维的值，此处相当于view(2,1)
# gather的用处是根据第一个参数指定的维度(0是按竖行查找替换，1是按横行查找替换)
# 再按照第二个参数指定的索引配置将y_hat中的元素进行重新安置返回，第二个参数与结果维度相同
# 此处指定索引是1，也就是按横行索引替换，然后y正好是列向量，意味着结果是和y一样索引的
# 也就是y_hat第1行的0号元素放置在结果列的第0个，y_hat第2行的2号元素放置在结果列的第1个
print(y_hat.gather(1, y.view(-1, 1)))


# 利用上面的工具就可以写出交叉熵损失的函数
def cross_entropy(y_hat, y):
    # 这里先用gather法得到了对应的预测值，然后求log
    # 需要取负值是因为我们希望这个预测值变大，也就是希望优化的损失变小
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))


# 下面准备计算分类准确率的函数accuracy()
# 对于预测值，取其预测向量中最大可能性的那个元素的索引和y比较是否相同
# 然后对所有值取平均得到相同值在所有样本中所占的比率
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# 这里输出可以看到，之前的预测值里取最大后y_hat预测了[2,2]
# y是[0,2],所以预测正确的有一半，也即是0.5
print(accuracy(y_hat, y))
# 对测试集的准确率用前面完全随机的模型进行测试，约0.05的准确率
print(data_process.evaluate_accuracy(test_iter, net))

# 真正来训练softmax模型，仍然采用前面的小批梯度下降算法来优化
# 迭代次数
epoch_num = 5
# 学习率
learning_rate = 0.03
# 调用写好的train_ch3对这个softmax回归进行训练，注意这里带入了交叉熵作为代价函数
softmax.train_ch3(net, train_iter, test_iter, cross_entropy,
                  epoch_num, batch_size, [W, b], learning_rate)

# 最后对训练好的网络来实际预测一下看看真实的效果
# 随意从test中取一组数据出来
X, y = iter(test_iter).next()
# 从下标值取出真实的标签名
true_labels = data_process.get_fashion_mnist_labels(y.numpy())
pred_labels = data_process.get_fashion_mnist_labels(
    net(X).argmax(dim=1).numpy())
# 设置好绘制的时候的标题，用一个数组内循环解决
# zip函数将可迭代的对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
# 所以这里用zip把后面的标签数组转成了标签组形式
titles = [true + '\n'+pred for true, pred in zip(true_labels, pred_labels)]
# 将对比图绘制出来
plot.show_fashion_mnist(X[0:9], titles[0:9])
