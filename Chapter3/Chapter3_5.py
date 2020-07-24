# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import *
from d2lzh_pytorch import plot
from d2lzh_pytorch import data_process

import time
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
import torch


"""
这一节介绍了如何引入常用的多类图像分类数据集Fashion-MNIST
"""

# 通过torchvision从网上下载数据集,需要如下分别下载trian和test
# 这里的transform将下载下来的数据转为了Tensor方便使用，且此时图片的数据格式变为uint8
mnist_train = torchvision.datasets.FashionMNIST(
    root=r"./Datasets", train=True,
    download=True, transform=transforms.ToTensor())
# 同上操作的测试集，注意train=False
mnist_test = torchvision.datasets.FashionMNIST(
    root=r"./Datasets", train=False,
    download=True, transform=transforms.ToTensor())

# 可以用len得到数据集的大小，用type得到数据集的类型，用下标得到某个具体的数据
print(len(mnist_train), len(mnist_test))
feature, label = mnist_train[10]
# 能看到图片为1*28*28，是一维的灰度图
print(feature.shape, label)

# 编写好用索引获取标签真名的函数get_fashion_mnist_labels
# 和绘制数据图片的show_fashion_mnist后，下面来尝试显示一下
# 类似之前，这里也有特征X和标签y
X, y = [], []
for i in range(10):
    # 这个数据集的数据组织方式就是第0是特征数据，第1是标签
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
# 将这个得到的特征和标签转换为图片绘制
# plot.show_fashion_mnist(X, data_process.get_fashion_mnist_labels(y))

# 将读取数据的流程封装到load_data_fashion_mnist中使用
train_iter, _ = data_process.load_data_fashion_mnist(256)

# 测试下读取完数据所需的时间
start = time.time()
for X, y in train_iter:
    continue
print('%.2f s' % (time.time()-start))
