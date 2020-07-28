# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
import numpy as np
import torch
from d2lzh_pytorch import data_process
from d2lzh_pytorch import train

"""
这一节从零开始实现了一个多层感知机
"""

# 仍然使用Fashion-MNIST数据集
batch_size = 256
train_iter, test_iter = data_process.load_data_fashion_mnist(batch_size)

# 初始化模型参数
num_featrues, num_outputs, num_hiddens = 28*28, 10, 256
# 要利用矩阵叉乘的特性将参数数量连接起来，
W1 = torch.tensor(np.random.normal(
    0, 0.01, (num_featrues, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(
    0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)
# 将参数组成tensor
params = [W1, b1, W2, b2]
for p in params:
    # 批量设置参数的梯度设置
    p.requires_grad_(requires_grad=True)


def relu(X):
    # ReLU函数的设置
    return torch.max(input=X, other=torch.tensor(0.0))


def net(X):
    # 把X扁平化
    X = X.view((-1, num_featrues))
    # 用ReLU函数对隐藏层的预测进行激活
    H = relu(torch.matmul(X, W1)+b1)
    # 然后在应用输出层的参数
    return torch.matmul(H, W2)+b2


# 定义交叉熵损失的函数指针
loss = torch.nn.CrossEntropyLoss()

# 用之前定义好的train_ch3函数进行训练
num_epochs, lr = 5, 100.0
train.train_ch3(net, train_iter, test_iter, loss,
                  num_epochs, batch_size, params, lr)
