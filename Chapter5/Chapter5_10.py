# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import train, data_process,layers
import torch.nn.functional as F
import torch
import time
from torch import nn, optim


"""
这一节介绍了在训练途中的批量归一化，让训练和预测的时候中间数据更稳定从而让下降更有效
"""

# 加上这个限定才能支持多线程读取
if __name__ == "__main__":
    # 设置计算设备，让计算在GPU上进行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 从零开始实现批量归一化，首先是批量归一化层函数
    def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps,
                   momentum):
        # 判断当前的情况
        if not is_training:
            # 进行归一化，这里的moving_mean和moving_var是传进来的动态的参数
            # 而eps是一个很小的非零值防止除法出错，这样计算出归一化后的X矩阵
            X_hat = (X-moving_mean)/torch.sqrt(moving_var+eps)
        else:
            # 对于预测模式，限制输入是二或四维
            assert len(X.shape) in (2, 4)
            # 如果是二维代表是全连接层的输入，直接计算当前的平均值和方差
            if len(X.shape) == 2:
                mean = X.mean(dim=0)
                var = ((X-mean)**2).mean(dim=0)
            # 否则四维的情况是有多通道的卷积层的输入，这里要计算通道维上的均值和方差
            else:
                # 保持X不变，叠加计算输入通道和二维特征的平均值
                # 也就是求出各个输出通道自己的平均值(将其他维组合了)
                mean = X.mean(dim=0, keepdim=True).mean(
                    dim=2, keepdim=True).mean(dim=3, keepdim=True)
                # 根据上面的平均值求各个输出通道的方差
                var = ((X-mean)**2).mean(dim=0, keepdim=True).mean(
                    dim=2, keepdim=True).mean(dim=3, keepdim=True)
            # 同样计算出归一化后的X矩阵
            X_hat = (X-mean)/torch.sqrt(var+eps)
            # 然后这里利用动量方法预测出下一次调用的时候该有的平均值和方差
            # 这些个移动平均值和移动方差是根据现在的平均值和方差预测的，并不是准确的
            moving_mean = momentum*moving_mean+(1.0-momentum)*mean
            moving_var = momentum*moving_var+(1.0-momentum)*var
        # 结果是另一个仿射变换，gamma和beta是这个归一化层自己可以学习的参数
        # 代表了对归一化的拉伸和偏移，用来调整归一化的具体效果，例如有的地方不用归一化
        Y = gamma*X_hat+beta
        return Y, moving_mean, moving_var


    # 然后是建立一个批量归一化的类，负责代理批量归一化层
    class BatchNorm(nn.Module):
        def __init__(self, num_features, num_dims):
            super(BatchNorm, self).__init__()
            # 批量归一化是归一化到输出通道数上，因此关键是输出通道的特征数量
            if num_dims == 2:
                shape = (1, num_features)
            else:
                shape = (1, num_features, 1, 1)
            # 初始化几个参数
            self.gamma = nn.Parameter(torch.ones(shape))
            self.beta = nn.Parameter(torch.zeros(shape))
            # 这两个参数不参与梯度和迭代，它是由动量计算出来的
            self.moving_mean = torch.zeros(shape)
            self.moving_var = torch.zeros(shape)

        def forward(self, X):
            # 这里将两个特殊参数适配所需的内存位置
            if self.moving_mean.device != X.device:
                self.moving_mean = self.moving_mean.to(X.device)
                self.moving_var = self.moving_var.to(X.device)
            # 调用一次梯度归一化，计算得到的两个动态参数保留在类内，值返回出来
            Y, self.moving_mean, self.moving_var = batch_norm(
                self.training, X, self.gamma, self.beta, self.moving_mean,
                self.moving_var, eps=1e-5, momentum=0.9)
            return Y

    # 这里用最简单效果最差的LeNet来改进测试，在每次卷积或全连接后加入归一化
    net = nn.Sequential(
        nn.Conv2d(1, 6, 5),
        # 加入的归一化层不会改变输出通道数，保证相同即可，因此可以直接夹在中间
        # 这里的num_dims参数用来说明当前是卷积部分还是全连接部分，卷积部分是4维的
        BatchNorm(6, num_dims=4),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5),
        BatchNorm(16, num_dims=4),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2),
        # 这里打平特征进入了全连接部分
        layers.FlattenLayer(),
        nn.Linear(16*4*4, 120),
        # 因此这里的num_dims就变成了全连接层的二维
        BatchNorm(120, num_dims=2),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        BatchNorm(84, num_dims=2),
        nn.Sigmoid(),
        nn.Linear(84, 10)
    )

    # 用老办法测试训练效果，可以看到比原先的LeNet效果好了很多
    # 网络很小，大胆加大batch，1epoch = 8.8sec
    batch_size = 1024
    train_iter, test_iter = data_process.load_data_fashion_mnist(batch_size)
    lr, num_epochs = 0.001, 5
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    train.train_ch5(net, train_iter, test_iter,
                    batch_size, optim, device, num_epochs)
    # 输出第一个批量归一化层训练得到的自己的偏移和拉伸参数
    print(net[1].gamma.view((-1,)), net[1].beta.view((-1,)))
    print('————————————————————————————')


    # 然后自然我们也要测试一下批量归一化如何简洁地利用库来实现批量归一化的LeNet
    net = nn.Sequential(
        nn.Conv2d(1, 6, 5),
        # nn提供了二维批量归一化函数BatchNorm2d，参数是输出通道(也相当于输入通道数)
        nn.BatchNorm2d(6),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5),
        nn.BatchNorm2d(16),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2),
        layers.FlattenLayer(),
        nn.Linear(16*4*4, 120),
        # 这里要注意在扁平层后批量归一化变为一通道的，对应BatchNorm1d
        nn.BatchNorm1d(120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.BatchNorm1d(84),
        nn.Sigmoid(),
        nn.Linear(84, 10)
    )

    # 用相同的方法和参数来测试训练效果，速度比自己实现的要快一些，结果是一样的
    # 1epoch = 7.6sec
    batch_size = 1024
    train_iter, test_iter = data_process.load_data_fashion_mnist(batch_size)
    lr, num_epochs = 0.001, 5
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    train.train_ch5(net, train_iter, test_iter,
                    batch_size, optim, device, num_epochs)
    print('————————————————————————————')
