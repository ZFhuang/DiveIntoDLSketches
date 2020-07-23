# coding=utf-8

"""
这一节从零开始实现一个线性回归训练
"""

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的路径添加到系统路径中
# 注意导入包内的具体文件必须用from，然后使用时还需要带上文件名检索
import sys
sys.path.append(r".")
from d2lzh_pytorch import plot
from d2lzh_pytorch import data_process
from d2lzh_pytorch import linear_reg

# 导入许多计算和绘制结果的包
import numpy as np
from matplotlib import pyplot as plt
import random
from IPython import display
import torch

# 首先尝试生成样例样本，1000个样本，每个样本2个特征和1个标签
# 由于目标是生成测试用的带有线性关系的结果，所以首先生成符合标准正态分布的特征张量
num_features = 2
num_examples = 1000
# 利用numpy生成高斯分布后再转到tensor，参数是均值，方差，维度
features = torch.from_numpy(np.random.normal(
    0, 1, (num_examples, num_features)))
# 然后利用特征生成线性标签集,依靠权重real_w和偏置real_b
real_w = [2, -3.4]
real_b = 4.2
labels = real_w[0]*features[:, 0]+real_w[1]*features[:, 1]+real_b
# 再给标签附加高斯噪声模拟真实数据
e = np.random.normal(0, 0.01, size=labels.size())
labels += e
# 然后用pyplot将tensor转换为矢量图,参数是轴1，轴2，点的大小
# https://blog.csdn.net/u013634684/article/details/49646311
plot.set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# 一定要调用show才会显示
# plt.show()
print('————————————')

# 训练读取数据的时候，通常我们是按照小批量来读取数据的
# 因此首先定义一个函数data_iter每次从数据集中抽取batch_size大小的数据
# 然后在循环中读取一个个的batch，这里由于break只取了一个
b_size = 10
for x, y in data_process.data_iter(b_size, features, labels):
    print(x)
    print(y)
    break

# 然后为了创建自己的线性回归模型，首先准备待学习的权重和偏差参数,并启动其梯度追踪
# 这里的模型将会是y=wx+b，因此特征个w，一个b，都设置为float提高精度
# 权重初始化为0,0.01的正态随机数
w = torch.tensor(np.random.normal(0, 0.01, (num_features, 1)), dtype=float)
w.requires_grad_(True)
# 偏差初始化为0，一个元素
b = torch.zeros(1, dtype=torch.float32)
b.requires_grad_(True)

# 然后定义模型的线性模型linreg,损失函数squared_loss和梯度下降函数sgd
# 然后开始正式的训练
# 学习率一般都是3的倍数,batch_size大则快而不稳，反之。学习周期是学习完全部样本一次
learning_rate = 0.03
epochs = 5
b_size = 10
# 用函数指针先简化下后面的调用
net = linear_reg.linreg
loss = linear_reg.squared_loss
# 以周期为单位进行循环
for ep in range(epochs):
    # 子循环以batch为单位，取出feature和label
    for X, y in data_process.data_iter(b_size, features, labels):
        # 先尝试预测一下并算出loss，此时的loss是一个tensor，要sum一下方便求梯度
        # loss需要除batchsize的部分已经在函数里处理过了，所以sum就是求平均
        l = loss(net(X, w, b), y).sum()
        # 对loss进行反向传播计算w和b的梯度
        l.backward()
        # 在sgd中用梯度修改w和b的值
        linear_reg.sgd([w, b], learning_rate, b_size)

        # 不要忘了需要进行梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    # 完成一周期的训练后对这个模型用所有训练样本进行一次loss计算，也就是训练损失记录
    train_l = loss(net(features, w, b), labels)
    # 输出训练误差结果
    print('epoch %d, loss %f' % (ep+1, train_l.mean().item()))

#输出最终学习到的参数
print(real_w,'\n', w)
print(real_b,'\n',b)