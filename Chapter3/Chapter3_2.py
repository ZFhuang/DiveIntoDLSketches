# coding=utf-8

"""
这一节从零开始实现一个线性回归训练
"""

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的路径添加到系统路径中
# 注意导入包内的具体文件必须用from，然后使用时还需要带上文件名检索
import sys
sys.path.append(r".")
from d2lzh_pytorch import plot

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
# 然后利用特征生成线性标签集,依靠权重w和偏置b
w = [2, -3.4]
b = 4.2
labels = w[0]*features[:, 0]+w[1]*features[:, 1]+b
# 再给标签附加高斯噪声模拟真实数据
e = np.random.normal(0, 0.01, size=labels.size())
labels += e
# 然后用pyplot将tensor转换为矢量图,参数是轴1，轴2，点的大小
# https://blog.csdn.net/u013634684/article/details/49646311
plot.set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# 一定要调用show才会显示
plt.show()
print('————————————')

# 训练读取数据的时候，通常我们是按照小批量来读取数据的
# 因此首先定义一个函数每次从数据集中抽取batch_size大小的数据
