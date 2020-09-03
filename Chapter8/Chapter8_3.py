# coding=utf-8

import torch
import time
from d2lzh_pytorch import train

# 声明电脑不止一个可CUDA的GPU
assert torch.cuda.device_count() >= 2

"""
这一节介绍了pytorch中的自动并行计算，至少需要两个GPU，由于电脑只有一个独显所以无法测试
"""

# 首先为了下面好判断并行的效果，保存一个计时类Benchamrk，用with来使用这个计时类


# 然后定义一个用于模拟运算的20000次矩阵乘法函数run
def run(x):
    for _ in range(20000):
        y = torch.mm(x, x)


# 输出gpu信息
x_gpu1 = torch.rand(size=(100, 100), device='cuda:0')
x_gpu2 = torch.rand(size=(100, 100), device='cuda:1')

# 单独让两个GPU分别运行，测试时间
with train.Benchmark('Run on GPU1.'):
    run(x_gpu1)
    torch.cuda.synchronize()
with train.Benchmark('Run on GPU2.'):
    run(x_gpu2)
    torch.cuda.synchronize()

# 再测试并行计算的效果如何
with train.Benchmark('Run on both GPU1 and GPU2 in parallel.'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()