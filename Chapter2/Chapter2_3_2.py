# coding=utf-8

import torch

"""
这一节介绍了自动求梯度autograd的相关知识
"""

# 为了模拟实际的梯度传播过程，这里假设了一个典型的线性计算图：
# pytorch的特点就是计算的时候是基于一个动态的计算图进行的，可以方便实现纸面设计
# 直接创建的tensor，是计算图的叶子结点，也就是求导中最底层的变量
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

# 梯度计算的重点是backward函数，会从你想要开始的值作为loss自动计算出反向传播的梯度
# 调用后我们可以对某个变量用grad，这代表梯度传播后对目标变量求导
# 但是Pytorch不允许对张量求导，只能让标量对张量求导，结果是和自变量同型的张量
# 原因是防止出现矩阵对矩阵求偏导这样复杂奇怪的操作
# 因此此处我们只能调用x.grad，这代表着求d(out)/dx
# 此处我们可以通过给backward带上参数retain_graph来让计算后的梯度图保留
# 这样我们后面可以多次调用同一个反向传播而得到梯度的累加，默认情况下不能累加
out.backward(retain_graph=True)
print(out)
# 只能对标量使用
print(x.grad)
print('————————————')

# 由于前面的设置这里的梯度是会积累的，也就是你每次调用backward数值都会累加
out.backward(retain_graph=True)
print(x.grad)
print('————————————')

# backward还带有另一个很神奇的参数grad_variables，这个参数有几篇文章详细解释：
# https://www.cnblogs.com/king-lps/p/8336494.html
# https://zhuanlan.zhihu.com/p/29923090
# 这个参数的基础效果是一个梯度的权重张量，我们通过设置这个权重能改变得到的梯度
# 会让d(out)/dx变成backward*d(out)/dx，第一个效果就是实现"有权重loss"的需求
# 例如这里给out.backward带上参数2.0改变out的值，最终得到的梯度会出现相应的倍率
# 先把梯度置零,这里由于是单链的线性系统所以只有x有梯度项
x.grad.data.zero_()
out.backward(torch.tensor(2.0), retain_graph=True)
print(x.grad)
print('————————————')

# 另一种重要效果就是函数求导的链式法则，我们知道链式法则中只要知道了中间的某一个导数结果
# 就可以从这个结点处继续往内链式求导，这个参数就可以充当中间导数的作用
# 如果我们给这个参数传入需要backward的结点处的导数，就可以从那个点开始求梯度
# 这样就可以理解之所以out不需要参数，是因为最终的结点的导数可以理解为1
# 那么如下可以尝试从z开始求导,这里经过人工推导得知z处的导数为[[.25, .25], [.25, .25]]
# 因此从z处进行反向传播可以得到正确的x的梯度
x.grad.data.zero_()
z.backward(torch.tensor([[.25, .25], [.25, .25]]), retain_graph=True)
print(x.grad)
print('————————————')

# 有些文章可能会提到Variable类型，这个类型在2018.4之后已经和tensor合并
