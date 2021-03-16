# coding=utf-8

# 用这个来导入PyTorch包
import torch

"""
这一节介绍了tensor基础的操作
"""

# 默认数据格式是float32
x = torch.ones(2, 4)
y = torch.zeros(2, 4)

# tensor加法1
z = x+y
# tensor加法2
z = x.add(y)
# tensor加法3,注意pytorch中函数后缀是下划线的均代表此操作是原地操作
z = x.add_(y)
print(z)
print('————————————')

# 可以获得tensor的索引，冒号和matlab一样用于得到一个范围值
y = x[0, :]
# 对索引的操作会影响到原tensor的值，也就是此时y+1的操作会影响x的值
y += 1
print(y)
print('————————————')

# 利用名为mask的二进制tensor选取input中所需的元素组成tensor
# 要注意得到的tensor是一维的
mask = torch.tensor([[False, False, True, True], [False, False, True, False]])
q = torch.masked_select(x, mask)
print(q)
print('————————————')

# view可以改变tensor的形状，也就是重新排布tensor
# 这个操作返回的tensor和原tensor共享内存，因为view实际上只是改变了观察的角度
# 要注意tensor必须能够正好符合重排
q = q.view(3, 1)
# 如果想要返回副本，最好先clone得到新副本在view，reshape也能得到类似效果但不保证是副本
q = q.clone()
print(q)
print('————————————')

# item函数可以将一个标量tensor转为一个普通数字
x = torch.rand(1)
print(x)
print(x.item())
print('————————————')

# tensor还支持很多线代操作
# trace求对角线之和(迹)，diag求对角线元素，triu/tril求上下三角
# t转置，dot内积，cross外积，inverse求逆，svd奇异值分解
