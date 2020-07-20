# coding=utf-8

# 用这个来导入PyTorch包,
import torch

"""
这一节介绍了tensor的基础的创建操作
"""

# tensor(张量)是一个可用来表示在一些矢量、标量和其他张量之间的线性关系的多线性函数
# 通常来说0阶叫纯量，1阶叫向量，2阶叫矩阵，三阶以上统称张量
# 创建空tensor,注意并不是0tensor，而是带有值的
x = torch.empty(4, 6)

# 创建随机tensor
# rand函数创建[0,1)范围内的随机数
x = torch.rand(5, 7)

# 创建全0tensor
x = torch.zeros(2, 9)
# 可以指定dtype参数来指定需要由什么类型的数据来组成
# 还可以用device参数指定数据要用cpu还是gpu处理,分为字符串'cpu'和'cuda'
# 例如这里torch.int表现为torch.int32
x = torch.zeros(2, 9, dtype=torch.int, device='cuda')

# 直接根据数据创建tensor，类似C++的列表初始化
# 对于如下的多维tensor需要用子括号分隔
x = torch.tensor([[1, 2, 3],
                  [4, 6, 0]], dtype=torch.float)

# 使用现有的tensor来创建tensor，这类方法会重用现有的tensor的一些设置
# 例如这里上面指定了元素是float类型，下面重用后的元素类型也还是float类型
# new_ones指创建一个新的全1的tensor，大小可以另外指定，例如这里创建的大小就和之前不同
x = x.new_ones(2, 4)

# 创建一个值随机但形式相同的tensor
# randn是标准正态分布（均值0，方差1）
x = torch.randn_like(x)

# 输出tensor本身
print(x)
# 输出tensor形状, 返回torch.Size为一个tuple
print(x.shape)
# 输出tensor形状的另一种写法
print(x.size())
