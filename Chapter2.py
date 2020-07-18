# 用这个来导入包
import torch

# 创建空tensor,注意其值并非严格的0
x = torch.empty(4, 6)

# 创建随机tensor
x = torch.rand(5, 7)


# 输出
print(x)
