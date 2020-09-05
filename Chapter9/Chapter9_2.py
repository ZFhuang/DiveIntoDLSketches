# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import data_process, plot, train, rnn

import torch
import time
import os
import torchvision
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models

"""
这一节介绍了如何利用简单的迁移学习来从一个泛化模型得到知识并应用到另外的问题上
"""


# 加上这个限定才能支持多线程读取
if __name__ == "__main__":
    # 设置计算设备，让计算在GPU上进行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 首先读取网上下载好的热狗数据集,这是第一次读取没有打包的数据集
    data_dir=r'./Datasets'
    # 文件夹中是trian和test两个子文件夹，子文件夹中是hotdogs和not-hotdogs两种分类
    # print(os.listdir(os.path.join(data_dir, "Hotdog")))
    
    # 这里用两个ImageFolder来读取两个文件夹中的文件
    train_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/train'))
    test_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/test'))

    # 这里为了测试画出头8张图片
    # hotdogs = [train_imgs[i][0] for i in range(8)]
    # not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
    # plot.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
    print('————————————————————————————')

    # 然后开始对图像进行预处理(增广)，要注意如果是迁移学习的话，一定要保证源模型和目标
    # 模型的输入有一样的预处理，最显然的就是缩放为相同的大小
    # 这里的归一化是和torchvision的模型一样的归一化，参数可以在网上找到
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])

    # 然后从网上下载在ImageNet上训练好的ResNet18,也可以用这里的模型自己训练
    pretrained_net = models.resnet18(pretrained=True)
    # 打印最后的全连接层来测试模型是否正确下载，而且可以看到最后入512出1000类标签
    print(pretrained_net.fc)
    # 将最后的全连接层修改为符合热狗数据集的形式，512转2个输出,修改后会自动初始化参数
    pretrained_net.fc = nn.Linear(512, 2)

    # 然后由于迁移学习的特性，重点是训练最后的全连接输出层，因此这里要把最后全连接层的
    # 学习率调大，然后其他层的学习率调低
    # 提取出全连接层的参数，命名为其自己的id
    output_params = list(map(id, pretrained_net.fc.parameters()))
    # 从参数中提取出不是全连接层的参数项出来
    feature_params = filter(lambda parameter_list: id(
        parameter_list) not in output_params, pretrained_net.parameters())
    
    lr = 0.01
    # 选用sgd优化器，特点是普通参数用0.01的学习率，全连接层用10倍的学习率以关注于它
    # weight_decay是防止过拟合的那个衰减项
    optimizer = optim.SGD([
        {'params': feature_params},
        {'params': pretrained_net.fc.parameters(), 'lr': lr*10}
    ], lr=lr, weight_decay=0.001)

    # 这里写一个训练函数接口函数
    def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
        train_iter = DataLoader(ImageFolder(os.path.join(
            data_dir, 'hotdog/train'), transform=train_augs), batch_size, shuffle=True)
        test_iter = DataLoader(ImageFolder(os.path.join(
            data_dir, 'hotdog/test'), transform=test_augs), batch_size)
        loss = torch.nn.CrossEntropyLoss()
        # 调用之前写的通用训练函数来测试
        train.train(train_iter, test_iter, net, loss,
                    optimizer, device, num_epochs)

    # 对迁移模型进行训练并查看效果
    train_fine_tuning(pretrained_net, optimizer)
    # 对比没有迁移的模型训练，可以看到迁移模型的效果要比没有迁移直接训练的好很多
    scratch_net = models.resnet18(pretrained=False, num_classes=2)
    lr = 0.1
    optimizer = optim.SGD(scratch_net.parameters(), lr=lr, weight_decay=0.001)
    train_fine_tuning(scratch_net, optimizer)
    print('————————————————————————————')
