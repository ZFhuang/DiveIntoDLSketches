# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import data_process, plot, train, rnn

import torch
import time
import torchvision
from PIL import Image
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader

"""
这一节介绍了如何对图像数据集进行增广以获得更多的可用数据
"""

# 加上这个限定才能支持多线程读取
if __name__ == "__main__":
    # 设置计算设备，让计算在GPU上进行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 先尝试显示一下猫图
    plot.set_figsize()
    img = Image.open(r"./Datasets"+'/Img/cat1.jpg')
    # plot.plt.imshow(img)
    # # 记得要显示时需要show一下
    # plot.plt.show()
    print('————————————————————————————')

    # 翻转和裁剪是最常见的增广方法
    # 写一个增广图片的函数，其中aug是对应的增广操作函数，让目标图像增广多次然后显示出来
    def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
        # 在循环内对图像执行num_rows * num_cols次操作并组合，然后下面显示出来
        Y = [aug(img) for _ in range(num_rows * num_cols)]
        # 这里保存了一个在一图中显示图片集的函数show_images，需要代入网格尺寸和缩放
        plot.show_images(Y, num_rows, num_cols, scale)

    # # 此处应用了随机水平翻转的库函数，翻转是最方便的增广方法，效果有限
    # apply(img, torchvision.transforms.RandomHorizontalFlip())
    # # 这里是更少用的垂直翻转
    # apply(img, torchvision.transforms.RandomVerticalFlip())
    # # 然后为了降低网络对图像目标所在位置的敏感性，可进行随机裁剪
    # # 这里的参数随机裁剪出面积0.1~1，宽高比0.5~2的区域，然后缩放到200像素来运用
    # apply(img, torchvision.transforms.RandomResizedCrop(
    #     200, scale=(0.1, 1), ratio=(0.5, 2)))
    print('————————————————————————————')

    # # 可从亮度，对比度，饱和度，色调，四个方面调整图像
    # # 可以同时设置
    # apply(img, torchvision.transforms.ColorJitter(
    #     brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    print('————————————————————————————')

    # # 可以将多种图像增广结合起来用，使用Compose函数组合
    # apply(img, torchvision.transforms.Compose([
    #     torchvision.transforms.RandomHorizontalFlip(0.5),
    #     torchvision.transforms.ColorJitter(
    #         brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    #     torchvision.transforms.RandomResizedCrop(
    #         200, scale=(0.1, 1), ratio=(0.5, 2))
    # ]))
    print('————————————————————————————')

    # 这里实际测试一下增广的方法，使用CIFAR10数据集，读取后用循环来显示
    all_imgs = torchvision.datasets.CIFAR10(
        train=True, root=r"./Datasets/CIFAR", download=True)
    # plot.show_images([all_imgs[i][0] for i in range(32)], 4, 8, scale=0.8)
    # 有处理的增广函数，进行了水平翻转
    flip_aug = torchvision.transforms.Compose([
        # 这个组合增广进行了随机水平翻转，然后用ToTensor将图像转为Pytorch可用的格式
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ])
    # 无处理的增广函数
    no_aug = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    num_workers = 4

    # 一个简单的图像读取函数
    def load_cifar10(is_train, augs, batch_size, root=r"./Datasets/CIFAR"):
        # 读取数据集，augs控制想要应用在数据集读取时的增广函数
        dataset = torchvision.datasets.CIFAR10(
            root=root, train=is_train, transform=augs, download=True)
        # 返回对应的读取器
        return DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                          num_workers=num_workers)

    # 然后把保存一个基本的训练函数train
    # 接着再写一个包含了读取，增广，训练全过程的接口函数来调用
    def train_with_data_aug(train_augs, test_augs, lr=0.001):
        batch_size, net = 256, rnn.resnet18(10)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        loss = torch.nn.CrossEntropyLoss()
        train_iter = load_cifar10(True, train_augs, batch_size)
        test_iter = load_cifar10(False, test_augs, batch_size)
        train.train(train_iter, test_iter, net, loss,
                    optimizer, device, num_epochs=10)

    # 调用接口函数进行训练，其中测试集不要进行增广保证结果的稳定
    train_with_data_aug(flip_aug, no_aug)
    print('————————————————————————————')
