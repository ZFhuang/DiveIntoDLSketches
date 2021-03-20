import sys
sys.path.append(r'C:\Work\DiveIntoDLSketches')

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from Utils.data_process import ImagePairDataset, bicubic_images,cut_images,random_move,reinit_folder,apply_net
from Utils.core import train,eval_with_img,eval
from SRCNN import SRCNN

# 初始化路径
root_folder=r'./Datasets/Set14/'
LR_folder=root_folder+r'LR'
HR_folder=root_folder+r'HR'
Outputs_folder=root_folder+r'Outputs'
Inputs_folder_train=root_folder+r'Inputs_train'
Labels_folder_train=root_folder+r'Labels_train'
Inputs_folder_test=root_folder+r'Inputs_test'
Labels_folder_test=root_folder+r'Labels_test'
MODEL_PATH = r'Models/SRCNN.pth'

# 初始化数据集文件夹
reinit_folder(Inputs_folder_train)
reinit_folder(Labels_folder_train)
reinit_folder(Inputs_folder_test)
reinit_folder(Labels_folder_test)

# 采样并复制图像
bicubic_images(LR_folder, Inputs_folder_train,1)
bicubic_images(HR_folder, Labels_folder_train,1)

# 然后将图像分割
size=128
cut_images(Inputs_folder_train, size,size//1)
cut_images(Labels_folder_train, size,size//1)

# 随机分配文件到测试集中
random_move(Inputs_folder_train,Labels_folder_train,Inputs_folder_test,Labels_folder_test,0.05)

# 设置训练参数
net=SRCNN()
lr, num_epochs = 0.0003, 500
batch_size = 8
optim=torch.optim.Adam(net.parameters(),lr=lr)
loss = torch.nn.MSELoss()

# 读取数据集
train_dataset=ImagePairDataset(Inputs_folder_train,Labels_folder_train)
test_dataset=ImagePairDataset(Inputs_folder_test,Labels_folder_test)
train_iter = DataLoader(train_dataset, batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, 1, shuffle=True)

# 训练
train(train_iter, test_iter, net, loss, optim, num_epochs)

# # 测试
# print('full test loss %.4f'%eval(test_iter,net,loss,0))

# # 保存网络
# torch.save(net.state_dict(), MODEL_PATH)

# 读取网络
net.load_state_dict(torch.load(MODEL_PATH))

# 应用完整图片并写入
output=apply_net(LR_folder+r'/2.png', Outputs_folder+r'/1.png',net)