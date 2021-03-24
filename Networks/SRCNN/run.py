# 计算工作路径
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
print(curPath)
rootPath = os.path.split(curPath)[0]
rootPath = os.path.split(rootPath)[0]
print(rootPath)
sys.path.append(rootPath)
dataPath = os.path.split(rootPath)[0]
dataPath = os.path.split(dataPath)[0]
print(dataPath)

import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from SRCNN import SRCNN
from Utils.core import train,eval,apply_net
from Utils.data_process import ImagePairDataset_y, sample_images,cut_images,random_move,init_folder,align_images
from Utils.metrics import img_metric
import cv2


# 初始化路径
root_folder=dataPath+r'/datasets/BSDS200/'
Raw_folder=root_folder+r'Raw'
LR_folder=root_folder+r'LR'
HR_folder=root_folder+r'HR'
Outputs_folder=curPath+r'/Outputs'
Inputs_folder_train=root_folder+r'Inputs_train'
Labels_folder_train=root_folder+r'Labels_train'
Inputs_folder_test=root_folder+r'Inputs_test'
Labels_folder_test=root_folder+r'Labels_test'
MODEL_PATH = rootPath+r'/Models/SRCNN.pth'

# # 初始化数据集文件夹
# init_folder(LR_folder)
# init_folder(HR_folder)
# init_folder(Outputs_folder)
# init_folder(Inputs_folder_train)
# init_folder(Labels_folder_train)
# init_folder(Inputs_folder_test)
# init_folder(Labels_folder_test)

# # 采样并复制图像
# sample_images(Raw_folder, LR_folder,0.5)
# sample_images(Raw_folder, HR_folder,1)
# align_images(LR_folder, HR_folder, LR_folder)
# sample_images(LR_folder, Inputs_folder_train,1)
# sample_images(HR_folder, Labels_folder_train,1)

# # 然后将图像分割
# size=64
# cut_images(Inputs_folder_train, size, size//1)
# cut_images(Labels_folder_train, size, size//1)

# # 随机分配文件到测试集中
# random_move(Inputs_folder_train,Labels_folder_train,Inputs_folder_test,Labels_folder_test,0.1)

# 设置训练参数
net=SRCNN()
lr, num_epochs = 0.03, 400
batch_size = 32
my_optim=torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=0.0001)
# 自适应学习率
scheduler = torch.optim.lr_scheduler.StepLR(my_optim,step_size=40,gamma = 0.1)
loss = torch.nn.MSELoss()

# 读取数据集
train_dataset=ImagePairDataset_y(Inputs_folder_train,Labels_folder_train)
test_dataset=ImagePairDataset_y(Inputs_folder_test,Labels_folder_test)
train_iter = DataLoader(train_dataset, batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, 1, shuffle=True)
print('Datasets loaded!')

# 训练
train(train_iter, test_iter, net, loss, my_optim, num_epochs,scheduler)

# 测试
print('Full test loss %.4f'%eval(test_iter,net,loss,0))

# 保存网络
torch.save(net.state_dict(), MODEL_PATH)

# 读取网络
net.load_state_dict(torch.load(MODEL_PATH))

# 应用完整图片并写入
output=apply_net(LR_folder+r'/img_1.png', Outputs_folder+r'/img_1_out.png',net)

# 对测试图片进行评估
img_metric(Outputs_folder+r'/img_1_out.png', HR_folder+r'/img_1.png')