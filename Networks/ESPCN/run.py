# 计算工作路径
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
print('Current path: '+curPath)
rootPath = os.path.split(curPath)[0]
rootPath = os.path.split(rootPath)[0]
print('Root path: '+rootPath)
sys.path.append(rootPath)
dataPath = os.path.split(rootPath)[0]
dataPath = os.path.split(dataPath)[0]
print('Data path: '+dataPath)

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from ESPCN import ESPCN
from Utils.core import train,eval,apply_net
from Utils.data_process import ImagePairDataset_y, sample_images,cut_images,random_move,init_folder,crop_images,expand_dataset
from Utils.metrics import img_metric
import getpass

# 初始化路径
ROOT_FOLDER=dataPath+r'/datasets/BSDS200/'
RAW_FOLDER=ROOT_FOLDER+r'Raw'
TEMP_FOLDER=ROOT_FOLDER+r'Temp'
LR_FOLDER=ROOT_FOLDER+r'LR'
HR_FOLDER=ROOT_FOLDER+r'HR'
OUTPUTS_FOLDER=curPath+r'/Outputs'
INPUTS_FOLDER_TRAIN=ROOT_FOLDER+r'Inputs_train'
LABELS_FOLDER_TRAIN=ROOT_FOLDER+r'Labels_train'
INPUTS_FOLDER_TEST=ROOT_FOLDER+r'Inputs_test'
LABELS_FOLDER_TEST=ROOT_FOLDER+r'Labels_test'
format_time=str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
computer_name=getpass.getuser()
SAVE_PATH = rootPath+r'/Models/ESPCN_'+computer_name+r'_'+format_time+r'.pth'
LOAD_PATH = rootPath+r'/Models/'

# 数据参数
ratio=2
LR_size=17

# 初始化数据集文件夹
init_folder(TEMP_FOLDER)
init_folder(LR_FOLDER)
init_folder(HR_FOLDER)
# init_folder(OUTPUTS_FOLDER)
init_folder(INPUTS_FOLDER_TRAIN)
init_folder(LABELS_FOLDER_TRAIN)
init_folder(INPUTS_FOLDER_TEST)
init_folder(LABELS_FOLDER_TEST)

# 扩张数据集
expand_dataset(RAW_FOLDER,TEMP_FOLDER,1,0)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,1,1)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,1,2)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,1,3)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,0.9,0)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,0.9,1)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,0.9,2)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,0.9,3)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,0.8,0)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,0.8,1)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,0.8,2)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,0.8,3)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,0.7,0)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,0.7,1)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,0.7,2)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,0.7,3)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,0.6,0)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,0.6,1)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,0.6,2)
# expand_dataset(RAW_FOLDER,TEMP_FOLDER,0.6,3)

# 采样并复制图像
sample_images(TEMP_FOLDER, LR_FOLDER, 1/ratio)
sample_images(TEMP_FOLDER, HR_FOLDER,1)

# 移动文件
sample_images(LR_FOLDER, INPUTS_FOLDER_TRAIN,1)
sample_images(HR_FOLDER, LABELS_FOLDER_TRAIN,1)

# 将图像分割
cut_images(INPUTS_FOLDER_TRAIN,LABELS_FOLDER_TRAIN, LR_size, ratio)
# crop_images(LABELS_FOLDER_TRAIN, LABELS_FOLDER_TRAIN,(ratio-1))

# 随机分配文件到测试集中
random_move(INPUTS_FOLDER_TRAIN,LABELS_FOLDER_TRAIN,INPUTS_FOLDER_TEST,LABELS_FOLDER_TEST,0.1)

# 训练参数
net=ESPCN(ratio)
print(net)
lr, num_epochs = 0.1, 100
batch_size = 128
my_optim=torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=0.0001)
# 自适应学习率
scheduler = torch.optim.lr_scheduler.StepLR(my_optim,step_size=40,gamma = 0.3)
loss = torch.nn.MSELoss()

# 读取数据集
train_dataset=ImagePairDataset_y(INPUTS_FOLDER_TRAIN,LABELS_FOLDER_TRAIN)
test_dataset=ImagePairDataset_y(INPUTS_FOLDER_TEST,LABELS_FOLDER_TEST)
train_iter = DataLoader(train_dataset, batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, 1, shuffle=True)
print('Datasets loaded!')

# 训练
train(train_iter, test_iter, net, loss, my_optim, num_epochs,scheduler)

# 测试
print('Full test loss %.4f'%eval(test_iter,net,loss,0))

# 保存网络
torch.save(net.state_dict(), SAVE_PATH)
print('Network saved: '+SAVE_PATH)

# 读取网络
# net.load_state_dict(torch.load(LOAD_PATH))
print('Network loaded: '+LOAD_PATH)

# 应用完整图片并写入
IMG_NAME=r'img_0'
IMG_OUT=OUTPUTS_FOLDER+r'/'+IMG_NAME+r'_'+computer_name+r'_'+format_time+r'.png'
output=apply_net(LR_FOLDER+r'/'+IMG_NAME+r'.png', IMG_OUT,net,ratio)

# 对测试图片进行评估
img_metric(IMG_OUT, HR_FOLDER+r'/'+IMG_NAME+r'.png')