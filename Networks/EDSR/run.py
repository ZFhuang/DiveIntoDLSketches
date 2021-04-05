# 计算工作路径
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
# print('Current path: '+curPath)
rootPath = os.path.split(curPath)[0]
rootPath = os.path.split(rootPath)[0]
# print('Root path: '+rootPath)
sys.path.append(rootPath)
dataPath = os.path.split(rootPath)[0]
dataPath = os.path.split(dataPath)[0]
# print('Data path: '+dataPath)

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 7'

from Utils.data_process import *
from Utils.core import *
from Utils.metrics import img_metric
from Utils.loggings import *
from EDSR import EDSR
from torch.utils.data import DataLoader
import getpass
import time
import torch
import logging

# 初始化路径
format_time = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
computer_name = getpass.getuser()
ROOT_FOLDER = dataPath+r'/datasets/T91/'
logging.info('Dataset: '+ROOT_FOLDER)
RAW_FOLDER = ROOT_FOLDER+r'Raw'
TEMP_FOLDER = ROOT_FOLDER+r'Temp'
LR_FOLDER = ROOT_FOLDER+r'LR'
HR_FOLDER = ROOT_FOLDER+r'HR'
OUTPUTS_FOLDER = curPath+r'/Outputs'
LOGGING_FOLDER=curPath+r'/logs'
INPUTS_FOLDER_TRAIN = ROOT_FOLDER+r'Inputs_train'
LABELS_FOLDER_TRAIN = ROOT_FOLDER+r'Labels_train'
INPUTS_FOLDER_TEST = ROOT_FOLDER+r'Inputs_test'
LABELS_FOLDER_TEST = ROOT_FOLDER+r'Labels_test'
SAVE_PATH = rootPath+r'/Models/EDSR_'+computer_name+r'_'+format_time+r'.pth'
LOAD_PATH = rootPath+r'/Models/EDSR_ZFH_2021-03-31_19-19-14.pth'

# 日志设置
check_folder(LOGGING_FOLDER)
set_logger(LOGGING_FOLDER, format_time, computer_name, logging.INFO)
logging.info('Start: '+curPath)

# 数据参数
ratio = 2
LR_size = 48
logging.info('Ratio: '+str(ratio))
logging.info('LR_size: '+str(LR_size))

# 初始化数据集文件夹
check_folder(OUTPUTS_FOLDER)
init_folder(TEMP_FOLDER)
init_folder(LR_FOLDER)
init_folder(HR_FOLDER)
init_folder(INPUTS_FOLDER_TRAIN)
init_folder(LABELS_FOLDER_TRAIN)
init_folder(INPUTS_FOLDER_TEST)
init_folder(LABELS_FOLDER_TEST)
logging.info('Folders inited!')

# 扩张数据集
expand_dataset(RAW_FOLDER, TEMP_FOLDER, scale=1, rotate=0, flip=0)
expand_dataset(RAW_FOLDER, TEMP_FOLDER, scale=1, rotate=1, flip=0)
expand_dataset(RAW_FOLDER, TEMP_FOLDER, scale=1, rotate=2, flip=0)
expand_dataset(RAW_FOLDER, TEMP_FOLDER, scale=1, rotate=3, flip=0)
expand_dataset(RAW_FOLDER, TEMP_FOLDER, scale=1, rotate=0, flip=1)
expand_dataset(RAW_FOLDER, TEMP_FOLDER, scale=1, rotate=1, flip=1)
expand_dataset(RAW_FOLDER, TEMP_FOLDER, scale=1, rotate=2, flip=1)
expand_dataset(RAW_FOLDER, TEMP_FOLDER, scale=1, rotate=3, flip=1)
logging.info('Dataset expanded!')

# 采样并复制图像
sample_images(TEMP_FOLDER, LR_FOLDER, 1/ratio)
sample_images(TEMP_FOLDER, HR_FOLDER, 1)
# 移动文件
align_images(LR_FOLDER, LR_FOLDER, INPUTS_FOLDER_TRAIN)
align_images(HR_FOLDER, HR_FOLDER, LABELS_FOLDER_TRAIN)
# 将图像分割
cut_images(INPUTS_FOLDER_TRAIN,LABELS_FOLDER_TRAIN, LR_size, ratio)
logging.info('Training data generated!')

# 随机分配文件到测试集中
random_move(INPUTS_FOLDER_TRAIN,LABELS_FOLDER_TRAIN,INPUTS_FOLDER_TEST,LABELS_FOLDER_TEST,0.1)
logging.info('Testing data generated!')

# 初始化网络
net = EDSR(scale=ratio, in_channel=1, num_filter=256, num_resiblk=32, resi_scale=0.1)
net.apply(weights_init_kaiming)
logging.debug('\n'+str(net))
logging.info('Network inited!')

# 训练参数
lr, num_epochs = 0.0003, 1000
batch_size = 32
my_optim = torch.optim.Adam(net.parameters(), lr=lr)
# my_optim = torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    my_optim, mode='min', factor=0.1, patience=20, verbose=True)
loss = nn.L1Loss()
log_training_settings(lr,num_epochs,batch_size,my_optim,scheduler,loss)

# 读取数据集
train_dataset = ImagePairDataset_y(INPUTS_FOLDER_TRAIN, LABELS_FOLDER_TRAIN)
test_dataset = ImagePairDataset_y(INPUTS_FOLDER_TEST, LABELS_FOLDER_TEST)
train_iter = DataLoader(train_dataset, batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, 1, shuffle=True)
logging.info('Datasets loaded!')

# 训练
train(train_iter, test_iter, net, loss, my_optim, num_epochs, scheduler)

# 测试
logging.info('Final test loss %e' % eval(test_iter, net, loss, 0))

# 保存网络
torch.save(net.state_dict(), SAVE_PATH)
logging.info('Network saved: '+SAVE_PATH)

# # 读取网络
# net.load_state_dict(torch.load(LOAD_PATH))
# logging.info('Network loaded: '+LOAD_PATH)

# 应用完整图片并写入
IMG_NAME = r'img_0'
IMG_OUT = OUTPUTS_FOLDER+r'/'+IMG_NAME+r'_' + \
    computer_name+r'_'+format_time+r'.png'
output = apply_net(LR_FOLDER+r'/'+IMG_NAME+r'.png', IMG_OUT, net, ratio)

# 对测试图片进行评估
img_metric(IMG_OUT, HR_FOLDER+r'/'+IMG_NAME+r'.png')
