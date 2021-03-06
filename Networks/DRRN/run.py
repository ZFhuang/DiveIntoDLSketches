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

# 设置Cuda
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 7'

from Utils.data_process import ImagePairDataset_y, sample_images, cut_images, random_move, init_folder, crop_images, expand_dataset, align_images
from Utils.core import train, eval, weights_init_kaiming, apply_net_preupsample
import getpass
from Utils.metrics import img_metric
from DRRN import DRRN
from torch.utils.data import DataLoader
import time
import torch

# 初始化路径
ROOT_FOLDER = dataPath+r'/datasets/T91/'
RAW_FOLDER = ROOT_FOLDER+r'Raw'
TEMP_FOLDER = ROOT_FOLDER+r'Temp'
LR_FOLDER = ROOT_FOLDER+r'LR'
HR_FOLDER = ROOT_FOLDER+r'HR'
OUTPUTS_FOLDER = curPath+r'/Outputs'
INPUTS_FOLDER_TRAIN = ROOT_FOLDER+r'Inputs_train'
LABELS_FOLDER_TRAIN = ROOT_FOLDER+r'Labels_train'
INPUTS_FOLDER_TEST = ROOT_FOLDER+r'Inputs_test'
LABELS_FOLDER_TEST = ROOT_FOLDER+r'Labels_test'
format_time = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
print('Start in: '+format_time)
computer_name = getpass.getuser()
SAVE_PATH = rootPath+r'/Models/DRRN_'+computer_name+r'_'+format_time+r'.pth'
LOAD_PATH = rootPath+r'/Models/DRRN_ZFH_2021-03-27_16-35-58.pth '

# 数据参数
ratio = 2
LR_size = 31
# init_folder(OUTPUTS_FOLDER)

# # 初始化数据集文件夹
# init_folder(TEMP_FOLDER)
# init_folder(LR_FOLDER)
# init_folder(HR_FOLDER)
# init_folder(INPUTS_FOLDER_TRAIN)
# init_folder(LABELS_FOLDER_TRAIN)
# init_folder(INPUTS_FOLDER_TEST)
# init_folder(LABELS_FOLDER_TEST)

# # 扩张数据集
# expand_dataset(RAW_FOLDER, TEMP_FOLDER, scale=1, rotate=0, flip=0)
# expand_dataset(RAW_FOLDER, TEMP_FOLDER, scale=1, rotate=1, flip=0)
# expand_dataset(RAW_FOLDER, TEMP_FOLDER, scale=1, rotate=2, flip=0)
# expand_dataset(RAW_FOLDER, TEMP_FOLDER, scale=1, rotate=3, flip=0)
# expand_dataset(RAW_FOLDER, TEMP_FOLDER, scale=1, rotate=0, flip=1)
# expand_dataset(RAW_FOLDER, TEMP_FOLDER, scale=1, rotate=1, flip=1)
# expand_dataset(RAW_FOLDER, TEMP_FOLDER, scale=1, rotate=2, flip=1)
# expand_dataset(RAW_FOLDER, TEMP_FOLDER, scale=1, rotate=3, flip=1)

# # 采样并复制图像
# sample_images(TEMP_FOLDER, LR_FOLDER, 1/ratio)
# sample_images(TEMP_FOLDER, HR_FOLDER, 1)

# # 移动文件
# align_images(LR_FOLDER, HR_FOLDER, INPUTS_FOLDER_TRAIN)
# align_images(HR_FOLDER, HR_FOLDER, LABELS_FOLDER_TRAIN)

# # 将图像分割
# cut_images(INPUTS_FOLDER_TRAIN, LABELS_FOLDER_TRAIN, LR_size, 1)

# # 随机分配文件到测试集中
# random_move(INPUTS_FOLDER_TRAIN, LABELS_FOLDER_TRAIN,
#             INPUTS_FOLDER_TEST, LABELS_FOLDER_TEST, 0.1)

# 训练参数
net = DRRN(num_recur_blocks=1, num_resid_units=7, num_filter=32, filter_size=3)
net.apply(weights_init_kaiming)
print(net)
lr, num_epochs = 0.1, 1000
batch_size = 128
my_optim = torch.optim.SGD(net.parameters(), lr=lr,
                           momentum=0.9, weight_decay=0.0001)
# 自适应学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    my_optim, mode='min', factor=0.1, patience=5, verbose=True)
loss = torch.nn.MSELoss()

# 读取数据集
train_dataset = ImagePairDataset_y(INPUTS_FOLDER_TRAIN, LABELS_FOLDER_TRAIN)
test_dataset = ImagePairDataset_y(INPUTS_FOLDER_TEST, LABELS_FOLDER_TEST)
train_iter = DataLoader(train_dataset, batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, 1, shuffle=True)
print('Datasets loaded!')

# 训练
train(train_iter, test_iter, net, loss, my_optim, num_epochs, scheduler,need_gclip=True)

# 测试
print('Full test loss %.4f' % eval(test_iter, net, loss, 0))

# 保存网络
torch.save(net.state_dict(), SAVE_PATH)
print('Network saved: '+SAVE_PATH)

# 读取网络
# net.load_state_dict(torch.load(LOAD_PATH))
print('Network loaded: '+LOAD_PATH)

# 应用完整图片并写入
IMG_NAME = r'img_0'
IMG_OUT = OUTPUTS_FOLDER+r'/'+IMG_NAME+r'_' + \
    computer_name+r'_'+format_time+r'.png'
output = apply_net_preupsample(LR_FOLDER+r'/'+IMG_NAME+r'.png', IMG_OUT, net, ratio)

# 对测试图片进行评估
img_metric(IMG_OUT, HR_FOLDER+r'/'+IMG_NAME+r'.png')
