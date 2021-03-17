import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import cv2
from torch.utils.data import DataLoader
from PIL import Image
from data_process import ImagePairDataset, bicubic_images,cut_images,train,random_move
from SRCNN import SRCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 生成可用数据集
LR_folder=r'./Datasets/Set5/LR'
HR_folder=r'./Datasets/Set5/HR'

Inputs_folder_train=r'./Datasets/Set5/Inputs_train'
Labels_folder_train=r'./Datasets/Set5/Labels_train'

Inputs_folder_test=r'./Datasets/Set5/Inputs_test'
Labels_folder_test=r'./Datasets/Set5/Labels_test'

Outputs_folder=r'./Datasets/Set5/Outputs'

# # 采样并复制图像
# bicubic_images(LR_folder, Inputs_folder,2)
# bicubic_images(HR_folder, Labels_folder,1)

# # 然后将图像分割
# size=28
# stride=size//2
# cut_images(Inputs_folder, size, stride)
# cut_images(Labels_folder, size, stride)

# 随机移动文件到测试集中
# random_move(Inputs_folder_train,Labels_folder_train,Inputs_folder_test,Labels_folder_test,0.1)

# # 测试数据集读取和图像显示
# dataset=ImagePairDataset(Inputs_folder,Labels_folder)
# train_iter = DataLoader(dataset, 1, shuffle=True)
# for X, y in train_iter:
#     X = X.to(device)
#     X = torchvision.utils.make_grid(X).numpy()
#     X = X.astype(np.uint8)
#     plt.imshow(X)
#     plt.show()
#     plt.pause(0.5)
#     y = y.to(device)
#     y = torchvision.utils.make_grid(y).numpy()
#     y = y.astype(np.uint8)
#     plt.imshow(y)
#     plt.show()
#     plt.pause(0.5)

# 训练网络
net=SRCNN()
lr, num_epochs = 0.001, 10000
batch_size = 128
optim=torch.optim.Adam(net.parameters(),lr=lr)
loss = torch.nn.MSELoss()

train_dataset=ImagePairDataset(Inputs_folder_train,Labels_folder_train)
test_dataset=ImagePairDataset(Inputs_folder_test,Labels_folder_test)

train_iter = DataLoader(train_dataset, batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size, shuffle=True)

train(train_iter, test_iter, net, loss, optim, device, num_epochs)

# 保存网络有两种方法：1.仅保留参数；2.保存整个网络
# 建议只保存参数减小空间加快加载
# 路径建议以pt或pth作为后缀
PATH = 'SRCNN.pth'
torch.save(net.state_dict(), PATH)

# 测试网络
I = cv2.imread(r'C:\Work\DiveIntoDLSketches\Datasets\Set5\\LR\1.png')
I = np.asarray(I,np.float32)
I = I.transpose((2,0,1))
I=torch.tensor(I)
I=I.unsqueeze(0)
I=I.to(device)
L=net(I)
L=L.to('cpu')
L=L.squeeze(0)
L=L.detach().numpy()
L = np.transpose(L, (1, 2, 0))
L = L.astype(np.uint8)    
cv2.imshow('',L)
cv2.waitKey(0)