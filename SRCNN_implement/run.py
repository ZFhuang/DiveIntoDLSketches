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
print("Running in "+str(device))

# 生成可用数据集
LR_folder=r'./Datasets/Set5/LR'
HR_folder=r'./Datasets/Set5/HR'

Inputs_folder_train=r'./Datasets/Set5/Inputs_train'
Labels_folder_train=r'./Datasets/Set5/Labels_train'

Inputs_folder_test=r'./Datasets/Set5/Inputs_test'
Labels_folder_test=r'./Datasets/Set5/Labels_test'

Outputs_folder=r'./Datasets/Set5/Outputs'

# # 采样并复制图像
# bicubic_images(LR_folder, Inputs_folder_train,2)
# bicubic_images(HR_folder, Labels_folder_train,1)

# # 然后将图像分割
# size=32
# stride=size//1
# cut_images(Inputs_folder_train, size, stride)
# cut_images(Labels_folder_train, size, stride)

# # 随机移动文件到测试集中
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
lr, num_epochs = 0.003, 500
batch_size = 4
optim=torch.optim.Adam(net.parameters(),lr=lr)
loss = torch.nn.MSELoss()

train_dataset=ImagePairDataset(Inputs_folder_train,Labels_folder_train)
test_dataset=ImagePairDataset(Inputs_folder_test,Labels_folder_test)

train_iter = DataLoader(train_dataset, batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, 1, shuffle=True)
net.train()
train(train_iter, test_iter, net, loss, optim, device, num_epochs)

# 保存网络有两种方法：1.仅保留参数；2.保存整个网络
# 建议只保存参数减小空间加快加载
# 路径建议以pt或pth作为后缀
PATH = 'Models/SRCNN.pth'
torch.save(net.state_dict(), PATH)

# 图形化测试
net.load_state_dict(torch.load(PATH))
net.eval()
net=net.to(device)

# for X, y in train_iter:
#     X = X.to(device)
#     y = y.to(device)
#     y_hat = net(X)

#     l = loss(y_hat, y)

#     X=X.to('cpu')
#     X=X.squeeze(0)
#     X=X.detach().numpy()
#     X=X[0,:,:]
#     X = np.transpose(X, (1, 2, 0))
#     X = X.astype(np.uint8)    
#     cv2.imshow('',X)
#     cv2.waitKey(0)

#     y=y.to('cpu')
#     y=y.squeeze(0)
#     y=y.detach().numpy()
#     y=y[0,:,:]
#     y = np.transpose(y, (1, 2, 0))
#     y = y.astype(np.uint8)    
#     cv2.imshow('',y)
#     cv2.waitKey(0)

#     y_hat=y_hat.to('cpu')
#     y_hat=y_hat.squeeze(0)
#     y_hat=y_hat.detach().numpy()
#     y_hat=y_hat[0,:,:]
#     y_hat = np.transpose(y_hat, (1, 2, 0))
#     y_hat = y_hat.astype(np.uint8)    
#     cv2.imshow('',y_hat)
#     cv2.waitKey(0)

#     l=l.cpu().item()
#     print(l)

# 实际测试
X=cv2.imread(r'C:\Work\DiveIntoDLSketches\Datasets\Set5\\LR\1.png')
X = np.asarray(X,np.float32)
X = X.transpose((2,0,1))
X=torch.tensor(X)
X=X.unsqueeze(0)
X = X.to(device)
y_hat = net(X)
y_hat=y_hat.to('cpu')
y_hat=y_hat.squeeze(0)
y_hat=y_hat.detach().numpy()
y_hat = np.transpose(y_hat, (1, 2, 0))
y_hat = y_hat.astype(np.uint8)
cv2.imshow('',y_hat)
cv2.waitKey(0)