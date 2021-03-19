import time
import torch
from torch import nn
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

# VDSR, 应用了残差网络达成深度学习, 改进的SRCNN, 2016
# 思路接近VGG11
class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        seq=[]
        for i in range(2):
            seq.append(VDSR_Block())
        self.loop=nn.Sequential(*seq)
    
    def forward(self, res):
        res=self.loop(res)
        return res

class VDSR_Block(nn.Module):
    def __init__(self):
        super(VDSR_Block, self).__init__()
        self.inp=nn.Conv2d(3,64,3,padding=1)
        seq=[]
        for j in range(62):
            seq.append(nn.Conv2d(64,64,3,padding=1))
            seq.append(nn.ReLU())
        self.conv=nn.Sequential(*seq)
        self.out=nn.Conv2d(64,3,3,padding=1)
    
    def forward(self, img):
        res=img
        res=self.inp(res)
        res=torch.sigmoid(self.conv(res))
        res=self.out(res)
        return res+img

def train(train_iter, test_iter, net, loss, optimizer, num_epochs,print_epochs_gap=10, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    net.train()
    net = net.to(device)
    print("Training on ", str(device))
    # 记录开始时间
    train_start=time.time()
    train_l_sum,start,batch_count =  0.0, time.time(),0
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 读取数据对
            X = X.to(device)
            y = y.to(device)
            # 预测
            y_hat = net(X)
            # 计算损失
            l = loss(y_hat, y-X)
            # 反向转播
            optimizer.zero_grad()
            l.backward()
            # 梯度裁剪
            # nn.utils.clip_grad_norm_(net.parameters(), max_norm=1, norm_type=2)
            optimizer.step()
            # 记录损失和数量
            train_l_sum += l.cpu().item()
            batch_count += 1
        # 每隔一段epochs就计算并输出一次loss
        if epoch%print_epochs_gap==0:
            print('epoch %d, train loss %.4f, test loss %.4f, time %.1f sec' % (
            epoch + 1, train_l_sum / batch_count,eval(test_iter,net,loss), time.time() - start))
            train_l_sum,start,batch_count =  0.0, time.time(),0
    print("Train in %.1f sec"% (time.time() - train_start))

def eval(data_iter, net,loss, eval_num=3,device=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')):
    # 在指定的数据集上测试, eval_num=0时完全测试
    net.eval()
    net = net.to(device)
    l_sum,batch_count =  0.0,0
    for X, y in data_iter:
        # 读取数据对
        X = X.to(device)
        y = y.to(device)
        # 预测
        y_hat = net(X)
        # 计算损失
        l = loss(y_hat, y-X)
        # 记录损失和数量
        l_sum += l.cpu().item()
        batch_count += 1
        if eval_num!=0 and batch_count>=eval_num:
            break
    net.train()
    return l_sum/batch_count

def eval_with_img(data_iter, net,loss, eval_num=10,device=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')):
    # 在指定的数据集上测试并绘制每个测试的数据对
    net.eval()
    net = net.to(device)
    l_sum,batch_count =  0.0,0
    for X, y in data_iter:
        # 读取数据对
        X = X.to(device)
        y = y.to(device)
        # 预测
        y_hat = net(X)
        y_hat=X+y_hat
        # 计算损失
        l = loss(y_hat, y-X)
        # 记录损失和数量
        l_sum += l.cpu().item()
        batch_count += 1
        if eval_num!=0 and batch_count>=eval_num:
            break

        # 显示此迭代中的LR, HR, OUT
        X=X.to('cpu')
        X=X.squeeze(0)
        X=X.detach().numpy()
        # X=X[0,:,:]
        X = np.transpose(X, (1, 2, 0))
        X = X.astype(np.uint8)    
        plt.imshow(X)
        plt.show()

        y=y.to('cpu')
        y=y.squeeze(0)
        y=y.detach().numpy()
        # y=y[0,:,:]
        y = np.transpose(y, (1, 2, 0))
        y = y.astype(np.uint8)    
        plt.imshow(X)
        plt.show()

        y_hat=y_hat.to('cpu')
        y_hat=y_hat.squeeze(0)
        y_hat=y_hat.detach().numpy()
        # y_hat=y_hat[0,:,:]
        y_hat = np.transpose(y_hat, (1, 2, 0))
        y_hat = y_hat.astype(np.uint8)    
        plt.imshow(X)
        plt.show()
    
    net.train()
    return l_sum/batch_count

def apply_net(image_path, target_path, net,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # 应用完整图片并写入
    X=Image.open(image_path)
    X = np.asarray(X,np.float32)
    X = X.transpose((2,0,1))
    X=torch.tensor(X)
    X=X.unsqueeze(0)
    net.eval()
    net = net.to(device)
    X=X.to(device)
    y_hat = net(X)
    y_hat=X+y_hat
    y_hat=y_hat.to('cpu')
    y_hat=y_hat.squeeze(0)
    y_hat=y_hat.detach().numpy()
    y_hat = np.transpose(y_hat, (1, 2, 0))
    y_hat = y_hat.astype(np.uint8)
    plt.imshow(y_hat)
    plt.show()
    Image.fromarray(y_hat).save(target_path)
    print('Saved: '+target_path)