import time
import torch
from torch import nn
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# VDSR, 应用了残差网络达成深度学习, 改进的SRCNN, 2016
# 思路接近VGG11
class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.body=VDSR_Block()
    
    def forward(self, img):
        img=self.body(img)
        for i in range(img.shape[0]):
            img[i,0,:,:]-=torch.mean(img[i,0,:,:])
        return img
        
class VDSR_Block(nn.Module):
    def __init__(self):
        super(VDSR_Block, self).__init__()
        self.inp=nn.Conv2d(1,64,3,bias=False,padding=1)
        seq=[]
        for j in range(20):
            seq.append(nn.Conv2d(64,64,3,padding=1))
            seq.append(nn.ReLU(True))
        self.conv=nn.Sequential(*seq)
        self.out=nn.Conv2d(64,1,3,padding=1)
    
    def forward(self, img):
        img=torch.relu(self.inp(img))
        img=self.conv(img)
        img=self.out(img)
        return img

def train(train_iter, test_iter, net, loss, optimizer, num_epochs,scheduler,print_epochs_gap=10, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    net.train()
    net = net.to(device)
    print("Training on ", str(device))
    # 记录开始时间
    train_start=time.time()
    train_l_sum,start,batch_count =  0.0, time.time(),0
    # 初始化写入器
    writer = SummaryWriter()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 读取数据对
            X = X.to(device)
            y = y.to(device)
            X=X.unsqueeze(1)
            y=y.unsqueeze(1)
            # 预测
            y_hat = net(X)
            # 计算损失
            l = loss(y_hat, y-X)
            # 反向转播
            optimizer.zero_grad()
            l.backward()
            # 梯度裁剪
            nn.utils.clip_grad_value_(net.parameters(), 1/optimizer.param_groups[0]['lr'])
            optimizer.step()
            # 记录损失和数量
            train_l_sum += l.item()
            batch_count += 1
    
        # 每个epoch记录一次当前train_loss
        writer.add_scalar('loss/train', train_l_sum / batch_count, epoch)

        # 每隔一段epochs就计算并输出一次loss
        if epoch%print_epochs_gap==0:
            test_loss=eval(test_iter,net,loss)
            print('epoch %d, train loss %.4f, test loss %.4f, time %.1f sec' % (
            epoch + 1, train_l_sum / batch_count,test_loss, time.time() - start))
            writer.add_scalar('loss/test', test_loss, epoch)
            train_l_sum,start,batch_count =  0.0, time.time(),0
        scheduler.step()
    print("Train in %.1f sec"% (time.time() - train_start))

def eval(data_iter, net,loss, eval_num=10,device=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')):
    # 在指定的数据集上测试, eval_num=0时完全测试
    with torch.no_grad():
        net.eval()
        net = net.to(device)
        l_sum,batch_count =  0.0,0
        for X, y in data_iter:
            # 读取数据对
            X = X.to(device)
            y = y.to(device)
            X=X.unsqueeze(1)
            y=y.unsqueeze(1)
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

def apply_net(image_path, target_path, net,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # 应用完整图片并写入
    X=cv2.imread(image_path).astype(np.float32)
    X = cv2.normalize(X,  X, 0, 1, cv2.NORM_MINMAX)
    X = cv2.cvtColor(X, cv2.COLOR_BGR2YCrCb)
    X_y = X[:,:,0]
    X_y=torch.tensor(X_y)
    X_y=X_y.unsqueeze(0)
    X_y=X_y.unsqueeze(1)
    net = net.to(device)
    X_y=X_y.to(device)
    # 预测
    y_hat_y=net(X_y)
    y_hat_y+=X_y
    y_hat_y=y_hat_y.to('cpu')
    y_hat_y=y_hat_y.detach().numpy()
    y_hat=X
    y_hat[:,:,0]=y_hat_y[0,0,:,:]
    y_hat = cv2.cvtColor(y_hat, cv2.COLOR_YCrCb2RGB)
    y_hat = np.clip(y_hat, 0, 1)
    plt.imshow(y_hat.astype('float32'))
    plt.show()
    Image.fromarray((y_hat*255).astype(np.uint8)).save(target_path)
    print('Saved: '+target_path)