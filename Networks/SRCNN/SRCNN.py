import torch
import matplotlib.pyplot as plt
from torch import nn, optim
import cv2
import numpy as np
from PIL import Image

# SRCNN, 最早的超分辨率卷积神经网络, 2014
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # 三个卷积层
        # O=（I-K+2P）/S+1
        # 三层的大小都是不变的, 通道数在改变
        self.conv1=nn.Conv2d(1,64,9, padding=4)
        self.conv2=nn.Conv2d(64,32,1, padding=0)
        self.conv3=nn.Conv2d(32,1,5, padding=2)
    
    def forward(self, img):
        # 三层的学习率不同
        # 两个激活层
        img=torch.relu(self.conv1(img))
        img=torch.relu(self.conv2(img))
        # 注意最后一层不要激活
        return self.conv3(img)

# def train(train_iter, test_iter, net, loss, optimizer, num_epochs,print_epochs_gap=10, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
#     net.train()
#     net = net.to(device)
#     writer = SummaryWriter()
#     print("Training on ", str(device))
#     # 记录开始时间
#     train_start=time.time()
#     train_l_sum,start,batch_count =  0.0, time.time(),0
#     for epoch in range(num_epochs):
#         for X, y in train_iter:
#             # 读取数据对
#             X = X.to(device)
#             y = y.to(device)
#             X=X.unsqueeze(1)
#             y=y.unsqueeze(1)
#             # 预测
#             y_hat = net(X)
#             # 计算损失
#             l = loss(y_hat, y)
#             # 反向转播
#             optimizer.zero_grad()
#             l.backward()
#             optimizer.step()
#             # 记录损失和数量
#             train_l_sum += l.cpu().item()
#             batch_count += 1
#         # 每个epoch记录一次当前train_loss
#         writer.add_scalar('loss/train', train_l_sum / batch_count, epoch)

#         # 每隔一段epochs就计算并输出一次loss
#         if epoch%print_epochs_gap==0:
#             test_loss=eval(test_iter,net,loss)
#             print('epoch %d, train loss %.4f, test loss %.4f, time %.1f sec' % (
#             epoch + 1, train_l_sum / batch_count,test_loss, time.time() - start))
#             writer.add_scalar('loss/test', test_loss, epoch)
#             train_l_sum,start,batch_count =  0.0, time.time(),0
#     print("Train in %.1f sec"% (time.time() - train_start))

# def eval(data_iter, net,loss, eval_num=3,device=torch.device(
#         'cuda' if torch.cuda.is_available() else 'cpu')):
#     # 在指定的数据集上测试, eval_num=0时完全测试
#     net.eval()
#     net = net.to(device)
#     l_sum,batch_count =  0.0,0
#     for X, y in data_iter:
#         # 读取数据对
#         X = X.to(device)
#         y = y.to(device)
#         X=X.unsqueeze(1)
#         y=y.unsqueeze(1)
#         # 预测
#         y_hat = net(X)
#         # 计算损失
#         l = loss(y_hat, y)
#         # 记录损失和数量
#         l_sum += l.cpu().item()
#         batch_count += 1
#         if eval_num!=0 and batch_count>=eval_num:
#             break
#     net.train()
#     return l_sum/batch_count

# def apply_net(image_path, target_path, net,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
#     # 应用完整图片并写入
#     X=cv2.imread(image_path).astype(np.float32)
#     X = cv2.normalize(X,  X, 0, 1, cv2.NORM_MINMAX)
#     X = cv2.cvtColor(X, cv2.COLOR_BGR2YCrCb)
#     X_y = X[:,:,0]
#     X_y=torch.tensor(X_y)
#     X_y=X_y.unsqueeze(0)
#     X_y=X_y.unsqueeze(1)
#     net = net.to(device)
#     X_y=X_y.to(device)
#     # 预测
#     y_hat_y=net(X_y)
#     y_hat_y=y_hat_y.to('cpu')
#     y_hat_y=y_hat_y.detach().numpy()
#     y_hat=X
#     y_hat[:,:,0]=y_hat_y[0,0,:,:]
#     y_hat = cv2.cvtColor(y_hat, cv2.COLOR_YCrCb2RGB)
#     y_hat = np.clip(y_hat, 0, 1)
#     plt.imshow(y_hat.astype('float32'))
#     # plt.show()
#     Image.fromarray((y_hat*255).astype(np.uint8)).save(target_path)
#     print('Saved: '+target_path)