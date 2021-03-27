import time
import torch
from torch import nn
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter


def train(train_iter, eval_iter, net, loss, optimizer, num_epochs, scheduler, print_epochs_gap=10, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    net.train()
    net = net.to(device)
    writer = SummaryWriter()
    print("Training on ", str(device))
    # 记录开始时间
    train_start = time.time()
    train_l_sum, start, batch_count = 0.0, time.time(), 0
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 读取数据对
            X = X.to(device)
            y = y.to(device)
            X = X.unsqueeze(1)
            y = y.unsqueeze(1)
            # 预测
            y_hat = net(X)
            # 计算损失
            l = loss(y_hat, y)
            # 反向转播
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # 记录损失和数量
            train_l_sum += l.cpu().item()
            batch_count += 1
        # 每个epoch记录一次当前train_loss
        train_loss=train_l_sum / batch_count
        writer.add_scalar(net._get_name()+'/train',
                          train_loss, epoch)
        # 每隔一段epochs就计算并输出一次loss
        if epoch % print_epochs_gap == 0:
            eval_loss = eval(eval_iter, net, loss)
            remain_time = (time.time() - train_start) * \
                num_epochs/(epoch+1)*(1-(epoch+1)/num_epochs)
            print('epoch %d, train loss %.6f, eval loss %.6f, remaining %.1fs' % (
                epoch + 1, train_loss, eval_loss, remain_time))
            writer.add_scalar(net._get_name()+'/eval', eval_loss, epoch)
            train_l_sum, batch_count = 0.0, 0
        
        # 更新学习率
        scheduler.step(train_loss)

        # 提前终止
        if optimizer.state_dict()['param_groups'][0]['lr']<1e-6:
            print('Terminated in epoch %d, lr= %.9f' %(epoch+1, optimizer.state_dict()['param_groups'][0]['lr']))
            break
    print("Training cost %.1fs" % (time.time() - train_start))


def eval(data_iter, net, loss, eval_num=20, device=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')):
    # 在指定的数据集上测试, eval_num=0时完全测试
    net.eval()
    net = net.to(device)
    l_sum, batch_count = 0.0, 0
    for X, y in data_iter:
        # 读取数据对
        X = X.to(device)
        y = y.to(device)
        X = X.unsqueeze(1)
        y = y.unsqueeze(1)
        # 预测
        y_hat = net(X)
        # 计算损失
        l = loss(y_hat, y)
        # 记录损失和数量
        l_sum += l.cpu().item()
        batch_count += 1
        if eval_num != 0 and batch_count >= eval_num:
            break
    net.train()
    return l_sum/batch_count


def apply_net(image_path, target_path, net, ratio=2, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # 应用图片并写入
    X = cv2.imread(image_path).astype(np.float32)
    # 普通上采样作为基底
    up_base = cv2.resize(
        X, (X.shape[1]*ratio, X.shape[0]*ratio), interpolation=cv2.INTER_CUBIC)
    # 裁剪ratio个像素, 部分算法需要使用
    up_base=up_base[0:up_base.shape[0]-(ratio-1),0:up_base.shape[1]-(ratio-1),:]
    up_base = cv2.normalize(up_base, up_base, 0, 1, cv2.NORM_MINMAX)
    up_base = cv2.cvtColor(up_base, cv2.COLOR_BGR2YCrCb)
    X = cv2.normalize(X, X, 0, 1, cv2.NORM_MINMAX)
    X = cv2.cvtColor(X, cv2.COLOR_BGR2YCrCb)
    X_y = X[:, :, 0]
    X_y = torch.tensor(X_y)
    X_y = X_y.unsqueeze(0)
    X_y = X_y.unsqueeze(1)
    net = net.to(device)
    X_y = X_y.to(device)
    # 预测
    y_hat_y = net(X_y)
    y_hat_y = y_hat_y.to('cpu')
    y_hat_y = y_hat_y.detach().numpy()
    y_hat = up_base
    y_hat[:, :, 0] = y_hat_y[0, 0, :, :]
    y_hat = cv2.cvtColor(y_hat, cv2.COLOR_YCrCb2RGB)
    y_hat = np.clip(y_hat, 0, 1)
    plt.imshow(y_hat.astype('float32'))
    # plt.show()

    Image.fromarray((y_hat*255).astype(np.uint8)).save(target_path)
    print('Saved: '+target_path)

def apply_net_preupsample(image_path, target_path, net, ratio=2, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # 需要对图像提前上采样的测试版本
    X = cv2.imread(image_path).astype(np.float32)
    # 普通上采样作为基底
    up_base = cv2.resize(
        X, (X.shape[1]*ratio, X.shape[0]*ratio), interpolation=cv2.INTER_CUBIC)
    X=up_base
    # 裁剪ratio个像素, 部分算法需要使用
    # up_base=up_base[0:up_base.shape[0]-(ratio-1),0:up_base.shape[1]-(ratio-1),:]
    up_base = cv2.normalize(up_base, up_base, 0, 1, cv2.NORM_MINMAX)
    up_base = cv2.cvtColor(up_base, cv2.COLOR_BGR2YCrCb)
    X = cv2.normalize(X, X, 0, 1, cv2.NORM_MINMAX)
    X = cv2.cvtColor(X, cv2.COLOR_BGR2YCrCb)
    X_y = X[:, :, 0]
    X_y = torch.tensor(X_y)
    X_y = X_y.unsqueeze(0)
    X_y = X_y.unsqueeze(1)
    net = net.to(device)
    X_y = X_y.to(device)
    # 预测
    y_hat_y = net(X_y)
    y_hat_y = y_hat_y.to('cpu')
    y_hat_y = y_hat_y.detach().numpy()
    y_hat = up_base
    y_hat[:, :, 0] = y_hat_y[0, 0, :, :]
    y_hat = cv2.cvtColor(y_hat, cv2.COLOR_YCrCb2RGB)
    y_hat = np.clip(y_hat, 0, 1)
    plt.imshow(y_hat.astype('float32'))
    # plt.show()

    Image.fromarray((y_hat*255).astype(np.uint8)).save(target_path)
    print('Saved: '+target_path)