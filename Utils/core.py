import time
from numpy.core.numeric import ones
import torch
from torch import nn
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter
import logging


def train(train_iter, eval_iter, net, loss, optimizer, num_epochs, scheduler, need_gclip=False, print_epochs_gap=10, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    logging.info("Start training on: "+str(device))
    net.train()
    net = net.to(device)
    writer = SummaryWriter()
    # 记录开始时间
    train_start = time.time()
    train_l_sum, batch_count = 0.0,  0
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
            if need_gclip:
                # 梯度裁剪
                nn.utils.clip_grad_value_(
                    net.parameters(), 0.01/optimizer.param_groups[0]['lr'])
            optimizer.step()
            # 记录损失和数量
            train_l_sum += l.cpu().item()
            batch_count += 1

        # 每个epoch记录一次当前train_loss
        train_loss = train_l_sum / batch_count
        writer.add_scalar(net._get_name()+'/train', train_loss, epoch)
        logging.debug('epoch %d, train loss %e' % (epoch + 1, train_loss))

        # 每隔一段epochs就计算并输出一次loss
        if epoch % print_epochs_gap == 0:
            if eval_iter==None:
                eval_loss=-1
            else:
                eval_loss = eval(eval_iter, net, loss)
            remain_time = (time.time() - train_start) * \
                num_epochs/(epoch+1)*(1-(epoch+1)/num_epochs)
            logging.info('epoch %d, train loss %e, eval loss %e, lr %e, remaining %.1fs' % (
                epoch + 1, train_loss, eval_loss, optimizer.state_dict()['param_groups'][0]['lr'], remain_time))
            writer.add_scalar(net._get_name()+'/eval', eval_loss, epoch)
            train_l_sum, batch_count = 0.0, 0

        # 更新学习率
        scheduler.step(train_loss)

        # 提前终止
        if optimizer.state_dict()['param_groups'][0]['lr'] <= 1e-6:
            logging.info('Terminated in epoch %d, lr %e' % (
                epoch+1, optimizer.state_dict()['param_groups'][0]['lr']))
            break
    logging.info("Training end. Cost %.1fs" % (time.time() - train_start))


def train_GAN(train_iter, eval_iter, generator, discriminator, loss_gene, loss_disc, optimizer_gene, optimizer_disc, num_epochs, scheduler, need_gclip=False, print_epochs_gap=10, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    logging.info("Start training on: "+str(device))
    generator.train()
    generator = generator.to(device)
    discriminator.train()
    discriminator = discriminator.to(device)
    writer = SummaryWriter()
    # 记录开始时间
    train_start = time.time()
    train_l_gene_sum, train_l_disc_sum, batch_count_disc, batch_count_gene = 0.0, 0.0, 0, 0
    for epoch in range(num_epochs):
        # 训练判别器
        for X, y in train_iter:
            # 读取数据对
            X = X.to(device)
            y = y.to(device)
            X = X.unsqueeze(1)
            y = y.unsqueeze(1)
            y_hat = generator(X)
            input = torch.cat((y_hat, y), dim=0).to(device)
            label = torch.cat(
                (torch.ones(y_hat.shape[0]), (torch.zeros(y_hat.shape[0]))), dim=0).to(device)
            # 预测
            y_hat_disc = discriminator(input)
            # 计算损失
            l_disc = loss_disc(y_hat_disc, label)
            # 反向转播
            optimizer_disc.zero_grad()
            l_disc.backward()
            if need_gclip:
                # 梯度裁剪
                nn.utils.clip_grad_value_(
                    discriminator.parameters(), 0.01/optimizer_disc.param_groups[0]['lr'])
            optimizer_disc.step()
            # 记录损失和数量
            train_l_disc_sum += l_disc.cpu().item()
            batch_count_disc += 1

        # 训练生成器
        for X, y in train_iter:
            # 读取数据对
            X = X.to(device)
            y = y.to(device)
            X = X.unsqueeze(1)
            y = y.unsqueeze(1)
            # 生成器预测
            y_hat = generator(X)

            # 得到判别概率, 越0就越接近真图片
            y_hat_disc = discriminator(y_hat)
            # 计算损失
            l_gene = loss_gene(y_hat, y, y_hat_disc)
            # l_gene = loss_gene(y_hat, y)
            # 反向转播
            optimizer_gene.zero_grad()
            l_gene.backward()
            if need_gclip:
                # 梯度裁剪
                nn.utils.clip_grad_value_(
                    generator.parameters(), 0.01/optimizer_gene.param_groups[0]['lr'])
            optimizer_gene.step()
            # 记录损失和数量
            train_l_gene_sum += l_gene.cpu().item()
            batch_count_gene += 1

        # 每个epoch记录一次当前train_loss
        train_gene_loss = train_l_gene_sum / batch_count_gene
        writer.add_scalar(generator._get_name() +
                          '/train_gene', train_gene_loss, epoch)
        train_disc_loss = train_l_disc_sum / batch_count_disc
        writer.add_scalar(discriminator._get_name() +
                          '/train_disc', train_disc_loss, epoch)
        logging.debug('epoch %d, generate loss %e, discriminate loss %e' % (
            epoch + 1, train_gene_loss, train_disc_loss))

        # 每隔一段epochs就计算并输出一次loss
        if epoch % print_epochs_gap == 0:
            eval_loss = eval(eval_iter, generator, nn.MSELoss())
            remain_time = (time.time() - train_start) * \
                num_epochs/(epoch+1)*(1-(epoch+1)/num_epochs)
            logging.info('epoch %d, generate loss %e, discriminate loss %e, MSEloss %e, lr %e, remaining %.1fs' % (
                epoch + 1, train_gene_loss, train_disc_loss, eval_loss, optimizer_gene.state_dict()['param_groups'][0]['lr'], remain_time))
            train_l_gene_sum, train_l_disc_sum, batch_count_disc, batch_count_gene = 0.0, 0.0, 0, 0

        # 更新学习率
        scheduler.step(train_gene_loss)

        # 提前终止
        if optimizer_gene.state_dict()['param_groups'][0]['lr'] <= 1e-6:
            logging.info('Terminated in epoch %d, lr %e' % (
                epoch+1, optimizer_gene.state_dict()['param_groups'][0]['lr']))
            break
    logging.info("Training end. Cost %.1fs" % (time.time() - train_start))


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
    # 直接应用图片并写入
    X = cv2.imread(image_path).astype(np.float32)
    # 普通上采样作为基底
    up_base = cv2.resize(
        X, (X.shape[1]*ratio, X.shape[0]*ratio), interpolation=cv2.INTER_CUBIC)
    # 裁剪ratio个像素, 部分算法由于反卷积核尺寸设置问题需要使用
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
    y_hat_y = (y_hat_y+1.0)/2.0
    y_hat_y = y_hat_y.to('cpu')
    y_hat_y = y_hat_y.detach().numpy()
    y_hat = up_base
    y_hat[:, :, 0] = y_hat_y[0, 0, :, :]
    y_hat = cv2.cvtColor(y_hat, cv2.COLOR_YCrCb2RGB)
    y_hat = np.clip(y_hat, 0, 1)
    plt.imshow(y_hat.astype('float32'))
    # plt.show()

    Image.fromarray((y_hat*255).astype(np.uint8)).save(target_path)
    logging.info('Saved: '+target_path)


def apply_net_preupsample(image_path, target_path, net, ratio=2, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # 需要对图像提前上采样的测试版本
    X = cv2.imread(image_path).astype(np.float32)
    # 普通上采样作为基底
    up_base = cv2.resize(
        X, (X.shape[1]*ratio, X.shape[0]*ratio), interpolation=cv2.INTER_CUBIC)
    X = up_base
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
    y_hat_y = (y_hat_y+1.0)/2.0
    y_hat_y = y_hat_y.to('cpu')
    y_hat_y = y_hat_y.detach().numpy()
    y_hat = up_base
    y_hat[:, :, 0] = y_hat_y[0, 0, :, :]
    y_hat = cv2.cvtColor(y_hat, cv2.COLOR_YCrCb2RGB)
    y_hat = np.clip(y_hat, 0, 1)
    plt.imshow(y_hat.astype('float32'))
    # plt.show()

    Image.fromarray((y_hat*255).astype(np.uint8)).save(target_path)
    logging.info('Saved: '+target_path)


def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    https://www.pythonf.cn/read/5242
    返回双线性插值核,用来初始化反卷积层中卷积核的参数
    '''
    #先生成一层双线性插值核
    kernel_size = kernel_size[0]
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    fliter = (1 - abs(og[0] - center) / factor) * \
        (1 - abs(og[1] - center) / factor)
    #赋值到每个卷积核的每个通道
    weight = np.zeros((in_channels, out_channels, kernel_size,
                       kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = fliter
    return torch.from_numpy(weight)


def weights_init_kaiming(net):
    # kaiming初始化网络参数
    if isinstance(net, nn.Conv2d):
        torch.nn.init.kaiming_normal_(net.weight)
        if net.bias is not None:
            torch.nn.init.zeros_(net.bias)
    if isinstance(net, nn.ConvTranspose2d):
        net.weight.data = bilinear_kernel(
            net.in_channels, net.out_channels, net.kernel_size)
