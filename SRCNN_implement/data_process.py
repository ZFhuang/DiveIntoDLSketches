import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
import time
import shutil
import random
from torch.utils.data import Dataset

class ImagePairDataset(Dataset):
    # from: https://blog.csdn.net/tuiqdymy/article/details/84779716
    # 数据集需要保存在Inputs和Labels文件夹中

    def __init__(self, inputs_folder, labels_folder):
        super(ImagePairDataset,self).__init__()
        # 用来保存数据集的路径集合
        self.files = []
        for _,_,files in os.walk(inputs_folder):
            for file in files:
                # 计算对应文件的路径
                img_file = os.path.join(inputs_folder,file)
                label_file = os.path.join(labels_folder,file)
                # 每个文件都由下面这样的路径对组成
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                })
 
    def __len__(self):
        # 总的文件数量
        return len(self.files)
 
    def __getitem__(self, index):
        # 被DataLoader以下标调用
        datafiles = self.files[index]
        # 得到路径并读取图片
        image = cv2.imread(datafiles["img"])
        label = cv2.imread(datafiles["label"])
        I = np.asarray(image,np.float32)
        I = I.transpose((2,0,1))
        L = np.asarray(label,np.float32)
        L = L.transpose((2,0,1))
        # 返回对应的读取信息
        return I.copy(), L.copy()

def bicubic_images(folder, target_folder,size):
    # 批量双三次图像
    for _,_,files in os.walk(folder):
        # 文件路径
        for file in files:
            image =cv2.imread(os.path.join(folder,file))
            if size!=1:
                image=cv2.resize(image, None, fx=size, fy=size, interpolation=cv2.INTER_CUBIC)
                print(os.path.join(folder,file)+' is bicubiced!')
            cv2.imwrite(os.path.join(target_folder,file),image)

def cut_images(folder, size, stride):
    # 批量裁剪图像
    for _,_,files in os.walk(folder):
        for file in files:
            image =cv2.imread(os.path.join(folder,file))
            h_num=image.shape[0]//stride
            w_num=image.shape[1]//stride
            fore_name=os.path.splitext(file)[0]
            back_name=os.path.splitext(file)[-1]
            for i in range(0,h_num):
                if i*stride+size<=image.shape[0]:
                    for j in range(0,w_num):
                        if j*stride+size<=image.shape[1]:
                            tmp_img=image[i*stride:i*stride+size,j*stride:j*stride+size,:]
                            tmp_name=fore_name+'_'+str(i)+'_'+str(j)+back_name
                            cv2.imwrite(os.path.join(folder,tmp_name),tmp_img)
            print(os.path.join(folder,file)+' is cut!')
            os.remove(os.path.join(folder,file))

def random_move(Inputs_folder_train,Labels_folder_train,Inputs_folder_test,Labels_folder_test,ratio):
    for _,_,files in os.walk(Inputs_folder_train):
        file_num=len(files)
        sample_num=int(file_num*ratio)
        samples=random.sample(range(0,file_num),sample_num)
        for i in samples:
            file_name=files[i]
            shutil.move(os.path.join(Inputs_folder_train,file_name), os.path.join(Inputs_folder_test,file_name))
            shutil.move(os.path.join(Labels_folder_train,file_name), os.path.join(Labels_folder_test,file_name))

def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            n += y.shape[0]
            batch_count += 1
        # test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,time %.1f sec' % (
            epoch + 1, train_l_sum / batch_count, train_acc_sum / n, 0, time.time() - start))

def evaluate_accuracy(data_iter, net, device=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval()
            acc_sum += (net(X.to(device)).argmax(dim=1) ==
                        y.to(device)).float().sum().cpu().item()
            net.train()
        else:
            if('is_training' in net.__code__.co_varnames):
                acc_sum += (net(X, is_training=False).argmax(dim=1)
                            == y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum/n