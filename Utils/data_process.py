import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import time
import shutil
import random
import cv2
from torch.utils.data import Dataset

class ImagePairDataset_y(Dataset):
    # from: https://blog.csdn.net/tuiqdymy/article/details/84779716
    # 数据集需要保存在Inputs和Labels文件夹中

    def __init__(self, inputs_folder, labels_folder):
        super(ImagePairDataset_y,self).__init__()
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
        X = cv2.imread(datafiles["img"]).astype(np.float32)
        X = cv2.normalize(X,  X, 0, 1, cv2.NORM_MINMAX)
        X = cv2.cvtColor(X, cv2.COLOR_BGR2YCrCb)
        X_y = X[:,:,0]
        y = cv2.imread(datafiles["label"]).astype(np.float32)
        y = cv2.normalize(y,  y, 0, 1, cv2.NORM_MINMAX)
        y = cv2.cvtColor(y, cv2.COLOR_BGR2YCrCb)
        y_y = y[:,:,0]
        # 返回对应的读取信息
        return X_y.copy(), y_y.copy()

def init_folder(folder):
    # 初始化所需的文件夹
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("Alloc folder: "+folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)
        print("Reinit folder: "+folder)
    
def sample_images(folder, target_folder,ratio,method=cv2.INTER_CUBIC):
    # 采样并保存图像
    for _,_,files in os.walk(folder):
        idx=0
        for file in files:
            image =cv2.imread(os.path.join(folder,file))
            image=cv2.resize(image, (int(image.shape[1]*ratio), int(image.shape[0]*ratio)), interpolation=method)
            # 重命名
            cv2.imwrite(target_folder+'/'+'img_'+str(idx)+'.png',image)
            idx+=1
    print('Sampled folder: '+folder)

def align_images(LR_folder,HR_folder, target_folder,method=cv2.INTER_CUBIC):
    # 对齐图像大小
    for _,_,files in os.walk(LR_folder):
        for file in files:
            image =cv2.imread(os.path.join(LR_folder,file))
            target_image =cv2.imread(os.path.join(HR_folder,file))
            image=cv2.resize(image, (target_image.shape[1],target_image.shape[0]), interpolation=method)
            cv2.imwrite(target_folder+'/'+file,image)
    print('Alingned folder: '+LR_folder)

def crop_images(folder, target_folder,pixel):
    # 裁剪图像右下角用于loss比较
    for _,_,files in os.walk(folder):
        for file in files:
            img=Image.open(os.path.join(folder,file))
            img=np.asarray(img)
            img=img[0:img.shape[0]-pixel,0:img.shape[1]-pixel,:]
            Image.fromarray(img).save(target_folder+'/'+file)
    print('Croped folder: '+folder)

def cut_images(LR_folder, HR_folder, LR_size, ratio):
    # 批量成对裁切图像, ratio是HR比LR的比例
    for _,_,files in os.walk(LR_folder):
        for file in files:
            LR_image =cv2.imread(os.path.join(LR_folder,file))
            HR_image =cv2.imread(os.path.join(HR_folder,file))
            h_num=int(LR_image.shape[0]//LR_size)
            w_num=int(LR_image.shape[1]//LR_size)
            fore_name=os.path.splitext(file)[0]
            back_name=os.path.splitext(file)[-1]
            for i in range(0,h_num):
                for j in range(0,w_num):
                    LR_tmp_img=LR_image[i*LR_size:(i+1)*LR_size,j*LR_size:(j+1)*LR_size,:]
                    HR_tmp_img=HR_image[i*LR_size*ratio:(i+1)*LR_size*ratio,j*LR_size*ratio:(j+1)*LR_size*ratio,:]
                    tmp_name=fore_name+'_'+str(i)+'_'+str(j)+back_name
                    cv2.imwrite(os.path.join(LR_folder,tmp_name),LR_tmp_img)
                    cv2.imwrite(os.path.join(HR_folder,tmp_name),HR_tmp_img)
            os.remove(os.path.join(LR_folder,file))
            os.remove(os.path.join(HR_folder,file))
    print('Cut folder: '+LR_folder+' & '+HR_folder)

def random_move(Inputs_folder_train,Labels_folder_train,Inputs_folder_test,Labels_folder_test,ratio):
    # 随机将一定比例的文件移动到另一文件夹， 且是对应数量的
    for _,_,files in os.walk(Inputs_folder_train):
        file_num=len(files)
        sample_num=int(file_num*ratio)
        samples=random.sample(range(0,file_num),sample_num)
        for i in samples:
            file_name=files[i]
            shutil.move(Inputs_folder_train+'/'+file_name,Inputs_folder_test+'/'+file_name)
            shutil.move(Labels_folder_train+'/'+file_name,Labels_folder_test+'/'+file_name)
    print('Random move completed!')

def expand_dataset(folder, target_folder, scale=1, rotate=0, method=cv2.INTER_CUBIC):
    # 扩张数据集
    # scale是所需的缩放倍率
    # rotate是旋转的四种选项:
    #   0: 不旋转
    #   1: 顺旋转90
    #   2: 顺旋转180
    #   3: 顺旋转270
    if rotate !=0:
        if rotate==1:
            rotate_setting=cv2.ROTATE_90_CLOCKWISE
        elif rotate==2:
            rotate_setting=cv2.ROTATE_180
        elif rotate==3:
            rotate_setting=cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            print('Rotate input ERROR!')
            return
        for _,_,files in os.walk(folder):
            idx=0
            for file in files:
                image =cv2.imread(os.path.join(folder,file))
                image=cv2.rotate(image, rotate_setting)
                image=cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)), interpolation=method)
                # 重命名
                cv2.imwrite(target_folder+'/'+'img_'+str(idx)+'_s'+str(scale)+'_r'+str(rotate)+'.png',image)
                idx+=1
    else:
        for _,_,files in os.walk(folder):
            idx=0
            for file in files:
                image =cv2.imread(os.path.join(folder,file))
                image=cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)), interpolation=method)
                # 重命名
                cv2.imwrite(target_folder+'/'+'img_'+str(idx)+'_s'+str(scale)+'_r'+str(rotate)+'.png',image)
                idx+=1
    print('Expanded folder: '+folder)