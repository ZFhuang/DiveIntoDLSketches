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
        image = Image.open(datafiles["img"])
        label = Image.open(datafiles["label"])
        I = np.asarray(image,np.float32)
        I = I.transpose((2,0,1))
        L = np.asarray(label,np.float32)
        L = L.transpose((2,0,1))
        # 返回对应的读取信息
        return I.copy(), L.copy()

def init_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("Alloc folder: "+folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)
        print("Reinit folder: "+folder)
    
def sample_images(folder, target_folder,size,method=cv2.INTER_NEAREST):
    for _,_,files in os.walk(folder):
        idx=0
        for file in files:
            image =cv2.imread(os.path.join(folder,file))
            if size!=1:
                image=cv2.resize(image, None, fx=size, fy=size, interpolation=method)
                print(os.path.join(folder,file)+' is bicubiced!')
            cv2.imwrite(target_folder+'/'+'img_'+str(idx)+'.png',image)
            idx+=1

def align_images(LR_folder,HR_folder, target_folder,method=cv2.INTER_CUBIC):
    for _,_,files in os.walk(LR_folder):
        idx=0
        for file in files:
            image =cv2.imread(os.path.join(LR_folder,file))
            target_image =cv2.imread(os.path.join(HR_folder,file))
            image=cv2.resize(image, (target_image.shape[1],target_image.shape[0]), interpolation=method)
            print(os.path.join(LR_folder,file)+' is bicubiced!')
            cv2.imwrite(target_folder+'/'+'img_'+str(idx)+'.png',image)
            idx+=1

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
    # 随机将一定比例的文件移动到另一文件夹， 且是对应数量的
    for _,_,files in os.walk(Inputs_folder_train):
        file_num=len(files)
        sample_num=int(file_num*ratio)
        samples=random.sample(range(0,file_num),sample_num)
        for i in samples:
            file_name=files[i]
            shutil.move(Inputs_folder_train+'/'+file_name,Inputs_folder_test+'/'+file_name)
            shutil.move(Labels_folder_train+'/'+file_name,Labels_folder_test+'/'+file_name)

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
    y_hat=y_hat.to('cpu')
    y_hat=y_hat.squeeze(0)
    y_hat=y_hat.detach().numpy()
    y_hat = np.transpose(y_hat, (1, 2, 0))
    y_hat = y_hat.astype(np.uint8)
    plt.imshow(y_hat)
    plt.show()
    Image.fromarray(y_hat).save(target_path)
    print('Saved: '+target_path)