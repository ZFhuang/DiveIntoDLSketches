# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import torch
import math
import numpy as np
from PIL import Image
import sys
sys.path.append(r".")
from d2lzh_pytorch import plot, detection

"""
这一节介绍如何利用锚框来进行目标检测
"""

# 首先先尝试生成多个锚框,要注意的一点是下面的每个像素是对应了多个锚框的
# 修改numpy的打印精度为2位
np.set_printoptions(2)
plot.set_figsize()

# 直接用PIL来读取图片并输出尺寸测试
img = Image.open(r"./Datasets"+'/Img/catdog.jpg')
w, h = img.size
print("w=%d, h=%d" % (w, h))

# 这里保存函数MultiBoxPrior来生成锚框

# 构造输⼊数据，这里先构造出图片尺寸X
X = torch.Tensor(1, 3, h, w)
# 然后输入希望的锚框的大小和宽高比，配合图片尺寸X，会自动返回所有生成的锚框
# 下面出现了三个尺寸指的是有三种可能的尺寸，三个宽高比是三种形状，是(3+3-1)种锚框(见书)
# 这里的尺寸和宽高比都是先验的式子而已，一个锚框的真正宽为ws*sqrt(r)，高为hs/sqrt(r)
# 返回的尺寸为(批量大小，锚框个数，4)，4表示描述锚框左上角和右下角的两个坐标，坐标以对
# 应图像宽高的比例来表示
Y = detection.MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)
# 再然后将Y变形为容易使用的情况，hw是图像本身的高宽，因锚框填满每个像素，5是锚框的种类
boxes = Y.reshape((h, w, 5, 4))
# 然后我们可以用坐标和序号来很方便地得到锚框的左上和右下的坐标(x,y,x,y)
print(boxes[250, 250, 0, :])

# # 接着保存一个绘制多个锚框的函数show_bboxes,然后尝试进行绘制
# plot.set_figsize()
# 坐标比率，为了将坐标恢复为原始坐标便于绘制
bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
# fig = plot.plt.imshow(img)
# # 这里绘制了250，250处的所有锚框
# # 注意这里的坐标乘上了一个比率，这是为了从比率坐标恢复为原始坐标
# plot.show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
#                  ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1',
#                   's=0.75,r=2', 's=0.75, r=0.5'])
# plot.plt.show()
print('————————————————————————————')

# 接下来要计算锚框的交并比来用于训练，也称Jaccard系数，也就是两区域相交区域/相并区域
# 得到交并比后测试计算锚框的偏移值，为之后对锚框位置进行训练做准备
# 尝试绘制两个真实锚框和五个待判断的假锚框来测试，其标签不同
# 这里有大量的函数被写入了detection中
ground_truth = torch.tensor(
    [[0, 0.1, 0.08, 0.52, 0.92], [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                        [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                        [0.57, 0.3, 0.92, 0.9]])
fig = plot.plt.imshow(img)
plot.show_bboxes(
    fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog',  'cat'], 'k')
plot.show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
# plot.plt.show()

# 保存的MultiBoxTarget利用待测锚框和真值框为锚框标注标签，得到比对的结果
# 第三个参数1，3，5是 批量大小，包括背景的类别数，锚框数
# 第一维是锚框的四个偏移项，负类锚框为0
# 第二维是掩码，指出那些是负类别，将负类锚框掩为0了
# 第三维是标注的类别，0是背景，1，2...是目标类别索引(狗，猫)
# 在这里面的锚框配对是依据交并比，首先利用交并比将配对最准确的锚框挑走，剩下的配对中再
# 按照交并比顺序，将阈值以上的部分挑走，阈值以下的部分当作背景来标记
# 然后这里的第一个返回值：偏移量是依据四个边角的偏移值除宽/高来得到的比率偏移
labels = detection.MultiBoxTarget(anchors.unsqueeze(dim=0),
                                  ground_truth.unsqueeze(dim=0))
# 类别标记，用索引号标出
print(labels[2])
# 掩码，将负类别用0标出
print(labels[1])
# 偏移量，负类为0
print(labels[0])
print('————————————————————————————')

# 模型预测的时候会输出多个预测框，但是有时候会有很多个接近的预测框被输出，因此需要用非
# 极大值抑制(NMS)来移除相似的边界框，方法是将所有分类中的锚框按照置信度排列，然后选取里
# 面置信度最高的一个为基准，移除所有交并比在一定水平上的锚框，接着选取剩余的置信度第二的
# 锚框，继续移除，直到所有剩下的锚框都曾经作为基准为止
# 下面这里首先构建几个测试的例子来模拟NMS
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0.0] * (4 * len(anchors)))
cls_probs = torch.tensor([[0., 0., 0., 0., ],  # 背景的预测概率
                          [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                          [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
fig = plot.plt.imshow(img)
plot.show_bboxes(fig.axes, anchors * bbox_scale,
                 ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
# 可以看到有多个置信度相近且本身也很接近的锚框存在
# plot.plt.show()

# 保存了non_max_suppression和MultiBoxDetection用来做非极大值抑制NMS
# 进行抑制
output = detection.MultiBoxDetection(
    cls_probs.unsqueeze(dim=0), offset_preds.unsqueeze(dim=0),
    anchors.unsqueeze(dim=0), nms_threshold=0.5)
# 输出抑制后的锚框结果
print(output)

# 将抑制后的结果绘制到图中
# 由于之前的结果过多所以记得用clf把图清空
plot.plt.clf()
fig = plot.plt.imshow(img)
for i in output[0].detach().cpu().numpy():
    if i[0] == -1:
        continue
    # 输出锚框和标签和置信率
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    plot.show_bboxes(fig.axes, torch.tensor(i[2:]) * bbox_scale, label)
plot.plt.show()
