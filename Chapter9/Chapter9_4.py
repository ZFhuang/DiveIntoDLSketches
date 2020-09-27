# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import plot

from mxnet import contrib,gluon,image,nd
import numpy as np

"""
这一节介绍如何利用锚框来进行目标检测
"""

# 首先先尝试生成多个锚框,要注意的一点是下面的每个像素是对应了多个锚框的
# 修改numpy的打印精度为2位
np.set_printoptions(2)

# 读取图片
img = image.imread(r"./Datasets"+'/Img/catdog.jpg').asnumpy()
h, w = img.shape[0:2]
print(h, w)
# 构造输⼊数据，这里先构造出图片尺寸X
X = nd.random.uniform(shape=(1, 3, h, w))
# 然后输入希望的锚框的大小和宽高比，配合图片尺寸X，会自动返回所有生成的锚框
# 下面出现了三个尺寸指的是有三种可能的尺寸，三个宽高比是三种形状，是(3+3-1)种锚框(见书)
# 这里的尺寸和宽高比都是先验的式子而已，一个锚框的真正宽为ws*sqrt(r)，高为hs/sqrt(r)

# 返回的尺寸为(批量大小，锚框个数，4)，4表示描述锚框左上角和右下角的两个坐标，坐标以对
# 应图像宽高的比例来表示
Y = contrib.nd.MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)
# 再然后将Y变形为容易使用的情况，hw是图像本身的高宽，因锚框填满每个像素，5是锚框的种类
boxes = Y.reshape((h, w, 5, 4))
# 然后我们可以用坐标和序号来很方便地得到锚框的左上和右下的坐标(x,y,x,y)
print(boxes[250, 250, 0, :])

# # 接着保存一个绘制多个锚框的函数show_bboxes,然后尝试进行绘制
# plot.set_figsize()
# 坐标比率，为了将坐标恢复为原始坐标便于绘制
bbox_scale = nd.array((w, h, w, h))
# fig = plot.plt.imshow(img)
# # 这里绘制了250，250处的所有锚框
# # 注意这里的坐标乘上了一个比率，这是为了从比率坐标恢复为原始坐标
# plot.show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
#                  ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1',
#                   's=0.75,r=2', 's=0.75, r=0.5'])
# plot.plt.show()
print('————————————————————————————')

# 测试计算锚框的偏移值，为之后对锚框位置进行训练做准备
# 尝试绘制两个真实锚框和五个待判断的假锚框来测试，其标签不同
ground_truth = nd.array(
    [[0, 0.1, 0.08, 0.52, 0.92], [1, 0.55, 0.2, 0.9, 0.88]])
anchors = nd.array([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])
fig = plot.plt.imshow(img)
plot.show_bboxes(
    fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog',  'cat'], 'k')
plot.show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
# plot.plt.show()

# 用MultiBoxTarget自动利用待测锚框和真值框为锚框标注标签，得到比对的结果
# 第三个参数1，3，5是 批量大小，包括背景的类别数，锚框数
# 第一维是锚框的四个偏移项，负类锚框为0
# 第二维是掩码，指出那些是负类别，将负类锚框掩为0了
# 第三维是标注的类别，0是背景，1，2...是目标类别索引(狗，猫)
labels = contrib.nd.MultiBoxTarget(anchors.expand_dims(
    axis=0), ground_truth.expand_dims(axis=0), nd.zeros((1, 3, 5)))
print(labels)
print('————————————————————————————')

