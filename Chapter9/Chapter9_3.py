# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import plot

import torch
from PIL import Image

"""
这一节介绍了计算机视觉中的目标检测问题，简单说就是要在一张图中分类框选出多个目标
"""

# 首先尝试绘制测试图
plot.set_figsize()
img = Image.open(r"./Datasets"+'/Img/catdog.jpg')
plot.plt.imshow(img)
plot.plt.show()

# 所谓边界框就是圈出目标的方框，是训练的目标,这里手动指定两个边界框，是框的四个点的坐标
dog_bbox, cat_bbox = [60, 45, 378, 516],\
    [400, 112, 655, 493]

# 保存一个将数组标注的边界框转为可被绘制的矩形方框的函数bbox_to_rect
# 然后尝试绘制上面坐标标注的框
fig = plot.plt.imshow(img)
fig.axes.add_patch(plot.bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(plot.bbox_to_rect(cat_bbox, 'red'))
plot.plt.show()
