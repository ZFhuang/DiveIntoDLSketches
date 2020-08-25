# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import data_process, plot

from mpl_toolkits import mplot3d # 三维画图
import numpy as np

"""
这一节介绍了一些关于深度学习中优化目标函数的知识
"""


# 首先是由于很多问题在优化途中没有解析解，所以需要用机器学习方法得到数值解
# 而对于数值解，求解过程中容易掉入鞍点或局部最小值，这里是局部最小值的示例
# 函数在x的值比邻近区域都要小就称为局部最小值，此时优化算法可能会被困住
def f(x):
    # 示例函数，f(x) = x*cos(pi*x)
    return x*np.cos(np.pi*x)


plot.set_figsize((4.5, 2.5))
# 列出一组输入用的x
x = np.arange(-1.0, 2.0, 0.1)
# 将输入代入得到一组输出然后绘制到fig上
fig, = plot.plt.plot(x, f(x))
# 绘制两条指示线，指向的点是提前看出来的
fig.axes.annotate('local minium', xy=(-0.3, -0.25),
                  xytext=(-0.77, -1.0), arrowprops=dict(arrowstyle='->'))
fig.axes.annotate('global minium', xy=(1.1, -0.95),
                  xytext=(0.6, 0.8), arrowprops=dict(arrowstyle='->'))
plot.plt.xlabel('x')
plot.plt.ylabel('f(x)')
# plot.plt.show()
print('————————————————————————————')

# 除了局部最小值由于周边梯度为0会卡住外，鞍点的周边梯度也是0，同样会困住优化器
# 鞍点的关键是梯度为0，也就是局部的一个比较平的点，也就可以是中间值最大值之类的
# 用close清除上一次的绘制（不会关闭窗口，仅仅是不影响下次绘制）
plot.plt.close()
x = np.arange(-2.0, 2.0, 0.1)
fig, = plot.plt.plot(x, x**3)
fig.axes.annotate('saddle point', xy=(0, -0.2),
                  xytext=(-0.52, -0.5), arrowprops=dict(arrowstyle='->'))
plot.plt.xlabel('x')
plot.plt.ylabel('f(x)')
# plot.plt.show()
print('————————————————————————————')

# 再尝试二维空间中的鞍点例子，二维坐标对应z值
plot.plt.close()
# 从-1到1的网格型绘制
x,y=np.mgrid[-1:1:31j,-1:1:31j]
z=x**2-y**2
# 在三维中绘制
ax = plot.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 2, 'cstride': 2})
# 在鞍点的位置绘制
ax.plot([0], [0], [0], 'rx')
# xy的刻度标签
ticks = [-1, 0, 1]
plot.plt.xticks(ticks)
plot.plt.yticks(ticks)
# z坐标的刻度设置不太一样
ax.set_zticks(ticks)
plot.plt.xlabel('x')
plot.plt.ylabel('y')
plot.plt.show()
print('————————————————————————————')
