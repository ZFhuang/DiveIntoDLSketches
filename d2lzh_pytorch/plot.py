from IPython import display
from matplotlib import pyplot as plt
import numpy as np


def use_svg_display():
    """
    Use svg formats to display things ploted
    """
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """
    Set the plot view size to figsize

    Parameters
    ----------
    figsize : tuple, optional
        the size of the plot view, length and width, [by default (3.5, 2.5)]
    """
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def show_fashion_mnist(images, labels):
    """
    Plot some data images of FashionMnistDataset in one line with labels

    Parameters
    ----------
    image : [tensor]
        the features or pixels of the image inputed
    labels : [string]
        the real name of those image to be shown
    """
    use_svg_display()
    # the '_' here means we don't need that parameter
    # here init a figure to plot images
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        # show the image
        f.imshow(img.view(28, 28).numpy())
        # set title
        f.set_title(lbl)
        # hide the x and y axis
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    # show the plot figure
    plt.show()


def xyplot(x_vals,y_vals,name):
    """
    Draw a two-dimensional chart of xy

    Parameters
    ----------
    x_vals : [tensor]
        the x values to be plot horizontal
    y_vals : [tensor]
        the y values to be plot vertical
    name : [string]
        the string that will show in the y axis label
    """
    # set the figure's size
    set_figsize(figsize=(5, 2.5))
    # detach() is used to get a variable from the current calculation graph
    # in which this variable is the not gradient tracking version
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    # set the constant x axis label
    plt.xlabel('x')
    # combine and set the y axis label
    plt.ylabel(name+'(x)')
    plt.show()


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    """
    plot data into a half log figure

    Parameters
    ----------
    x_vals : [tensor]
        the first graph
    y_vals : [tensor]
        the first graph
    x_label : [string]
        x axis' label
    y_label : [string]
        y axis' label
    x2_vals : [tensor], optional
        the second graph, by default None
    y2_vals : [tensor], optional
        the second graph, by default None
    legend : [legend], optional
        icons represent the data, by default None
    figsize : [tuple], optional
        size of the figure, by default (3.5, 2.5)
    """
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        # legend means icons
        plt.legend(legend)
    plt.show()


def show_trace_2d(f, results):
    """
    plot the trace of 2d function's figure and results

    Parameters
    ----------
    f : [function]
        using to plot the figure of funtion
    results : [tensor]
        positions of input points
    """
    plt.close()
    # draw input points
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    # get the field of figure
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    # draw the contour of function using x1,x2 as step
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def show_images(imgs, num_rows, num_cols, scale=2):
    """
    Show images in one figure

    Parameters
    ----------
    imgs : [Image]
        the images tensor wants to display
    num_rows : [int]
        the row number of figure
    num_cols : [int]
        the colunm number of figure
    scale : [int], optional
        control the size of figure, by default 2

    Returns
    -------
    [array]
        the images in figure
    """
    figsize = (num_cols*scale, num_rows*scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            # show the target image
            axes[i][j].imshow(imgs[i*num_cols+j])
            # set the sub-axis to be invisible
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    # remember to show figure at last
    plt.show()
    return axes


def bbox_to_rect(bbox, color):
    """
    Transform bbox to rectangle which can be plot in a figure.

    Parameters
    ----------
    bbox : [array]
        the top-left ratio position and bottom-right ratio position of this bbox
    color : [string]
        name of color you want to draw a bbox

    Returns
    -------
    [plt.Rectangle]
        the rectagle that can be draw in a plt-figure
    """
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2
    )


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """
    Draw some bboxes into plt figure.

    Parameters
    ----------
    axes : [imshow.axes]
        the axes of figure
    bboxes : [array]
        the top-left ratio position and bottom-right ratio position of these bboxes
    labels : [string], optional
        names of these bboxes, by default None
    colors : [string], optional
        names of color you want to draw for these bboxes, by default None
    """
    def _make_list(obj, default_values=None):
        # init obj to a list or nothing
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    # init labels and color
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    # for each bbox
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().cpu().numpy(), color)
        axes.add_patch(rect)
        # set labels
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center',
                      fontsize=9, color=text_color, bbox=dict(facecolor=color, lw=0))
