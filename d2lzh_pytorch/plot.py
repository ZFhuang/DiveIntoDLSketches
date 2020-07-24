from IPython import display
from matplotlib import pyplot as plt


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
