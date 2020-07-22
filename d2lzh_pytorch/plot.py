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
        the size of the plot view, length and width, by default (3.5, 2.5)
    """
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize