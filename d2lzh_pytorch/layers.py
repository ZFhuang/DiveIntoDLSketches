import torch
import torch.nn as nn
import torch.nn.functional as F


class FlattenLayer(nn.Module):
    """
    The flatten layer class that maybe useful in the future, it can flatten the 
    x to just one dimension

    Parameters
    ----------
    nn : [torch.nn]
        the network module
    """

    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        # the flatten operation. x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


def corr2d(X, K):
    """
    The cross-correlation layer, or named convolution layer. It multiply every 
    corresponding elements and add them together to be the output element.

    Parameters
    ----------
    X : [tensor]
        the original matrix to be calculate
    K : [tensor]
        the kernel matrix of the convolution layer

    Returns
    -------
    [tensor]
        return the output matrix
    """
    h, w = K.shape
    # pre-init
    Y = torch.zeros((X.shape[0]-h+1, X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # for every possible positions
            # get the sub matrix and multiply to kernel then add up
            Y[i, j] = (X[i:i+h, j:j+w]*K).sum()
    return Y


class GlobalAvgPool2d(nn.Module):
    """
    A pooling layer that use global average method which means calculate the of 
    average every elements

    Parameters
    ----------
    nn : [torch.nn]
        the network module
    """

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        # pay attention here, the kernal size is equal to the whole image size
        return torch.nn.functional.avg_pool2d(x, kernel_size=x.size()[2:])


class Residual(nn.Module):
    """
    The residual block of ResNet. It's feature is contain a bypath to persure
    gradients could at least stay still from training.

    Parameters
    ----------
    nn : [Module]
        nn.Module
    """

    def __init__(self, in_channels, out_channels, use_1x1_conv=False, stride=1):
        super(Residual, self).__init__()
        # two convolution layer which have same amount of output channels, using
        # to extract features twice
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1)
        # if we want the bypath, we should use a 1*1 convolution layer to make
        # the channels have same size
        if use_1x1_conv:
            self.conv3 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        # using two batch normalize layer to persure the stability
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        # conv twice
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        # if conv3 exist, means the bypath is exist
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y+X)
