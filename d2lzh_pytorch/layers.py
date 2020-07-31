import torch
import torch.nn as nn


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
