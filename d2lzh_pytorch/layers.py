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
