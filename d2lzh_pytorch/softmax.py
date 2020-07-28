import torch


def softmax(X):
    """
    The basic of softmax algorithm, compute the exp of every labels then divide 
    these exps by their summary to gain the possible values

    Parameters
    ----------
    X : [tensor]
        a vector that record all labels possible value

    Returns
    -------
    [tensor]
        the vector that has changed to the possible value that can be used
    """
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp/partition


def cross_entropy(y_hat, y):
    """
    Compute the cross entropy as the loss of some nets

    Parameters
    ----------
    y_hat : [tensor]
        the prediction value
    y : [tensor]
        the real labels

    Returns
    -------
    [tensor]
        the vector of every cross entropy loss. They are negetive to be optimized
    """
    # here we use gather() to get the according prediction values
    # then we compute the log of the tensor to satisfy the cross entropy's need
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))
