import torch


def linreg(X, w, b):
    """
    Return the matrix multiply result of X*w+b

    Parameters
    ----------
    X : tensor
        the variablies of the question
    w : tensor
        the weight of this linear reg
    b : tensor
        the bias of this linear reg

    Returns
    -------
    tensor
        the matrix multiply result of X*w+b
    """
    # mm means matrix multiply
    return torch.mm(X, w)+b


def squared_loss(y_hat, y):
    """
    Compute the squared loss of y_hat and y

    Parameters
    ----------
    y_hat : tensor
        the predictive value of the learning model
    y : tensor
        the real value of the learning model

    Returns
    -------
    tensor
        the squared loss of y_hat and y
    """
    # 'y_hat-y.view(y_hat.size())' is the original losses
    # **2 means square operation
    # devide by 2 to adapt the gradient compution
    # if you use the MSELoss instead, MSELoss did not divide by 2
    return (y_hat-y.view(y_hat.size()))**2/2


def sgd(params, learning_rate, batch_size):
    """
    The optimization function: Stochastic gradient descent (SGD)
    track the params' gradient and multiplied by the learning rate and then
    divided by the batch_size to gain the real changing value of params

    Parameters
    ----------
    params : tensor
        the w and b of the model to be learned
    learning_rate : float
        the value of the params' changing speed
    batch_size : int
        the number of elements of a batch
    """
    for p in params:
        p.data -= learning_rate*p.grad/batch_size
