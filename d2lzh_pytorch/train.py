from d2lzh_pytorch import linear_reg
from d2lzh_pytorch import data_process


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    """
    The training function of chapter3, it is a more useful function.
    In Chapter3_6, it is used to train the softmax regconition.

    Parameters
    ----------
    net : [function]
        input X and get y_hat, the main learning program
    train_iter : [producer]
        the train dataset
    test_iter : [producer]
        the test dataset
    loss : [function]
        input y and y_hat, get loss value
    num_epochs : [int]
        how many times do you want to learn through the whole training dataset?
    batch_size : [int]
        size of the batch
    params : [tensor], optional
        the weight and bias combine to be the params tensor, by default None
    lr : [float], optional
        learning rate, by default None
    optimizer : [function], optional
        the function to decrease the gradient, if you have a optimizer, you 
        don't need to input the params and lr before but input them directly 
        to the optimizer, by default None
    """
    for e in range(num_epochs):
        # init
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # depend on whether you have a optimizer
            # clean the grad of params
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for p in params:
                    p.grad.data.zero_()

            l.backward()
            # the function to decrease the gradient
            if optimizer is None:
                linear_reg.sgd(params, lr, batch_size)
            else:
                optimizer.step()

            # gain the loss and acc value of each iter
            train_loss_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        # use those value stored in iter to gain the final value of every epochs
        test_acc = data_process.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.3f, train acc %.3f, test acc %.3f' %
              (e+1, train_loss_sum/n, train_acc_sum/n, test_acc))
