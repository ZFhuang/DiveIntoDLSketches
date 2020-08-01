import random
import torch
import sys
import torchvision


def data_iter(batch_size, features, labels):
    """
    Get a batch of data from the parameter features and labels

    Parameters
    ----------
    batch_size : [int]
        the batch size which you want to return
    features : [tensor]
        the features of the data inputed, num*featureNum
    labels : [tensor]
        the features of the data inputed, num*1 size
    """
    # the length of labels infer the number of all samples
    num = len(labels)
    # init every index of samples
    indices = list(range(num))
    # shuffle indices to simulate the random sample selection
    random.shuffle(indices)
    # select a index i with a step of batch_size
    for i in range(0, num, batch_size):
        # load the next batch into the tensor j
        j = torch.LongTensor(indices[i:min(i+batch_size, num)])
        # return the batch
        # the function with yield will become a generator not a function
        # every time we call next() to it, will return a yield's content
        yield torch.index_select(features, 0, j), torch.index_select(labels, 0, j)


def get_fashion_mnist_labels(labels):
    """
    Change the label index to real label name

    Parameters
    ----------
    labels : [int list]
        the label indices of FashionMnistDataset, should use loop to load

    Returns
    -------
    [string]
        return the real name of the data index
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress',
                   'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    # for every labels, change it to text in loop
    return [text_labels[int(i)] for i in labels]


def load_data_fashion_mnist(batch_size, resize=None, root=r"./Datasets"):
    """
    Load trainset and testset with batch_size settings

    Parameters
    ----------
    batch_size : [int]
        the size of a batch
    resize : [int]
        the resize value want to apply on fashion_mnist's dataset
    root : [string]
        the root path of the dataset

    Returns
    -------
    [tensor, tensor]
        the train_iter and test_iter of dataset
    """
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    # get or download the dataset
    mnist_train = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform)

    # multi process settings
    # here looks like a little bug in Windows
    # if sys.platform.startswith('win'):
    #     num_worker = 1
    # else:
    #     num_worker = 4
    num_worker =4
    # load data by DataLoader
    train_iter = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    test_iter = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    
    return train_iter, test_iter


def evaluate_accuracy(data_iter, net, device=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Compute the accuracy of a net

    Parameters
    ----------
    data_iter : [producer]
        using for-loop inside to get the X and y from dataset
    net : [function]
        input the X and get the y_hat, compare with y to get the accuracy
    device : [device]
        the device want to calculate

    Returns
    -------
    [float]
        return the accuracy
    """
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        # if the net is a network module
        if isinstance(net, torch.nn.Module):
            # use eval mode to close dropout
            net.eval()
            # add the correct items together, change target to current device
            acc_sum += (net(X.to(device)).argmax(dim=1) ==
                        y.to(device)).float().sum().cpu().item()
            # back to the train mode
            net.train()
        else:
            # if the net has a parameter called 'is_training'
            # this selection is for our DIY function
            if('is_training' in net.__code__.co_varnames):
                acc_sum += (net(X, is_training=False).argmax(dim=1)
                            == y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        # refresh their number
        n += y.shape[0]
    # compute its average
    return acc_sum/n
