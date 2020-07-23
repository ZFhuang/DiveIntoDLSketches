import random
import torch


def data_iter(batch_size, features, labels):
    """
    Get a batch of data from the parameter features and labels

    Parameters
    ----------
    batch_size : int
        the batch size which you want to return
    features : tensor
        the features of the data inputed, [num*featureNum]
    labels : tensor
        the features of the data inputed, [num*1 size]
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
