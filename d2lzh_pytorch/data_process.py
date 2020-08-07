import random
import torch
import sys
import torchvision
import zipfile


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
    #     num_worker = 0
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


def load_data_jay_lyrics(root=r"./Datasets"):
    """
    Load Jay's lyrics trainset

    Parameters
    ----------
    root : [regexp], optional
        the root folder, by default r"./Datasets"

    Returns
    -------
    corpus_indices : [list]
        the dataset that every characters has changed to index
    char_to_idx : [list]
        the 'character to index' dictionary of dataset
    idx_to_char : [list]
        the 'index to character' dictionary of dataset
    vocab_size : [int]
        the amount of non-repeating characters
    """
    # get the zip first
    with zipfile.ZipFile(root+"/JaychouLyrics/jaychou_lyrics.txt.zip") as zin:
        # unpack and load the file inside
        with zin.open('jaychou_lyrics.txt') as f:
            # load the corpus file as array
            corpus_chars = f.read().decode('utf-8')
    # replace the newline mark to blank mark
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    # leave 10000 words to shrink the dataset
    corpus_chars = corpus_chars[:10000]

    # link all the non-repeating data to integer index
    # then make an integer to character list
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    # map characters to index and return
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    """
    Get a random sample batch of data from corpus_indices with batch_size

    Parameters
    ----------
    corpus_indices : [list]
        the dataset that every characters has changed to index
    batch_size : [int]
        the size of a batch
    num_steps : [int]
        the time steps that a batch would contain, which means samples length
    device : [device], optional
        the device we want this function to run on, by default None

    Yields
    -------
    X : [tensor]
        the first batch of this time
    Y : [tensor]
        the next batch of this time, X and Y are couple
    """
    # split 'real' batches, imply that there are num_examples Xs or Ys
    num_examples = (len(corpus_indices)-1)//num_steps
    # know this time how many small batch are we want to get
    epoch_size = num_examples//batch_size
    # list every 'real' batches
    example_indices = list(range(num_examples))
    # shuffle these 'real' batches
    random.shuffle(example_indices)

    # return a real character data of a sample of a batch
    def _data(pos):
        return corpus_indices[pos:pos+num_steps]

    # choose device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(epoch_size):
        # compute the start of a batch
        i = i*batch_size
        batch_indices = example_indices[i:i+batch_size]
        # the first sample of this time
        X = [_data(j*num_steps) for j in batch_indices]
        # the next sample of this time
        Y = [_data(j*num_steps+1) for j in batch_indices]
        # return X and Y
        # Y is one time step faster than X, to be used as the answer
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(
            Y, dtype=torch.float32, device=device)


def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    """
    Get consecutive sample batches of data from corpus_indices with batch_size
    Dataset is splited to many batches, batches is splited to times and every
    time has two samples

    Parameters
    ----------
    corpus_indices : [list]
        the dataset that every characters has changed to index
    batch_size : [int]
        the size of a batch
    num_steps : [int]
        the time steps that a batch would contain, which means samples length
    device : [device], optional
        the device we want this function to run on, by default None

    Yields
    -------
    X : [tensor]
        the first batch of this time
    Y : [tensor]
        the next batch of this time, X and Y are couple
    """
    # choose device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # change device
    corpus_indices = torch.tensor(
        corpus_indices, dtype=torch.float32, device=device)
    # calculate the number of batches
    data_len = len(corpus_indices)
    batch_len = data_len//batch_size
    # split dataset to batches, from zero to all
    indices = corpus_indices[0:batch_size *
                             batch_len].view(batch_size, batch_len)
    # get how many times in a batch would be
    epoch_size = (batch_len-1)//num_steps
    for i in range(epoch_size):
        # compute the start of this time
        i = i*num_steps
        # the first sample of this time
        X = indices[:, i:i+num_steps]
        # the next sample of this time
        Y = indices[:, i+1:i+num_steps+1]
        # return X and Y
        yield X, Y

