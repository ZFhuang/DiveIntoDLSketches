import random
import torch
import torch.nn as nn
from d2lzh_pytorch import layers

def to_onehot(X, n_class):
    """
    Batches base one-hot coding

    Parameters
    ----------
    X : [tensor]
        the input batch data, time_step pair * n_batches
    n_class : [int]
        the coding base, means the length of one-hot tensor

    Returns
    -------
    [list]
        change batches to a list that contain those one-hot coding pair by pair
        the list's first axis is time-step, contain X and Y of batches one time
        all data is coded to be one-hot formation
    """
    # one-hot coding is making a integer to a array that only the index equal
    # to this integer will be set by number one, others are zeros
    def one_hot(x, n_class, dtype=torch.float32):
        x = x.long()
        # initiate the result
        res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
        # use scatter to put one to the target position
        # out[index[i, j], j] = value[i, j] dim=0
        # out[i,index[i, j]] = value[i, j]] dim=1
        # this function can return more than one one-hot at a time
        res.scatter_(1, x.view(-1, 1), 1)
        return res

    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


def predict_rnn(prefix, num_chars, RNN, params, init_rnn_state, num_hiddens,
                vocab_size, device, idx_to_char, char_to_idx):
    """
    Input a prefix character and get the rest num_chars characters that are
    predicted by inputed RNN

    Parameters
    ----------
    prefix : [char]
        the start character of this character sequence
    num_chars : [int]
        the number of characters you want to predict
    RNN : [function]
        the recycle neural network
    params : [tuple]
        network's params
    init_rnn_state : [tuple]
        network's start time state
    num_hiddens : [int]
        how many hidden parameters you want to add
    vocab_size : [int]
        the number of non-repeating character in this vocab
    device : [device]
        device you want to run on
    idx_to_char : [tensor]
        index to character map
    char_to_idx : [tensor]
        character to index map

    Returns
    -------
    [tensor]
        the result of this predict, in character mode
    """
    # init
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars+len(prefix)-1):
        # get its one-hot code, every outputs will become new inputs
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # input the rnn and get its next character and refresh new state
        (Y, state) = RNN(X, state, params)
        if t < len(prefix)-1:
            # when the prefix is not ending yet
            output.append(char_to_idx[prefix[t+1]])
        else:
            # when prefix was end, append the best prediction
            output.append(int(Y[0].argmax(dim=1).item()))
    # result
    return ''.join([idx_to_char[i] for i in output])


def grad_clipping(params, theta, device):
    """
    Clip gradients in case the gradient become too large

    Parameters
    ----------
    params : [tensor]
        the gradient tensor
    theta : [float]
        use to be a threshold of gradients division
    device : [device]
        device
    """
    norm = torch.tensor([0.0], device=device)
    # compute the norm of gradients
    for p in params:
        norm += (p.grad.data**2).sum()
    norm = norm.sqrt().item()
    # min function
    if norm > theta:
        # if less than one, just divide them
        for p in params:
            p.grad.data *= (theta/norm)


class RNNModel(nn.Module):
    """
    RNN model, return (num_steps * batch_size, vocab_size)

    Parameters
    ----------
    nn : [torch.nn]
        the network module
    """
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * \
            (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state):
        # pretreat inputs
        X = to_onehot(inputs, self.vocab_size)
        # input X to rnnlayer to get mid-result, rnn here is the core
        Y, self.state = self.rnn(torch.stack(X), state)
        # using view and fc to make Y be (num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state


def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device,
                        idx_to_char, char_to_idx):
    """
    Predic characters using rnn module

    Parameters
    ----------
    prefixes : [string]
        the start characters of these character sequences
    num_chars : [int]
        the number of characters you want to predict
    model : [type]
        [description]
    vocab_size : [int]
        the number of non-repeating character in this vocab
    device : [device]
        device you want to run on
    idx_to_char : [tensor]
        index to character map
    char_to_idx : [tensor]
        character to index map

    Returns
    -------
    [string]
        the predicted character array
    """
    state = None
    # first output is the prefix
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars+len(prefix)-1):
        # the last output character will be new input
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)

        # predict
        (Y, state) = model(X, state)
        if t < len(prefix)-1:
            # prefix is not end yet
            output.append(char_to_idx[prefix[t+1]])
        else:
            # new character mode
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


def resnet18(num_classes):
    """
    ResNet-18 class. Just input the number of classes want to predict then will
    gain the network want to produce.

    Parameters
    ----------
    num_classes : [int]
        the amount of classes want to predict

    Returns
    -------
    [function]
        the customize network
    """
    net = nn.Sequential(
        # small convolution layer to extract features
        nn.Conv2d(3, 64, kernel_size=3, stride=1),
        # normalize
        nn.BatchNorm2d(64),
        # activation layer
        nn.ReLU()
    )

    # make a residual block factory here
    def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
        # first residual block's input channel should have the same size as the 
        # output channel
        if first_block:
            assert in_channels == out_channels
        blk = []
        # using this loop to determine the amount of residual blocks
        for i in range(num_residuals):
            # for different amount of residual blocks should use different 
            # preventive strategies
            if i == 0 and not first_block:
                blk.append(layers.Residual(in_channels, out_channels,
                                           use_1x1_conv=True, stride=2))
            else:
                blk.append(layers.Residual(out_channels, out_channels))
        # return a sequential block contain a list of residual blocks
        return nn.Sequential(*blk)

    # use these residual blocks to constract the target network
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    # global average pool layer will make multidimension to be two dimension
    net.add_module("global_avg_pool", layers.GlobalAvgPool2d())
    # this part often be called 'dense layer'
    net.add_module("fc", nn.Sequential(
        # flatten layer map 2d to 1d
        layers.FlattenLayer(),
        # linear layer map the long arguments list to result list
        nn.Linear(512, num_classes)
    ))
    return net
