import random
import torch


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
