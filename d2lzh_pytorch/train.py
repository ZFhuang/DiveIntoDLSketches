from d2lzh_pytorch import linear_reg, data_process, rnn
import time
import torch
import torch.nn as nn
import math


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


def train_ch5(net, train_iter, test_iter, batch_size, optimizer,
              device, num_epochs):
    """
    The new version of training function, it contain the device selection and
    became a fully module function, better than train_ch3.

    Parameters
    ----------
    net : [module]
        the net that want to apply
    train_iter : [producer]
        the train dataset
    test_iter : [producer]
        the test dataset
    batch_size : [int]
        size of the batch
    optimizer : [function]
        the function to decrease the gradient
    device : [device]
        the device want to run on
    num_epochs : [int]
        how many times do you want to learn through the whole training dataset?
    """
    # apply the net on target device
    net = net.to(device)
    print('training on:', device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for e in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            # apply datas to target device
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # for some elements not on GPU should move to CPU for calculation
            train_l_sum += l.cpu().item()
            train_acc_sum += ((y_hat.argmax(dim=1) == y).sum().cpu().item())
            n += y.shape[0]
            batch_count += 1
        # use those value stored in iter to gain the final value of every epochs
        test_acc = data_process.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.3f, train acc %.3f, test acc %.3f,time %.1f sec' %
              (e+1, train_l_sum/batch_count, train_acc_sum/n, test_acc, time.time()-start))


def train_and_predict_rnn(RNN, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    """
    Train and predict a sequence with a function building step by step called RNN

    Parameters
    ----------
    RNN : [function]
        the recycle neural network
    get_params : [function]
        the function that can return network's init params
    init_rnn_state : [tuple]
        network's start time state
    num_hiddens : [int]
        how many hidden parameters you want to add
    vocab_size : [int]
        the number of non-repeating character in this vocab
    device : [device]
        device you want to run on
    corpus_indices : [tensor]
        the index formation of input data
    idx_to_char : [tensor]
        index to character map
    char_to_idx : [tet]
        how many epoch would print a prediction
    pred_len : [int]
        the number of characters you want to predict
    prefixes : [string]
        the start characters of these character sequencesnsor]
        character to index map
    is_random_iter : [bool]
        choose the iter's type
    num_epochs : [int]
        number of epochs
    num_steps : [int]
        number of time steps
    lr : [float]
        learning rate
    clipping_theta : [float]
        use to be a threshold of gradients division
    batch_size : [int]
        how many times a batch would contain
    pred_period : [int]
        how many epoch would print a prediction
    pred_len : [int]
        the number of characters you want to predict
    prefixes : [string]
        the start characters of these character sequences
    """
    # choose which function to load data  
    if is_random_iter:
        data_iter_fn = data_process.data_iter_random
    else:
        data_iter_fn = data_process.data_iter_consecutive
    # get init params of network
    params = get_params()
    # use CrossEntropyLoss's exponent to be the perplexity
    loss = nn.CrossEntropyLoss()

    # repeat epoch times, to ensure a better result
    for e in range(num_epochs):
        # if it is not using random iter, init at the start of an epoch
        if not is_random_iter:
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)

        # load a batch of pair of data at a time from data_iter
        # this loop will loading data by time-step, one loop one step
        for X, Y in data_iter:
            if is_random_iter:
                # random_iter should re-init in every step calls
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
                # else we just need to detach it's state, in case the gradient
                # compution cost too much time
                for s in state:
                    s.detach_()

            # pretreat our datas, get (batch_size ,vocab_size)
            inputs = rnn.to_onehot(X, vocab_size)
            # put it into RNN and get its output and new state
            # outputs will has num_steps (batch_size, vocal_size) matrixes
            # here you can see that inputs is totally new, state may be old
            outputs, state = RNN(inputs, state, params)
            # cat outputs to be (num_steps*batch_size, vocal_size)
            outputs = torch.cat(outputs, dim=0)
            # make y be (batch*num_steps), y is the gt answer
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            # compute the loss
            l = loss(outputs, y.long())

            # set gradient to be zero
            if params[0].grad is not None:
                for p in params:
                    p.grad.data.zero_()
            # then backpropagate the gradient
            l.backward()
            # clip gradient
            rnn.grad_clipping(params, clipping_theta, device)
            # decent it
            linear_reg.sgd(params, lr, 1)
            # cal the whole loss
            l_sum += l.item()*y.shape[0]
            n += y.shape[0]

        # print some result
        if (e+1) % pred_period == 0:
            # use exp to cal perplexity here
            print('epoch %d, perplexity %f, time %.2f sec' %
                  (e + 1, math.exp(l_sum / n), time.time() - start))
            # print some prediction here
            for prefix in prefixes:
                print(' -', rnn.predict_rnn(prefix, pred_len, RNN, params,
                                            init_rnn_state, num_hiddens,
                                            vocab_size, device, idx_to_char, 
                                            char_to_idx))


def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, 
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes):
    """
    Train the net which is constructed by pytorch module and predict strings

    Parameters
    ----------
    model : [function]
        the recycle neural network
    num_hiddens : [function]
        the recycle neural network
    vocab_size : [int]
        the number of non-repeating character in this vocab
    device : [device]
        device you want to run on
    corpus_indices : [tensor]
        the index formation of input data
    idx_to_char : [tensor]
        index to character map
    char_to_idx : [tet]
        how many epoch would print a prediction
    num_epochs : [int]
        number of epochs
    num_steps : [int]
        number of time steps
    lr : [float]
        learning rate
    clipping_theta : [float]
        use to be a threshold of gradients division
    batch_size : [int]
        how many times a batch would contain
    pred_period : [int]
        how many epoch would print a prediction
    pred_len : [int]
        the number of characters you want to predict
    prefixes : [string]
        the start characters of these character sequences
    """
    # init
    loss = nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    model.to(device)
    state=None

    # repeat epoch times, to ensure a better result
    for e in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        # here only use the consecutive version
        data_iter = data_process.data_iter_consecutive(
            corpus_indices, batch_size, num_steps, device)

        # load a batch of pair of data at a time from data_iter
        # this loop will loading data by time-step, one loop one step
        # X is now's data and Y is its answer
        for X, Y in data_iter:
            if state is not None:
                if isinstance(state, tuple):
                    # for LSTM
                    state = (state[0].detach(), state[1].detach())
                else:
                    # detach state from graph to prevent the highly cost
                    state = state.detach()

            # predict and get outputs
            # outputs will has num_steps (batch_size, vocal_size) matrixes
            # here you can see that inputs is totally new, state may be old
            outputs, state = model(X, state)
            # make y be (batch*num_steps), y is the ground truth
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            # compute the loss
            l = loss(outputs, y.long())

            # set gradient to be zero
            optimizer.zero_grad()
            # then backpropagate the gradient
            l.backward()
            # clip gradient
            rnn.grad_clipping(model.parameters(), clipping_theta, device)
            # decent it
            optimizer.step()
            # cal the whole loss
            l_sum += l.item()*y.shape[0]
            n += y.shape[0]

        # use exp to cal perplexity
        try:
            perplexity=math.exp(l_sum/n)
        except OverflowError:
            perplexity=float('inf')

        # print some result
        if (e+1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' %
                  (e + 1, perplexity, time.time() - start))
            # print some prediction here
            for prefix in prefixes:
                print(' -', rnn.predict_rnn_pytorch(prefix, pred_len, model,
                                                    vocab_size, device,
                                                    idx_to_char, char_to_idx))


def train_2d(trainer):
    """
    Train gradient function pretreat from target function and return the decent 
    trace.

    Parameters
    ----------
    trainer : [function]
        the gradient function, input x1, x2 and s1, s2, return function value

    Returns
    -------
    [tensor]
        the trace of target function's decention's inputs
    """
    # s1, s2 here are states
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(20):
        # get new value
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results
