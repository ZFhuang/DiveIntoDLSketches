# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import data_process, train, rnn

import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

"""
这一节从零开始实现了一个循环神经网络RNN并将其初步用来进行歌词的字符预测
"""

# 加上这个限定才能支持多线程读取
if __name__ == "__main__":
    # 设置计算设备，让计算在GPU上进行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 读取数据集
    (corpus_indices, char_to_idx, idx_to_char,
     vocab_size) = data_process.load_data_jay_lyrics()

    # 这部分已经存入并组合出上层函数to_onehot
    # # 为了将词输入到神经网络中，用one-hot向量(独热编码)来表示，这里实现一个函数
    # # 独热编码之前也遇到过了，就是一个很长的向量，只有对应的整数的输出位是1
    # def one_hot(x, n_class, dtype=torch.float32):
    #     x = x.long()
    #     # 用一个数组来表示这个向量，这里先预分配空间(batch, n_class)
    #     res = torch.zeros((x.shape[0], n_class, dtype=dtype, device=x.device)
    #     # 用scatter重整矩阵,从index和对应的src中拿值(这里值只有一个1)
    #     # out[index[i, j], j] = value[i, j] dim=0
    #     # out[i,index[i, j]] = value[i, j]] dim=1
    #     # 其实这里就是按照输入的X作为索引，给对应行的向量的索引位置填上1
    #     # 因此这里是批量完成独热编码
    #     res.scatter_(1, x.view(-1, 1), 1)
    #     return res

    # # 给第一行的0位赋1，第二行的2位赋1
    # x = torch.tensor([0, 2])
    # # tensor([[1., 0., 0.,  ..., 0., 0., 0.],
    # #         [0., 0., 1.,  ..., 0., 0., 0.]])
    # print(one_hot(x, vocab_size))

    X = torch.arange(10).view(2, 5)
    print(X)
    # 这个函数将一组batch转为独热的list，例如这里是2*5的输入，也就是5组数据，每组两个
    # 函数将其转为5个list，每个list两个独热编码的一对数据
    inputs = rnn.to_onehot(X, vocab_size)
    # 因此这里得到的list是5个pair
    print(len(inputs), inputs[0].shape)
    print(inputs[0][0])  # 0的独热
    print(inputs[0][1])  # 5的独热
    print('————————————————————————————')

    # 然后要初始化模型的参数,输入是独热，输出是预测因此也是独热,隐藏层的数量是超参数
    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
    print('will use', device)

    # 初始化隐藏状态的变量，公式为 Out=激活(X*W_xh+(H_t-1)*W_hh+b_h)*W_hq+b_q
    def get_params():
        # 这个子函数用来生成一个有梯度的正态分布随机参数矩阵作为赋值
        def _one(shape):
            ts = torch.tensor(np.random.normal(
                0, 0.01, size=shape), device=device, dtype=torch.float32)
            return torch.nn.Parameter(ts, requires_grad=True)

        # 当前输入数据的权重
        W_xh = _one((num_inputs, num_hiddens))
        # 上一时间步的参数的权重
        W_hh = _one((num_hiddens, num_hiddens))
        # 当前输入的偏置
        b_h = torch.nn.Parameter(torch.zeros(
            num_hiddens, device=device, requires_grad=True))
        # 当前时间步得到的参数的权重
        W_hq = _one((num_hiddens, num_outputs))
        # 当前时间步的偏置
        b_q = torch.nn.Parameter(torch.zeros(
            num_outputs, device=device, requires_grad=True))
        # 返回的初始化后的参数组合在一个参数列表里方便下面网络来调用
        return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])

    # 接着定义这个循环神经网络的模型
    # 首先定义生成全零元组(结尾加个逗号，表示不能再更改了)提供给隐藏状态初始化的函数
    def init_rnn_state(batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device=device), )

    # 然后下面的rnn函数定义了如何在一个时间步内计算和输出，这个rnn非常浅，只有两层
    def RNN(inputs, state, params):
        # 解包输入的参数
        W_xh, W_hh, b_h, W_hq, b_q = params
        H,  = state
        # 初始化结果向量
        outputs = []
        # 这个循环是循环神经网络的核心循环，由于输入的关系，这里取到的X以时间步为单位的
        # 然后每个时间步内是batch，也就是很多组数据，类似之前的训练，一同求batch的loss
        # 然后再更新一次，每个batch的元素都是独热编码，可以作为gt输出去匹配计算
        for X in inputs:
            # 这两行在实现上面的带有隐藏状态的预测计算公式
            # 矩阵相乘可以用来批量batch地完成参数计算的过程
            # tanh是形状类似sigmoid的函数，但是这个函数0的时候结果是0，基于原点对称
            # 显然新数据和旧数据的权重是不相同的
            H = torch.tanh(torch.matmul(X, W_xh)+torch.matmul(H, W_hh)+b_h)
            Y = torch.matmul(H, W_hq)+b_q
            # 将结果不断叠加到输出向量中
            outputs.append(Y)
        # 和新的时间步状态一并返回
        return outputs, (H, )

    # 这里测试一下，初始化最初的状态，X是前面初始化的部分
    # X = torch.arange(10).view(2, 5)
    state = init_rnn_state(X.shape[0], num_hiddens, device)
    # 将X转为独热矩阵
    inputs = rnn.to_onehot(X.to(device), vocab_size)
    print(inputs[0].shape)
    print(inputs[1].shape)
    # 初始化参数
    params = get_params()
    # 输入到rnn中得到输出的数据和新的状态,RNN利用当前值和之前状态一起作为输入
    # 返回预测值和当前状态
    outputs, state_new = RNN(inputs, state, params)
    print(len(outputs), outputs[0].shape, state_new[0].shape)

    # 这里定义了预测函数predict_rnn，根据前缀来预测接下来的num_chars个字符
    # 试着随机预测一段序列
    print(rnn.predict_rnn('分开', 10, RNN, params, init_rnn_state,
                          num_hiddens, vocab_size, device, idx_to_char, char_to_idx))
    print('————————————————————————————')

    # 由于循环神经网络容易梯度爆炸，因此需要进行梯度裁剪，也就是取阈值/范数或1
    # 这里写了函数grad_clipping来用
    # 然后为了测试下训练模型和生成，这里写了函数train_and_predict_rnn
    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    # 调用训练与生成函数，选用随机采样版本，测试效果，loss越接近1越好
    train.train_and_predict_rnn(RNN, get_params, init_rnn_state, num_hiddens,
                                vocab_size, device, corpus_indices, idx_to_char,
                                char_to_idx, True, num_epochs, num_steps, lr,
                                clipping_theta, batch_size, pred_period, pred_len,
                                prefixes)
    print('————————————————————————————')

    # 调用训练与生成函数，选用相邻采样版本，测试效果，loss越接近1越好
    train.train_and_predict_rnn(RNN, get_params, init_rnn_state, num_hiddens,
                                vocab_size, device, corpus_indices, idx_to_char,
                                char_to_idx, False, num_epochs, num_steps, lr,
                                clipping_theta, batch_size, pred_period, pred_len,
                                prefixes)
    print('————————————————————————————')
