# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import data_process, train, rnn

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

"""
这一节实现长短期记忆LSTM网络，是GRU的前身，对大数据表现比GRU好，其他情况下效果类似
"""

# 加上这个限定才能支持多线程读取
if __name__ == "__main__":
    # 设置计算设备，让计算在GPU上进行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 首先是从零开始实现长短期记忆LSTM网络
    # 和上一节一样读取数据和初始化参数
    (corpus_indices, char_to_idx, idx_to_char,
     vocab_size) = data_process.load_data_jay_lyrics()

    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
    print('will use', device)

    # 初始化参数
    def get_params():
        def _one(shape):
            ts = torch.tensor(np.random.normal(
                0, 0.01, size=shape), device=device, dtype=torch.float32)
            return torch.nn.Parameter(ts, requires_grad=True)

        def _three():
            return(_one((num_inputs, num_hiddens)), _one(
                (num_hiddens, num_hiddens)), torch.nn.Parameter(
                    torch.zeros(num_hiddens, device=device,
                                dtype=torch.float32), requires_grad=True))

        # 输入门的参数
        W_xi, W_hi, b_i = _three()
        # 遗忘门的参数
        W_xf, W_hf, b_f = _three()
        # 输出门的参数
        W_xo, W_ho, b_o = _three()
        # 候选记忆细胞的参数
        W_xc, W_hc, b_c = _three()

        # 输出层的参数单独设置,分别是权重和偏移
        W_hq = _one((num_hiddens, num_outputs))
        b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device,
                                             dtype=torch.float32),
                                 requires_grad=True)
        # 最后将生成的参数返回
        return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho,
                                 b_o, W_xc, W_hc, b_c, W_hq, b_q])

    def init_lstm_state(batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device=device),
                torch.zeros((batch_size, num_hiddens), device=device))

    def lstm(inputs, state, params):
        # 初始化参数
        [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho,
            b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
        # 这次作为状态会被传递到下一时间步的值多了记忆细胞部分
        (H, C) = state
        outputs = []
        # 输入一个就输出一个
        for X in inputs:
            # 这里按照书中公式相类似地设置各个层的计算法则即可
            # 输入层
            I = torch.sigmoid(torch.matmul(X, W_xi)+torch.matmul(H, W_hi)+b_i)
            # 遗忘层
            F = torch.sigmoid(torch.matmul(X, W_xf)+torch.matmul(H, W_hf)+b_f)
            # 输出层
            O = torch.sigmoid(torch.matmul(X, W_xo)+torch.matmul(H, W_ho)+b_o)
            # 候选记忆细胞
            C_tilda = torch.tanh(torch.matmul(
                X, W_xc) + torch.matmul(H, W_hc) + b_c)
            # 组合权重,更新门决定是要用怎样的权重组合上一步的状态和当前的候选状态
            C = F*C+I*C_tilda
            # 应用输出参数
            H = O*C.tanh()
            # 应用得到结果
            Y = torch.matmul(H, W_hq)+b_q
            # 附加到结果上
            outputs.append(Y)
        return outputs, (H, C)

    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ['Creeper', '苦力怕?']
    # 调用训练与生成函数，选用相邻采样版本，测试效果，loss越接近1越好
    # train.train_and_predict_rnn(lstm, get_params, init_lstm_state, num_hiddens,
    #                             vocab_size, device, corpus_indices, idx_to_char,
    #                             char_to_idx, False, num_epochs, num_steps, lr,
    #                             clipping_theta, batch_size, pred_period, pred_len,
    #                             prefixes)
    print('————————————————————————————')

    # 简洁实现只要调用nn的库即可，效率也高一些
    lr = 1e-2
    lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
    model = rnn.RNNModel(lstm_layer, vocab_size).to(device)
    train.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                        corpus_indices, idx_to_char, char_to_idx,
                                        num_epochs, num_steps, lr,
                                        clipping_theta, batch_size, pred_period,
                                        pred_len, prefixes)
