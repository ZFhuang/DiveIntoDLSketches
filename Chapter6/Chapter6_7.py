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
这一节是要实现门控循环单元GRU，为了解决循环神经网络RNN中的梯度衰减和爆炸问题
组合成的网络称为门控循环神经网络GRNN
"""

# 加上这个限定才能支持多线程读取
if __name__ == "__main__":
    # 设置计算设备，让计算在GPU上进行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 首先是从零开始实现门控循环网络GRNN
    # 读取周杰伦歌词数据集
    (corpus_indices, char_to_idx, idx_to_char,
     vocab_size) = data_process.load_data_jay_lyrics()

    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
    print('will use',device)

    # 和之前类似，写一个工厂函数来生成网络用于赋值的参数
    def get_params():
        # 子函数返回一个和输入尺寸相同的正态分布的随机矩阵
        def _one(shape):
            ts = torch.tensor(np.random.normal(
                0, 0.01, size=shape), device=device, dtype=torch.float32)
            return torch.nn.Parameter(ts, requires_grad=True)

        # 用于初始化与层的尺寸有关的随机张量，前两个是对应的层，最后一个是全零的偏移
        # 这个函数的结构要结合书中的GRU图来进行分析
        def _three():
            return(_one((num_inputs, num_hiddens)), _one(
                (num_hiddens, num_hiddens)), torch.nn.Parameter(
                    torch.zeros(num_hiddens, device=device,
                                dtype=torch.float32), requires_grad=True))

        # 更新门部分的参数
        W_xz, W_hz, b_z = _three()
        # 重置门部分的参数
        W_xr, W_hr, b_r = _three()
        # 候选隐藏状态的参数
        W_xh, W_hh, b_h = _three()

        # 在这里单独初始化输出层的权重和偏移值
        W_hq = _one((num_hiddens, num_outputs))
        b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device,
                                             dtype=torch.float32),
                                 requires_grad=True)
        # 最后将生成的参数返回
        return nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh,
                                 b_h, W_hq, b_q])

    # 初始化门控循环单元GRU的隐藏状态的函数
    def init_gru_state(batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device=device),)

    # 门控循环单元的主部分函数
    def gru(inputs, state, params):
        # 初始化单元的一系列参数
        W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
        # 初始化当前状态H，核心计算是利用更新门和重置门来优化状态和梯度的更新
        H, = state
        outputs = []
        # 输入一个就输出一个
        for X in inputs:
            # 如果重置门0更新层1，就是传统的RNN的结构
            # 完全不抛弃当前状态并更新替换为下一个状态
            # 更新门层，负责控制当前层与过去层的混合比例，利用它掌握一些久远的知识
            Z = torch.sigmoid(torch.matmul(X, W_xz)+torch.matmul(H, W_hz)+b_z)
            # 重置门层，负责控制是否要抛弃当前层生成的状态，用于遗忘一些知识
            R = torch.sigmoid(torch.matmul(X, W_xr)+torch.matmul(H, W_hr)+b_r)
            # 候选层,是用重置门层的输出组合的，是新产生的状态参数，用正切函数来激活
            H_tilda = torch.tanh(torch.matmul(X, W_xh) +
                                 R * torch.matmul(H, W_hh) + b_h)
            # 组合权重,更新门决定是要用怎样的权重组合上一步的状态和当前的候选状态
            H = Z*H+(1-Z)*H_tilda
            # 将新的权重应用得到结果
            Y = torch.matmul(H, W_hq)+b_q
            # 附加到结果上
            outputs.append(Y)
        return outputs, (H,)

    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
    # 调用训练与生成函数，选用相邻采样版本，测试效果，loss越接近1越好
    train.train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens,
                                vocab_size, device, corpus_indices, idx_to_char,
                                char_to_idx, False, num_epochs, num_steps, lr,
                                clipping_theta, batch_size, pred_period, pred_len,
                                prefixes)
    print('————————————————————————————')

    # 简洁实现非常轻松，直接使用nn模块自带的GRU即可
    lr = 1e-2
    gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
    model = rnn.RNNModel(gru_layer, vocab_size).to(device)
    train.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                        corpus_indices, idx_to_char, char_to_idx,
                                        num_epochs, num_steps, lr,
                                        clipping_theta, batch_size, pred_period,
                                        pred_len, prefixes)
    print('————————————————————————————')
