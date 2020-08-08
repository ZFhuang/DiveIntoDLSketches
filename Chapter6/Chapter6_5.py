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
这一节用torch库来简洁实现循环神经网络
"""

# 加上这个限定才能支持多线程读取
if __name__ == "__main__":
    # 设置计算设备，让计算在GPU上进行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 读取数据集
    (corpus_indices, char_to_idx, idx_to_char,
     vocab_size) = data_process.load_data_jay_lyrics()

    # 隐藏单元数目，相当于输出时的尺寸大小
    num_hiddens = 256
    # 不同的层有着相似的运行效果，可以当作参数给网络实例提供
    # torch包直接帮我们提供了RNN层的实例，输入个数就是指语料数vocab_size
    rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
    # 也可以用后面会提到的长短期记忆网络层
    # rnn_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)

    # 测试一下RNN层的实例，作为实例其特性就是会返回各个时间步上计算得到的隐藏状态和输出
    # 注意这里由于只是一层，因此这里的输出并不涉及最终输出的计算
    num_steps = 35
    batch_size = 2
    state = None
    # X: (时间步，批，输入个数)
    X = torch.rand(num_steps, batch_size, vocab_size)
    print(X.shape)
    # Y: (时间步，批大小，隐藏单元数目*(方向))，也就是格式和输入相同，方向后面会将
    Y, state_new = rnn_layer(X, state)
    print(Y.shape)
    # state: (网络的层数*(方向)，批大小，隐藏单元数目)
    print(len(state_new), state_new[0].shape)
    print('————————————————————————————')

    # 写好模型类RNNModel后，再写个模型预测的函数predict_rnn_pytorch
    # 然后用随机权重来预测一下
    model = rnn.RNNModel(rnn_layer, vocab_size).to(device)
    print(rnn.predict_rnn_pytorch('分开', 10, model,
                                  vocab_size, device, idx_to_char, char_to_idx))
    print('————————————————————————————')

    # 和6.4相似的训练，但是超参数例如学习率有很大不同，这是之前提到的loss计算区别导致
    num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    # 调用训练与生成函数，选用随机采样版本，测试效果，loss越接近1越好
    train.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                        corpus_indices, idx_to_char, char_to_idx,
                                        num_epochs, num_steps, lr,
                                        clipping_theta, batch_size, pred_period,
                                        pred_len, prefixes)
