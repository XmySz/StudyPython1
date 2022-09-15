import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys

sys.path.append("..")
import dzl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('jaychou_lyrics.txt', 'rb') as f:
    corpus_chars = f.read().decode('utf-8')

# 方便起见把所有的换行符和回车符换成空白符
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')

# 映射字符到数字
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(idx_to_char)  # vocab---词汇
corpus_indices = [char_to_idx[char] for char in corpus_chars]  # 字符索引表

# 构造模型(单隐层,隐层单元个数为256)
num_hiddens = 256
# rnn_layer的输入形状为（时间步数，批量大小，输入个数），其中输入个数为词典的大小。
# 返回隐藏状态h和输出，而输出的形状为形状为(时间步数, 批量大小, 隐藏单元个数)，隐藏状态h的形状为(层数, 批量大小, 隐藏单元个数)
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)


# 构建模型
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)  # 全连接层
        self.state = None

    def forward(self, inputs, state):  # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        X = dzl.to_onehot(inputs, self.vocab_size)  # X是个list
        Y, self.state = self.rnn(torch.stack(X), state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

model = RNNModel(rnn_layer, vocab_size)
dzl.predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx)