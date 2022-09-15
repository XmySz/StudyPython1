import torch
import sys
from torch import nn
from torch.nn import init

sys.path.append("..")
import dzl

epochs = 30
batch_size = 256
train_iter, test_iter = dzl.load_data_fashion_mnist(batch_size)

# 定义模型参数并初始化
num_inputs, num_outputs, num_hidden1, num_hidden2 = 784, 10, 256, 256

net = nn.Sequential(
    dzl.FlattenLayer(),
    nn.Linear(num_inputs, num_hidden1),
    nn.ReLU(),
    nn.Linear(num_hidden1, num_hidden2),
    nn.ReLU(),
    nn.Linear(num_hidden2, num_outputs)
)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.3)

for param in net.parameters():
    init.normal_(param, mean=0, std=0.01)

dzl.train_ch3(net, train_iter, test_iter, loss, epochs, batch_size, None, None, optimizer)
