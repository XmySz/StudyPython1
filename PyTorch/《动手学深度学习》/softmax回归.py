import torch
import dzl
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data
import time
import sys
import os
from collections import OrderedDict
from torch import nn
from torch.nn import init


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

num_inputs = 784
num_outputs = 10

# 导入数据  十个类别，每个类别训练集6000，测试集1000  图片大小28×28
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                               transform=transforms.ToTensor())
# 读取数据
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):  # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


net = LinearNet(num_inputs, num_outputs)

# 初始化权重
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# 定义交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.3)

# 训练模型
num_epochs = 10
dzl.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)


def get_fashion_mnist_labels(labels):
    """
        将对应标签转换为相应的文本
    :param labels:
    :return:
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


