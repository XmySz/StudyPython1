import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from torch.autograd import Variable
from tensorboardX import SummaryWriter

"""
    PyTorch就像带GPU的Numpy，与Python一样都属于动态框架。
    tensor张量可以看作是一个多维数组。标量可以看作是0维张量，向量可以看作1维张量，矩阵可以看作是二维张量。
    torch:torch包包含了多维张量的数据机构以及基于其上的多种数学操作。另外，它也提供了多种工具，其中一些可以更有效的对张量和任意类型进行序列化
    torch.nn:所有神经网络的基类
            nn的核心数据结构是Module，它是一个抽象概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络,一个nn.Module实例应该包含一些层以及返回输出的前向传播（forward）方法。
    torch.nn.Module:所有网络的基类
    torch.nn.Sequential(*args):一个时序容器,Modules会以他们传入的顺序被添加到容器中
    torch.nn.functional:存放一些常见的激活函数,正则化函数,损失函数等等.
    torch.autograd:根据输入和前向传播过程自动构建计算图，并执行反向传播.要想使用自动求导,只需要将所有的tensor包含进Variable对象即可
    torch.utils.data:表示Dataset的抽象类
    torch.nn.init:提供初始化参数的功能
    torch.optim:该模块提供了优化算法
    tensorboardX:Tensorboard是Google TensorFlow的可视化工具，它可以记录训练数据、评估数据、网络结构、图像等，并且可以在web上展示，
                对于观察神经网络训练的过程非常有帮助。PyTorch可以采用tensorboard_logger、visdom等可视化工具，但这些方法比较复杂或不够
                友好。为解决这一问题，人们推出了可用于PyTorch可视化的新的更强大的工具——tensorboardX。
    torchvision:包含了目前流行的数据集,模型结构和常用的图片转换工具
        torch.datasets包含了数据集
            MNIST
            COCO
            LSUN
            ImageFolder
            Imagenet-12
            CIFAR10 and CIFAR100
            STL10
        torch.models包含了以下的模型结构
            AlexNet
            VGG
            ResNet
            SqueezeNet
            DenseNet 
    建模过程：
        一.获取数据  torch.utils.data
        二.定义模型
        三.初始化参数  torch.optim
        四.定义损失函数    
        五.定义优化算法
        六.训练模型
        七.测试模型
"""


def create_tensor():
    """
        几种常见的创建张量的方法
    """
    x = torch.empty(5, 3)  # 未初始化的
    x1 = torch.rand(5, 3)  # 0-1均匀分布
    x2 = torch.zeros(5, 3)  # 全是0的
    x10 = torch.ones(5, 3)  # 全是1的
    torch.tensor([1, 1, 2, 3, 4])  # 用自己的数据创建的，根据数据自动判断类型
    torch.Tensor(1)  # 默认类型FloatTensor
    x4 = torch.randn(2, 3)  # 符合标准正态分布的随机生成
    x5 = torch.eye(3)  # 对角线位置全为1，其他全为0
    x6 = torch.eye(3, 4)
    x7 = torch.from_numpy(np.array([1, 2, 3]))  # 从numpy数组中创建tensor张量
    x8 = torch.linspace(-1, 1, 100)  # 从等差数列中创建tensor
    x9 = torch.arange(-1, 1, 0.1)  # 从range中创建
    x11 = torch.normal(0, 0.01, (5, 3))  # 正态分布创建
    x.size() or x.shape()  # 获取张量的形状


def operation_tensor():
    """
        张量的常见操作
    """
    x = torch.rand(5, 3)
    y = torch.ones(5, 3)

    # 加法形式1
    x + y
    # 加法形式2
    torch.add(x, y)
    # 加法形式3（速度更快）
    y.add(x)

    # 改变形状方式1（共享同一段内存）
    y = x.view(15)
    z = x.view(-1, 5)

    # 改变形状方式2（不共享同一段内存）
    y = x.clone().view(15)

    # 改变形状方式3,类似与view
    y = x.resize(3, 5)

    # 改变形状方式4
    y = torch.reshape(x, (3, 5))

    # 指定维度增加一个“1”
    torch.unsqueeze(x, dim=1)

    # 指定维度压缩一个“1”
    torch.squeeze(x, dim=1)

    # 维度换位
    x = torch.rand(3, 2, 3).permute(2, 0, 1)

    # tensor转numpy
    a = x.numpy()

    # numpy转tensor
    a = np.ones(5)
    b = torch.from_numpy(a)

    # tensor转numpy()
    a = torch.randn(5, 3).numpy()  # 张量的required=False时
    b = torch.randn(5, 3, requires_grad=True).detach.numpy()  # 张量的required=True时

    # 张量的矩阵积
    a = torch.randn(3, 4)
    b = torch.randn(4, 3)
    torch.matmul(a, b)
    torch.mm(a, b)

    # 张量的按轴连接   0表示按列,1表示按行
    a = torch.randn(3, 4)
    b = torch.randn(3, 1)
    torch.cat((a, b), dim=1)

    # item函数将一个单元素的张量转化为一个数
    x1 = torch.randn(1)


def create_module():
    """
        列举了几种创建模型的方式
    :return:
    """
    num_inputs = 5
    num_outputs = 1

    class LinearNet(nn.Module):
        def __init__(self, n_feature):
            super(LinearNet, self).__init__()
            self.linear = nn.Linear(n_feature, 1)

        def forward(self, x):
            return self.linear(x)

    net = LinearNet(num_inputs)
    print(net.linear.weight)
    net = nn.Sequential(nn.Linear(num_inputs, num_outputs))
    print(net[0].weight)
    net = nn.Sequential()
    net.add_module('linear', nn.Linear(num_inputs, num_outputs))
    net = nn.Sequential(OrderedDict([('linear', nn.Linear(num_inputs, num_outputs))]))

    net = nn.ModuleList([nn.Linear(num_inputs, 10), nn.ReLU(), nn.Linear(10, num_outputs)])

    net.append(nn.Linear(num_outputs, 1))
    print(net[-1])
    for param in net.parameters():
        print(param)


def process_build_model():
    """
    建模过程：
        一.获取数据  torch.utils.data
        二.定义模型
        三.初始化参数  torch.optim
        四.定义损失函数
        五.定义优化算法
        六.训练模型
        七.测试模型
    """


def RNN_study():
    """
    Pytorch中RNN模块函数为torch.nn.RNN(input_size,hidden_size,num_layers,batch_first)，每个参数的含义如下:
        input_size：输入数据的特征数
        hidden_size：隐含层的神经元个数，这个维数要么参考别人的结构设置，要么自行设置，比如可以设置成20；
        num_layers：隐含层的层数。
        batch_first：当 batch_first设置为True时，输入的参数顺序变为：x：[batch_size, time_step, input_size]
                     h0：[batch_size, num_layers, hidden_size]，输出和输入保持一致
    RNN的输入：(input,h_0)
        input:(seq_len, batch, input_size):,保存输入序列的特征
        h_0:(num_layers, batch_size, hidden_size)，保存着初始隐状态
    RNN的输出：(output, h_n)
        output：(seq_len, batch, hidden_size)，保存着RNN最后一层的输出特征。
        h_n：(num_layers，batch, hidden_size)，保存着最后一个时刻隐状态。

    """
    rnn = nn.RNN(10, 20, 2)
    input = Variable(torch.randn(5, 3, 10))
    h0 = Variable(torch.randn(2, 3, 20))
    output, h_n = rnn(input, h0)
    print(output, '\n', h_n)
    print(output.shape, '\n', h_n.shape)


def tensorboardX():
    """
    可视化工具
    启动命令：到logs同级目录执行tensorboard --logdir=logs --port 6006
    :return:
    """
    writer = SummaryWriter(logdir="logs")
