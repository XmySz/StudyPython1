# ResNet与DenseNet在跨层连接上的主要区别：使用相加和使用连结
# DenseNet的主要构建模块是稠密块（dense block）和过渡层（transition layer）。
# 前者定义了输入和输出是如何连结的，后者则用来控制通道数，使之不过大。
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
import dzl

sys.path.append("..")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 稠密块
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )
    return blk


# 稠密块由多个conv_block组成，每块使用相同的输出通道数。
class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        """
        :param num_convs:# conv_block块的个数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        """
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels  # 计算输出通道数

    def forward(self, X):
        """
        :param X: shape(batch_size,C,H,W)
        :return:
        """
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输入和输出连结
        return X


# blk = DenseBlock(2, 3, 10)
# X = torch.rand(4, 3, 8, 8, )  # 包含了从区间[0, 1)的均匀分布中抽取的一组随机数
# Y = blk(X)

# 过渡层
"""
由于每个稠密块都会带来通道数的增加，使用过多则会带来过于复杂的模型。过渡层用来控制模型复杂度。
它通过1*1卷积层来减小通道数，并使用核大小和步幅为2的平均池化层减半高和宽，从而进一步降低模型复杂度。
"""
def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2),
    )
    return blk


# blk = transition_block(23, 10)
# print(blk(Y).shape)  # torch.Size([4, 10, 4, 4])


# DenseNet模型
# DenseNet首先使用同ResNet一样的单卷积层和最大池化层。
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2, padding=1),
)

# 类似于ResNet接下来使用的4个残差块，DenseNet使用的是4个稠密块。
# 同ResNet一样，我们可以设置每个稠密块使用多少个卷积层。
# 这里我们设成4，从而与上一节的ResNet-18保持一致。
# 稠密块里的卷积层通道数（即增长率）设为32，所以每个稠密块将增加128个通道。
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [3, 4, 6, 3]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    DB = DenseBlock(num_convs, num_channels, growth_rate)
    net.add_module("DenseBlosk_%d" % i, DB)
    # 上一个稠密块的输出通道数
    num_channels = DB.out_channels
    # 在稠密块之间加入通道数减半的过渡层
    if i != len(num_convs_in_dense_blocks) - 1:
        net.add_module("transition_block_%d" % i, transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

# 同ResNet一样，最后接上全局池化层和全连接层来输出。
net.add_module("BN", nn.BatchNorm2d(num_channels))
net.add_module("relu", nn.ReLU())
net.add_module("global_avg_pool", dzl.GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, num_ch   annels, 1, 1)
net.add_module("fc", nn.Sequential(dzl.FlattenLayer(), nn.Linear(num_channels, 10)))

# 测试模型是否存在问题
# X = torch.rand((1, 1, 96, 96))
# for name, layer in net.named_children():
#     X = layer(X)
#     print(name, ' output shape:\t', X.shape)

# 获取数据训练模型
batch_size = 256
num_epochs, lr = 15, 1e-3
optimizer = optim.Adam(net.parameters(), lr=lr)
train_iter, test_iter = dzl.load_data_fashion_mnist(batch_size=batch_size, resize=96)
dzl.train_ch5(net, train_iter, test_iter, batch_size, optimizer=optimizer, device=device, num_epochs=num_epochs)