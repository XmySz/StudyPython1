"""
GoogLeNet吸收了NiN中网络串联网络的思想，并在此基础上做了很大改进。
Inception块里有4条并行的线路。前3条线路使用窗口大小分别是1*1,3*3，5*5的卷积层来抽取不同空间尺寸下的信息，
其中中间2个线路会对输入先做1*1卷积来减少输入通道数，以降低模型复杂度。第四条线路则使用3*3最大池化层后接1*1卷积层来改变通道数。
4条线路都使用了合适的填充来使输入与输出的高和宽一致。最后我们将每条线路的输出在通道维上连结，并输入接下来的层中去。
"""
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys

sys.path.append("..")
import dzl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Inception(nn.Module):
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出


# GoogLeNet跟VGG一样，在主体卷积部分中使用5个模块（block），每个模块之间使用步幅为2的3*3最大池化层来减小输出高宽

# 第一个模块使用一个64通道的7*7卷积层
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)

# 第二个模块使用两个卷积层：首先是64通道的1*1卷积层，然后是将通道扩大3倍的3*3卷积层
b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)

# 第三个模块串联2个完整的inception块
b3 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),  # 64+128+32+32=256通道
    Inception(256, 128, (128, 192), (32, 96), 64),  # 128+192+96+64=480通道
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)

# 第四模块更加复杂。它串联了5个Inception块
b4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),  # 256+320+128+128=832通道
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# 第五模块
b5 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),  # 384+384+128+128=1024
    dzl.GlobalAvgPool2d(),
)

net = nn.Sequential(
    b1, b2, b3, b4, b5,
    dzl.FlattenLayer(),
    nn.Linear(1024, 10),
)

X = torch.rand(1, 1, 96, 96)
for blk in net.children():
    X = blk(X)
    print(X.shape)

batch_size = 128
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = dzl.load_data_fashion_mnist(batch_size, resize=96)
lr, num_epochs = 0.001, 10
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
dzl.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

"""
    inception块相当于一个有4条线路的子网络。它通过不同窗口形状的卷积层和最大池化层来并行抽取信息，并使用1*1卷积层减少通道数从而降低模型复杂度。
    GoogLeNet将多个设计精细的Inception块和其他层串联起来。其中Inception块的通道数分配之比是在ImageNet数据集上通过大量的实验得来的。
    GoogLeNet和它的后继者们一度是ImageNet上最高效的模型之一：在类似的测试精度下，它们的计算复杂度往往更低。
"""
