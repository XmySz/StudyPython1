"""
ResNet沿用了VGG全3*3卷积层的设计，残差块里首先有2个有相同输出通道数的3*3卷积层。每个卷积层后接一个批量归一化层和ReLU激活函数。
然后我们将输入跳过这两个卷积运算后直接加在最后的ReLU激活函数前。这样的设计要求两个卷积层的输出与输入形状一样，从而可以相加。
如果想改变通道数，就需要引入一个额外的1*1卷积层来将输入变换成需要的形状后再做相加运算。
这里每个模块有4个卷积层，加上最开始的卷积层和最后的全连接层，共计18层，这个模型也被称为ResNet-18
通过配置不同的通道数和模块里的残差块数可以得到不同的ResNet模型，例如更深的含152层的ResNet-152。
"""
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys

sys.path.append("..")
import dzl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnetNum = [3, 4, 6, 3]
resnet18 = [2, 2, 2, 2]
resnet34 = [3, 4, 6, 3]


# 残差块的实现，它可以设定输出通道数、是否使用额外的1*1卷积层来修改通道数以及卷积层的步幅。
class BaseResidual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(BaseResidual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


# 查看输入和输出形状一致的情况
blk = BaseResidual(3, 3)
X = torch.rand((4, 3, 6, 6))
print(blk(X).shape)  # torch.Size([4, 3, 6, 6])

# 也可以在增加输出通道数的同时减半输出的高和宽。
blk = BaseResidual(3, 6, use_1x1conv=True, stride=2)
print(blk(X).shape)  # torch.Size([4, 6, 3, 3])


# ResNet则使用4个由残差块组成的模块，每个模块使用若干个同样输出通道数的残差块。
# 第一个模块的通道数同输入通道数一致。由于之前已经使用了步幅为2的最大池化层，所以无须减小高和宽。
# 之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半。
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels  # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(BaseResidual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(BaseResidual(out_channels, out_channels))
    return nn.Sequential(*blk)


# ResNet的前两层,在输出通道数为64、步幅为2的卷积层后接步幅为2的最大池化层。不同之处在于ResNet每个卷积层后增加的批量归一化层。
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 接着我们为ResNet加入所有残差块。这里每个模块使用两个残差块。以下为34层以下使用
net.add_module("resnet-block1", resnet_block(64, 64, resnetNum[0], first_block=True))
net.add_module("resnet-block2", resnet_block(64, 128, resnetNum[1]))
net.add_module("resnet-block3", resnet_block(128, 256, resnetNum[2]))
net.add_module("resnet-block4", resnet_block(256, 512, resnetNum[3]))


# 最后，与GoogLeNet一样，加入全局平均池化层后接上全连接层输出。
net.add_module("global_avg_pool", dzl.GlobalAvgPool2d())  # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
net.add_module("fc", nn.Sequential(dzl.FlattenLayer(), nn.Linear(512, 10)))

# X = torch.rand((1, 1, 224, 224))
# for name, layer in net.named_children():
#     X = layer(X)
#     print(name, ' output shape:\t', X.shape)


# 下面我们在Fashion-MNIST数据集上训练ResNet。
batch_size = 256
train_iter, test_iter = dzl.load_data_fashion_mnist(batch_size, resize=96)
lr, num_epochs = 0.001, 10
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
dzl.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


