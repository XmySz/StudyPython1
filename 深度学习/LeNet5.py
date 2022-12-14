# Fashion-MNIST数据集
# 包括一个训练集60000个例子和测试集10000个例子。每个示例都是一个28x28灰度图像，与10个类中的一个标签相关联。
import time
import torch
from torch import nn, optim
import sys

sys.path.append("..")
import dzl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


net = LeNet()
print("网络结构：", net)

batch_size = 256
train_iter, test_iter = dzl.load_data_fashion_mnist(batch_size=batch_size)

lr, num_epochs = 0.001, 60
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
dzl.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
