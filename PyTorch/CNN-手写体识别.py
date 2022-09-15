import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.manual_seed(1)

EPOCH = 1   # 训练次数
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root = './mnist/',
    train=True,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST,

)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(    # 卷积层1
            nn.Conv2d(  # (1, 28, 28)
                in_channels=1,  # 输入图像的通道数,黑白图像为1,彩色图像为3
                out_channels=16,    # 卷积后产生的通道数(卷积核数量*通道数)
                kernel_size=5,  # 卷积核大小
                stride=1,       # 卷积核的步幅
                padding=2,      # 在外面的填充0的层数  padding = (kernel_size-1)/2
            ),  # (16, 28, 28)
            nn.ReLU(),  # (16, 28, 28)
            nn.MaxPool2d(   # (16, 14, 14)
                kernel_size=2,  # 卷积窗口的大小
            ),
        )
        self.cov2 = nn.Sequential(  # -> (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2), # -> (32, 14, 14)
            nn.ReLU(),  # -> (32, 14, 14)
            nn.MaxPool2d(2) # -> (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)    # 优化函数
loss_func = nn.CrossEntropyLoss()   # 损失函数

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):

        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy : %.2f' % accuracy)


test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')