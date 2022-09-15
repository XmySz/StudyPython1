import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import glob
import os
from torch.utils.data import Dataset
import random
from torch import optim
import numpy as np


class DoubleConv(nn.Module):
    """每个DoubleConv模块由两个“Conv2d+NatchNorm2d+ReLU”组成"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Down模块由一个“MaxPool2d+DoubleConv”组成"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    # 上采样和双卷积 裁切
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat((x2, x1), dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, n_channels, out_classes, bilinear=False):
        super(Unet, self).__init__()
        self.in_channels = n_channels
        self.classes = out_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)  # 第一层，输入通道1输出通道64

        self.down1 = Down(64, 128)  # 左侧第二层
        self.down2 = Down(128, 256)  # 左侧第三层
        self.down3 = Down(256, 512)  # 左侧第三层
        fac = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // fac)  # 左边第四层

        self.up1 = Up(1024, 512 // fac, bilinear)  # TransposedConv + copy_crop + DoubleConv生成右侧分支第四层
        self.up2 = Up(512, 256 // fac, bilinear)  # 同上，生成右侧第三层
        self.up3 = Up(256, 128 // fac, bilinear)  # 同上，生成右侧第二层
        self.up4 = Up(128, 64, bilinear)  # 同上，生成右侧第二层

        self.out_conv = OutConv(64, out_classes)  #

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out_conv(x)


# 加载自己的数据集
class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'test/*.tif'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


def train_net(net, device, data_path, epochs=10, batch_size=2, lr=1e-05):
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(isbi_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.99)

    losser = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    for epoch in range(epochs):
        print(f"epoch {epoch}")
        net.train()
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = net(image)
            loss = losser(pred, label)
            print("loss/train", loss.item())
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'data/datasets/ISBI/best_model.pth')
            loss.backward()
            optimizer.step()


isbi_dataset = ISBI_Loader("data/datasets/ISBI")
print("数据个数：", len(isbi_dataset))
train_loader = torch.utils.data.DataLoader(isbi_dataset, batch_size=2, shuffle=True)
for image, label in train_loader:
    print(image.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Unet(n_channels=1, out_classes=1)
net.to(device=device)
data_path = "data/datasets/ISBI"
train_net(net, device, data_path)


# 加载模型参数
net.load_state_dict(torch.load('data/datasets/ISBI/best_model.pth', map_location=device))
# 评估模式
net.eval()
# 读取所有图片路径
tests_path = glob.glob('data/datasets/ISBI/test/*.tif')
print(tests_path)

for test_path in tests_path:
    # 设置保存路径
    save_res_path = test_path.split('.')[0] + '_res.png'
    # 读取图片
    img = cv2.imread(test_path)
    # 转为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 转为batch为1，通道为1，大小为512*512的数组
    img = img.reshape(1, 1, img.shape[0], img.shape[1])
    # 转为tensor
    img_tensor = torch.from_numpy(img)
    # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    # 预测
    pred = net(img_tensor)
    # 提取结果
    pred = np.array(pred.data.cpu()[0])[0]
    print(pred)
    print(pred.shape)
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0
    cv2.imwrite(save_res_path, pred)
