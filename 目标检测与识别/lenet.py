import os
import time
import load_dataset
import numpy as np
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
            nn.Conv2d(3, 6, 5),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(355216, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 200)
        )

    def forward(self, img):
        feature = self.conv(img)
        print(feature)
        output = self.fc(feature.view(img.shape[0], -1))
        print(output)
        return output


train_iter, test_iter = load_dataset.train_iter, load_dataset.test_iter

net= LeNet()
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
dzl.train_ch5(net, train_iter, test_iter, 64, optimizer, device, num_epochs)