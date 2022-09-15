import torch.utils.data as Data
import torch
import torch.nn as nn
import numpy as np
import os

from matplotlib import pyplot as plt
from torch.nn import init
import torch.optim as optim
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 导入数据集
f = open('train_data.txt', 'r')
num_inputs = 5  # 特征数
num_examples = 30000  # 样本数
batch_size = 60  # 批量大小
d1, d2, d3, d4, d5, y1 = [], [], [], [], [], []
for line in f.readlines():
    col1 = float(line.split(",")[0])
    col2 = float(line.split(",")[1])
    col3 = float(line.split(",")[2])
    col4 = float(line.split(",")[3])
    col5 = float(line.split(",")[4])
    col6 = float(line.split(",")[5].split("\n")[0])
    d1.append(col1)
    d2.append(col2)
    d3.append(col3)
    d4.append(col4)
    d5.append(col5)
    y1.append(col6)
d1 = np.array(d1).reshape(-1, 1)
d2 = np.array(d2).reshape(-1, 1)
d3 = np.array(d3).reshape(-1, 1)
d4 = np.array(d4).reshape(-1, 1)
d5 = np.array(d5).reshape(-1, 1)
y1 = np.array(y1).reshape(-1, 1)

features = torch.tensor(np.concatenate((d1, d2, d3, d4, d5), axis=1), dtype=torch.float32)  # 特征
labels = torch.tensor(y1, dtype=torch.float32)  # 标签

dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)

# 初始化模型参数
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
optimzier = torch.optim.Adam(net.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False)

los = []
# 训练模型
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        ls = loss(output, y.view(-1, 1))
        los.append(ls)
        optimzier.zero_grad()
        ls.backward()
        optimzier.step()
    print('epoch %d, loss: %f' % (epoch, ls.item()))

for i in range(len(los)):
    los[i]=los[i].detach().numpy()

plt.plot([i for i in range(5000)], los)
plt.show()