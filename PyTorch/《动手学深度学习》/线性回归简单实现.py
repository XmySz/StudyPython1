import torch
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import init  # 提供初始化参数的功能
import torch.optim as optim  # 该模块提供了优化算法
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

batch_size = 10

# 训练集样本数为1000，特征数为2，使用真实权重[2, -3.4]和偏差4.2生成数据，并添加随机噪声项（服从均值为0，方差为1的高斯分布）
# 生成数据集
num_inputs = 2  # 特征数
num_examples = 1000  # 样本数
true_w = [2, -3.4]  # 真实权重
true_b = 4.2  # 真实偏差
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 使用data包来读取数据
dataset = Data.TensorDataset(features, labels)  # 组合特征和标签
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)

# 初始化模型参数
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)

# 训练模型
num_epochs = 3

ls = []
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        ls.append(l)
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

# 绘制损失值随迭代次数变化的曲线
for i in range(len(ls)):
    ls[i]=ls[i].detach().numpy()

plt.plot([i for i in range(300)], ls)
plt.show()
print(true_w, net.linear.weight)
print(true_b, net.linear.bias)

