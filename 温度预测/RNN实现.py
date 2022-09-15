import torch.utils.data as Data
import torch
import torch.nn as nn
import numpy as np
import os
from matplotlib import pyplot as plt, rcParams
from torch.nn import init
import torch.optim as optim

rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

f = open('train_data.txt', 'r')
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

# 设置全局变量
num_time_steps = 20  # 时间步长
input_size = 1  # 输入数据的维度
hidden_size = 20  # 隐层单元的个数
batch_size = 1  # 批大小
num_layers = 1  # 隐层层数
output_size = 1  # 输出数据的维度
lr = 0.001  # 学习率
nums_iter = 30000  # 迭代次数
pre_segment = 25000  # 预测哪一段


# 创建模型
class RNNmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNmodel, self).__init__()
        # self.rnn = nn.RNN(
        #     input_size=input_size,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     batch_first=True,
        # )
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        for i in self.rnn.parameters():
            nn.init.normal_(i, mean=0.0, std=0.01)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, X, hidden_prev):
        out, hidden_prev1 = self.rnn(X, hidden_prev)
        out = out.view(-1, hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        return out, hidden_prev1


# 训练模型
def train_Rnn_model(data):
    net = RNNmodel(input_size, hidden_size, num_layers)
    print('model: \n', net)
    loss = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr)
    hidden_prev = torch.zeros(num_layers, batch_size, hidden_size)
    ls = []  # 记录损失值
    for iter in range(nums_iter):
        start = np.random.randint(1000 - 2 * num_time_steps, size=1)[0]  # 末尾加[0]，是为了把array数组转换成具体的数
        end = start + num_time_steps
        # 随机选择连续的十个点作为输入，预测后十个
        x = torch.tensor(data[start:end]).float().view(1, num_time_steps, input_size)
        y = torch.tensor(data[start + num_time_steps:end + num_time_steps]).float().view(1, num_time_steps, input_size)
        output, hidden_prev = net(x, hidden_prev)
        hidden_prev = hidden_prev.detach()

        los = loss(output, y)
        net.zero_grad()
        los.backward()
        optimizer.step()

        if iter % 100 == 0:
            print("Iteration: {}, loss: {}".format(iter, los.item()))
            ls.append(los.item())

    plt.plot(ls, 'r')
    plt.xlabel('训练次数')
    plt.ylabel('loss')
    # plt.title('RNN损失函数下降曲线')
    plt.title('GRU损失函数下降曲线')
    plt.show()

    return hidden_prev, net


h_pre, model = train_Rnn_model(y1[pre_segment:pre_segment+1000])   # 决定训练哪个部分

pre = list(y1[pre_segment:pre_segment + num_time_steps].reshape(-1))    # 前时间步长个数据添加进列表

for i in range(0, 1000 - num_time_steps, num_time_steps):
    data_test = y1[pre_segment + i:pre_segment + num_time_steps + i]
    data_test = torch.tensor(np.expand_dims(data_test, 0), dtype=torch.float32)
    pred1, h1 = model(data_test, h_pre)
    pre.extend(list(pred1.view(-1).detach().numpy()))

plt.plot([i for i in range(1000)], list(y1[pre_segment:pre_segment+1000].reshape(-1)), label='Truth')
plt.plot([i for i in range(1000)], pre, label='predict')
plt.legend()
plt.xlabel("时间(ms)", loc='right')
plt.ylabel("温度(°C)", loc='top')
plt.show()
