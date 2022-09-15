import torch
from torch.autograd import Variable # Variable变量
import torch.nn.functional as F # 常用工具
import matplotlib.pyplot as plt # 画图包

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # 随机生成100个数的等差数列
y = x.pow(2) + 0.2*torch.rand(x.size())

x,y = Variable(x), Variable(y)  # 转化为Variable变量

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


# 构建神经网络类
class Net(torch.nn.Module): # 继承父类
    def __init__(self, n_feature, n_hidden, n_output):  # 实例化时接收输入个数，隐层神经元个数和输出个数三个参数
        super(Net, self).__init__() # 继承父类的初始化函数
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 把输入数据转化成隐层神经元
        self.predict = torch.nn.Linear(n_hidden, n_output) # 把隐层神经元个数输出为预测结果

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x





def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)  # 优化模型,learnrate学习效率
    loss_func = torch.nn.MSELoss()  # 均方差损失函数

    for t in range(200):  # 训练200次
        prediction = net1(x)  # 预测的值
        loss = loss_func(prediction, y)  # 计算损失

        # 优化神经网络模型的参数
        optimizer.zero_grad()  # 梯度设为0
        loss.backward()  # 反向传递
        optimizer.step()  # 优化梯度

    plt.figure(1, figsize=(10,3))
    plt.subplot(131)
    plt.title('net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    torch.save(net1, 'net.pkl') # 保存整个神经网络
    torch.save(net1.state_dict(), 'net_parames.pkl')    # 保存所有参数


def restore_net():
    net2= torch.load('net.pkl') # 提取
    prediction = net2(x)  # 预测的值

    plt.subplot(132)
    plt.title('net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net3.load_state_dict(torch.load('net_parames.pkl'))
    prediction = net3(x)  # 预测的值

    plt.subplot(133)
    plt.title('net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()

save()
restore_net()
restore_params()