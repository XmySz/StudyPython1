import torch
from torch.autograd import Variable # Variable变量
import torch.nn.functional as F # 常用工具
import matplotlib.pyplot as plt # 画图包

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)  # 类别1所有点的坐标
y0 = torch.zeros(100)   # 类别1所有点的标签
x1 = torch.normal(-2*n_data, 1) #类别2所有点的坐标
y1 = torch.ones(100)    # 类别2所有点的标签
x = torch.cat((x0,x1),0).type(torch.FloatTensor)    # 连接所有点坐标
y = torch.cat((y0,y1), ).type(torch.LongTensor) # 连接所有点的标签

x,y = Variable(x), Variable(y)  # 转化为Variable变量

# plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=y.data.numpy(), s=100, lw=0)
# plt.show()


# 构建神经网络类
class Net(torch.nn.Module): # 继承父类
    def __init__(self, n_feature, n_hidden, n_output):  # 实例化时接收数个数，隐层神经元个数和输出个数三个参数
        super(Net, self).__init__() # 继承父类的初始化函数
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 把输入数据转化成隐层神经元
        self.predict = torch.nn.Linear(n_hidden, n_output) # 把隐层神经元个数输出为预测结果

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(2, 10, 2)
print(net)

plt.ion()   # 实时打印
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.05)   # 优化模型
loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数

for t in range(200):
    out = net(x) # 预测的值
    loss = loss_func(out, y) # 计算损失

    # 优化神经网络模型的参数
    optimizer.zero_grad()   # 梯度设为0
    loss.backward() # 反向传递
    optimizer.step()    # 优化梯度

    if t % 2 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]    # 把预测值转换成概率
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=pred_y, s=100, lw=0)
        accuracy = sum(pred_y == target_y) / 200
        plt.text(0.5, -4, 'Accurary=%.2f' % accuracy, fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()