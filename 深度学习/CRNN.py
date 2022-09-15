import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class vgg_16(nn.Module):
    """
        特征提取使用vgg16的一部分，并加上了批量归一化层，以加快速度
    """
    def __init__(self):
        super(vgg_16, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        # 池化层的这个改动是因为文本图像多数都是高较小而宽较长，所以其feature map也是这种高小宽长的矩形形状，如果使用1×2的池化窗口则更适合英文字母识别
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(2, 1))
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(2, 1))
        self.conv7 = nn.Conv2d(512, 512, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.pool1(x)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.pool2(x)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = self.pool3(x)
        x = self.conv5(x)
        x = F.relu(self.norm1(x), inplace=True) # 批量归一化是紧跟在卷积之后的，其次才是激活
        x = self.conv6(x)
        x = F.relu(self.norm2(x), inplace=True)
        x = self.pool4(x)
        x = F.relu(self.conv7(x), inplace=True)
        return x


class RNN(nn.Module):
    """
        使用双向LSTM来学习关联信息生成预测标签
    """
    def __init__(self, class_num, hidden_unit):
        """
        :param class_num:字符的类别个数
        :param hidden_unit: 隐层单元个数
        """
        super(RNN, self).__init__()
        self.BiRNN1 = nn.LSTM(512, hidden_unit, bidirectional=True) # 参数分别为输入向量的维度，隐层单元数,隐层层数
        self.embedding1 = nn.Linear(hidden_unit * 2, 512)
        self.BiRNN2 = nn.LSTM(512, hidden_unit, bidirectional=True)
        self.embedding2 = nn.Linear(hidden_unit * 2, class_num)

    def forward(self, x):
        """
            # LSTM输入的形状为 (input,  (h0, c0))
                    input(时间步长, 批大小， 输入的维度)
                    h0(num_layers*num_directins，批大小，隐层单元个数)         如果bidirectional设置了为True，num_directions则为2，否则为1。
                    c0同上
            # LSTM输出的形状为 (output, (h_n, c_n))
                    output(时间步长, 批大小, 隐层单元个数*隐层层数)
                    h_n:同上
                    c_n:同上
        :param x:
        :return:
        """
        x = self.BiRNN1(x)
        T, b, h = x[0].size()   # 也即是output
        x = self.embedding1(x[0].view(T * b, h))
        x = x.view(T, b, -1)
        x = self.BiRNN2(x)
        T, b, h = x[0].size()
        x = self.embedding2(x[0].view(T * b, h))
        x = x.view(T, b, -1)
        return x    # x.shape:(时间步长，批大小，字符类别总数)


class CRNN(nn.Module):
    """
        参数：类别个数，隐层单元个数
    """
    def __init__(self, class_num, hidden_unit=256):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential()
        self.cnn.add_module("vgg_16", vgg_16())
        self.rnn = nn.Sequential()
        self.rnn.add_module("rnn", RNN(class_num, hidden_unit))

    def forward(self, x):
        x =self.cnn(x)
        b, c, h, w = x.size()   # 分别对应批大小，通道数，高和宽
        print(x.size())
        assert h == 1   # 高度必须被压缩成1
        x = x.squeeze(2)    # 移除掉h维，变成(b, 512, w)
        x = x.permute(2, 0, 1)  # 调整每个维度的位置, 变为[w, b, c] = [seq_len, batch, input_size]
        x = self.rnn(x)
        return x


net = CRNN(27)
loss_function = nn.CTCLoss()
print(net)
print(loss_function)