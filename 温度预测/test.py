import torch.utils.data as Data
import torch
import torch.nn as nn
import numpy as np
import os
from torch.nn import init
import torch.optim as optim

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 导入数据集
f = open('train_data.txt', 'r')
num_inputs = 5  # 特征数
num_examples = 30000  # 样本数
batch_size = 20  # 批量大小
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
features = np.concatenate((d1, d2, d3, d4, d5), axis=1)

mean = np.mean(features, axis=0)  # 计算每一列的平均值
ptp = np.ptp(features, axis=0)  # 计算每一列的最大最小值差
nor_data = (features - mean) / ptp  # ！！！！！！！！！！！！归一化(注意这个手法)
X = np.insert(nor_data, 0, 1, axis=1)  # 添加x0=1
y = y1

c = []


# 梯度下降
def gradient_descent(X, theta, y, alpha, iterations):
    m = X.shape[0]  # 获取样本个数
    for i in range(iterations):
        theta -= (alpha / m) * X.T.dot(X.dot(theta) - y)
        c.append(cost(X, theta, y))
    return theta, c


def cost(X, theta, y):
    m = X.shape[0]
    return (1.0 / (2 * m)) * np.sum(np.power((X.dot(theta) - y), 2))


print(gradient_descent(X, np.array([1, 2, 3, 4, 5, 6]).reshape(6, 1), y, 0.03, 1000))
