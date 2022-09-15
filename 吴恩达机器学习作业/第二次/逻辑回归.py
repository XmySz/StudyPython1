import numpy as np
import matplotlib.pyplot as plt


# 读数据
def read_data_fromfile(filename, type_tuple, separator=','):
    """
        从文件中读入数据,文件的数据存储应该是每组数据存在一行并用分隔符分开
        返回 ndarray
    """
    f = open(filename, 'r')
    lines = f.readlines()
    data = []

    if len(type_tuple) != len(lines[0]) and len(type_tuple) == 1:
        for line in lines:
            line = line[:-1]
            line = line.split(sep=separator)
            row = []
            for col in line:
                row.append(type_tuple[0](col))
            data.append(row)
    elif len(type_tuple) == len(lines[0].split(sep=separator)):
        for line in lines:
            line = line[:-1]
            line = line.split(sep=separator)
            row = []
            for i in range(len(line)):
                row.append(type_tuple[i](line[i]))
            data.append(row)
    else:
        data = None
    return np.array(data)


# 分隔数据
def separate_data(data, col, boundary):
    """
    将数据按照某列进行二分类

    parameters:
    ----------
    data : ndarray
            一组数据存在一行
    col : int
            分类标准应用到的列号
    boundary : double
            分类边界
    """
    data0 = np.array(data)
    data1 = np.array(data)
    dc0 = 0
    dc1 = 0
    for i in range(data.shape[0]):
        if data[i][col] < boundary:
            data1 = np.delete(data1, i - dc1, axis=0)
            dc1 += 1
        else:
            data0 = np.delete(data0, i - dc0, axis=0)
            dc0 += 1
    return data0, data1


# sigmoid函数
def sigmoid(z): return 1 / (1 + np.exp(-z))


# 代价函数
def costFuction(theta, X, y):
    return np.mean((-y)*np.log(sigmoid(X.dot(theta))) - (1-y) * np.log(1 - sigmoid(X.dot(theta))))


# 梯度下降函数（一次）
def gradient(theta, X, y):
    return X.T.dot(sigmoid(X.dot(theta)) - y) / X.shape[0]


# 梯度下降实现
def gradient_descent(theta, X, y, alpha, iterations):
    for i in range(iterations):
        theta -= alpha*gradient(theta, X, y)
        c.append(costFuction(theta, X, y))
    return theta


data = read_data_fromfile('ex2data1.txt', (float, float, float))
data0, data1 = separate_data(data, -1, 0.5)

# plt.title("raw data scatter")
# plt.xlabel("exam1 score")
# plt.ylabel("exam2 score")
# plt.xlim((20, 110))
# plt.ylim((20, 110))
# na = plt.scatter(data0[..., 0], data0[..., 1], marker='x', c='b', label='not admitted')
# a = plt.scatter(data1[..., 0], data1[..., 1], marker='x', c='y', label='admitted')
# plt.legend(handles=[na, a], loc='upper right')

data = np.array(data)
X = np.insert(data[..., :2], 0, 1, axis=1)
y = data[..., -1]
theta = np.zeros(3)


alpha = 0.2
iterations = 1000
c = []
# 特征归一化
mean = np.mean(X[..., 1:], axis=0)
std = np.std(X[..., 1:], axis=0, ddof=1)
X[..., 1:] = (X[..., 1:] - mean) / std

print(gradient_descent(theta, X, y, alpha, iterations))
print(c)

# plt.plot(range(iterations), c, c='b')
# plt.show()

# 画出决策边界
plt.subplot(1, 1, 1)
plt.scatter(data0[..., 0], data0[..., 1], marker='x', c='b', label="not admitted")
plt.scatter(data1[..., 0], data1[..., 1], marker='x', c='y', label="admitted")
x1 = np.arange(20, 110, 0.1)
# 因为进行了特征缩放，所以计算y时需要还原特征缩放
x2 = mean[1] - std[1] * (theta[0] + theta[1] * (x1 - mean[0]) / std[0]) / theta[2]
db = plt.plot(x1, x2, c='r', label="decision boundary")
plt.xlim((20, 110))
plt.ylim((20, 110))
plt.title("decision boundary")
plt.xlabel("exam1 score")
plt.ylabel("exam2 score")
plt.legend(handles=db, loc="upper right")
plt.show()
