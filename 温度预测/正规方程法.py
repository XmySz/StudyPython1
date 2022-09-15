import numpy as np
from matplotlib import pyplot as plt


def normalEqn(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y  # X.T@X 等价于 X.T.dot(X)
    return theta


f = open('train_data.txt', 'r')
num_inputs = 5  # 特征数
num_examples = 30000  # 样本数
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

X = np.concatenate((d1, d2, d3, d4, d5), axis=1)
pre_y = []

theta = normalEqn(X, y1)
print(theta)
for i in range(1, 1001):
    x1 = [1, 6, 35, 4, i]
    x1 = np.array(x1).reshape(1, 5)
    pre_y.append(x1.dot(theta))

pre_y = np.array(pre_y).reshape(-1)
plt.plot([i for i in range(1, 1001)], pre_y, label="predict")
plt.plot([i for i in range(1, 1001)], y1[2000:3000].reshape(-1), label='true')
plt.legend()
plt.show()
