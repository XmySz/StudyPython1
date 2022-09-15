import numpy as np
import matplotlib.pyplot as plt

iterations = 6000

# 读入数据
f = open('ex1data2.txt', 'r')
house_size = []
bedroom_number = []
house_price = []

# 存入已定义好的列表中
for line in f.readlines():
    col1 = float(line.split(",")[0])
    col2 = float(line.split(",")[1])
    col3 = float(line.split(",")[2].split("\n")[0])
    house_size.append(col1)
    bedroom_number.append(col2)
    house_price.append(col3)

# 组合并重塑为数组
x1 = np.array(house_size).reshape(-1, 1)
x2 = np.array(bedroom_number).reshape(-1, 1)
y = np.array(house_price).reshape(-1, 1)
data = np.concatenate((x1, x2, y), axis=1)

mean = np.mean(data, axis=0)  # 计算每一列的平均值
ptp = np.ptp(data, axis=0)  # 计算每一列的最大最小值差
nor_data = (data - mean) / ptp  # ！！！！！！！！！！！！归一化(注意这个手法)
X = np.insert(nor_data[..., :2], 0, 1, axis=1)  # 添加x0=1
y = nor_data[..., -1]

c = []  # 存储计算的损失值


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


print(gradient_descent(X, np.zeros((3)), y, 0.01, iterations)[0])

plt.title("Visualizing J(θ)")
plt.xlabel("iterations")
plt.ylabel("cost")
plt.plot([i for i in range(iterations)], c, color="red")
plt.show()
