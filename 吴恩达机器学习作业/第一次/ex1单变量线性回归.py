import matplotlib.pyplot as plt

f = open("ex1data1.txt", 'r')
population = []
profit = []
for line in f.readlines():
    col1 = line.split(',')[0]
    col2 = line.split(',')[1].split('\n')[0]
    population.append(float(col1))
    profit.append(float(col2))
# plt.title('Scatter plot')
# plt.xlabel('Population')
# plt.ylabel('Profit')
# plt.grid()
# plt.scatter(population, profit, marker='x')
# plt.show()

# 初始化参数
alpha = 0.01                    # 学习速率
iteration = 1500                # 迭代次数
theta = [0, 0]                  # 初始化参数
m = len(population)             # 获取样本总数

# 梯度下降
for i in range(iteration):
    temp0 = theta[0]
    temp1 = theta[1]
    for j in range(m):
        temp0 -= (alpha / m) * (theta[0] + theta[1] * population[j] - profit[j])
        temp1 -= (alpha / m) * (theta[0] + theta[1] * population[j] - profit[j]) * population[j]
    theta[0] = temp0
    theta[1] = temp1

x = [5.0, 22.5]
y = [5.0 * theta[1] + theta[0], 22.5 * theta[1] + theta[0]]
plt.plot(x, y, color='red')
plt.title('Liner Regression')
plt.xlabel('population')
plt.ylabel("profit")
plt.scatter(population, profit, marker='x')
plt.show()

# 代价计算
c = 0.0
for j in range(m):
    c += (1.0 / (2 * m)) * pow((theta[0] + theta[1] * population[j] - profit[j]), 2)
print(c)
print(theta[0], theta[1])
