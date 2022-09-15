import numpy as np


# 正规方程
def normalEqn(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))


X = []
y = []
f = open('ex1data2.txt', 'r')
lines = f.readlines()
for line in lines:
    X.append(float(line.split(',')[0]))
    X.append(float(line.split(',')[1]))
    y.append(float(line.split(',')[2].split('\n')[0]))
X = np.array(X).reshape(47, 2)
y = np.array(y).reshape(47, 1)

print(normalEqn(X, y))
