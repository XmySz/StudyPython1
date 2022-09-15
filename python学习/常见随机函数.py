from random import *

print(randint(1,10))    # 生成范围内的随机整数
print(random()) # 生成一个0~1之间的浮点数
print(uniform(1.1,10.1))    # 生成范围内的浮点数
print(randrange(1,10,2))    # 生成范围内间隔为指定值的随机整数
print(choice([1,2,3,4,5,6,7]))  # 从序列中随机选取一个元素
print(shuffle([1,2,3,4,5,6,7])) # 打乱序列