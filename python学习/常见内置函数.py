# 常见内置函数
a = 2
print(abs(-1))
print(max([1, 2, 3, 4, 5, 6, 7, 8]))
print(min([1, 2, 3, 4, 5, 6, 7, 8]))
print(sum([1, 2, 3, 4, 5, 6, 7, 8]))
print(sorted([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))
print(len([1, 2, 3]))
print(divmod(2, 6))                             # 获取商和余数
print(pow(2, 3))
print(round(5.21556, 3))                        # 获取指定位数的浮点数
print(list(range(1, 10)))
print(list(reversed([1, 2, 3, 4, 5])))          # 反转
print(id(a))                                    # 输出a的地址
print(hasattr(object, 'title'))                 # 判断某个对象是否有某属性
print(type('title'))                            # 判断某个对象是的类型
print(isinstance('title', object))              # 判断某个对象是否是某个类的实例化（子类是父类）
# zip(*iterables, strict=False)                 # 多个迭代器并行迭代

# print()函数
num = 100
num_sqrt = 10
print("hello,world")                            # 直接输出
print("{0}{1}{2}".format(0, 1, 2))              # 格式化输出方式1
print("%d的平方是%d" %(10, 100))                 # 格式化输出方式2
print(f"{num}的平方根是{num_sqrt:.3f}")
print('a', 'b', 'c', sep='OK!!')                # 指定间隔字符
print('hello', 'world', end=',')                # 指定结尾



