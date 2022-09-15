from collections import Iterable

"""
    为什么需要生成器？
        不同于列表的一开始就固定长度，生成器可以实现一边循环一边计算，节省内存空间，generator
        列表使用[]创建，生成器使用（）创建
"""

# 创建生成器
s = (x for x in range(10))
s1 = iter(range(10))
print(next(s))  # 该函数会返回一个新的生成器对象
print(type(s))

# 遍历生成器
next(s)  # 方式1
for i in s:  # 方式2
    print(i)


# 生成器函数（函数里有yield语句，就会自动转变为生成器，返回一个生成器对象）
# 执行方式：普通函数是顺序执行，遇到return语句或者最后一行函数语句就返回。
# 而变成generator的函数，在每次调用next()的时候执行，遇到yield语句返回，再次执行时从上次返回的yield语句处继续执行。
def odd():
    print('step1')
    yield 1
    print('step2')
    yield 2
    print('step3')
    yield 3


for i in odd():
    print(i)


def triangles(n=10):
    l = [1]
    while n > 0:
        yield l
        l = [1 if n in [0, len(l)] else l[n - 1] + l[n] for n in range(len(l) + 1)]
        n -= 1


for i in triangles():
    print(i)

"""
    迭代器
        1.iterable(可迭代对象):list,tuple,set,string,dict等
        2.iterator(迭代器):包括生成器和带yield的生成器函数
        生成器都是Iterator对象，但是Iterable不一定是Iterator，实现了__next__()方法的对象才是迭代器
"""
# 判断某个对象是否是可迭代对象
print(isinstance('abc', Iterable))
print(isinstance({1, 2, 3}, Iterable))
print(isinstance(1024, Iterable))
print(isinstance([1, 2, 3], Iterable))
