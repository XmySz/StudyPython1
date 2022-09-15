"""
    effeictive python书籍笔记:
        1.不要通过长度判断容器或者序列是不是为空,python会把空值自动判断为false
        2.3.5之前的版本dict字典不保证迭代顺序与插入顺序一致
        3.指定sort函数的key参数进行复杂排序，如果排序要依据的指标有很多项，可以把他们放在一个元组里，
"""
import contextlib

# 使用f-string代替str.format和%格式化字符串操作
key = 1
value = 1.234
print(f"{key} = {value}")
print(f"{key:x>3d} = {value:4.2f}")

# 海象操作符:= 赋值表达式的值就是赋给海象运算符左边那个标识符的值
if (a := 1) == 1: print(a)

# 使用contextlib.suppress取代try/except with pass
with contextlib.suppress(ZeroDivisionError):
    a = 1/2
    print(100)
