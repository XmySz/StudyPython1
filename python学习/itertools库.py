from itertools import *  # 为高效循环而创建迭代器的函数


def _itertools():
    """
        无穷迭代器
        根据最短输入序列长度停止的迭代器
        排列组合迭代器
    :return:
    """
    # 无穷迭代器
    count(1)  # 创建一个从1开始步长为1的迭代器
    count(1, 2)  # 创建一个从1开始步长为2的迭代器
    cycle("abc")  # 创建一个循环可迭代对象元素的迭代器
    repeat(10)  # 创建一个重复无限次的迭代器
    repeat(10, 10)  # 创建一个重复十次10的迭代器

    # 根据最短输入序列长度停止的迭代器
    pass

    # 排列组合迭代器
    product("AB", "cdef")
    product(range(2), repeat=3)  # 计算可迭代对象自身的笛卡尔积,大致相当于生成器表达式中的嵌套循环。
    # 例如，product(A, B)和((x,y) for x in A for y in B) 返回结果一样。
    # 要计算可迭代对象自身的笛卡尔积，将可选参数 repeat 设定为要重复的次数。
    #  例如，product(A, repeat=4) 和 product(A, A, A, A) 是一样的。
    permutations([1, 2, 3, 4], 2)  # permutations(p[,r]),长度r元组，全排列
    combinations([1, 2, 3, 4], 2)  # combinations(p,r),长度r的元组,有序,无重复元素
    combinations_with_replacement([1, 2, 3, 4], 2)  # combinations_with_replacement(p,r),长度r元组，有序，元素可重复


d = {
    "2": "abc",
    "3": "def",
    "4": "ghi",
    "5": "jkl",
    "6": "mno",
    "7": "pqrs",
    "8": "tuv",
    "9": "wxyz",
}
digits = "23"
ans = []
if len(digits) == 1:
    ans.extend("".join(i) for i in product(d[digits[0]]))
elif len(digits) == 2:
    ans.extend("".join(i) for i in product(d[digits[0]], d[digits[1]]))
elif len(digits) == 3:
    ans.extend("".join(i) for i in product(d[digits[0]], d[digits[1]], d[digits[2]]))

else:
    ans.extend("".join(i) for i in product(d[digits[0]], d[digits[1]], d[digits[2]], d[digits[3]]))

print(ans)
