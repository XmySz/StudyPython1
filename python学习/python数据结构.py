# Python3数据结构

# 一.列表(可变)
def _list():
    # 列表生成式
    list0 = []  # 创建新列表
    list1 = [1, 2, 2, 3, 4, 5, 3, 10, 7, 199]
    list2 = [6, 7, 8, 9]
    list3 = list(range(1, 11))
    list4 = [x * x for x in range(1, 11)]
    list5 = [x * x for x in range(1, 11) if x % 2 == 0]
    list6 = [m + n for m in 'abc' for n in 'def']
    list7 = [x if x % 2 == 0 else -x for x in range(1, 11)]

    # 列表切片(-1下标代表最后一个元素)
    print(list1[1:3])
    print(list1[:3])
    print(list1[1:])
    print(list1[1:9:2])
    print(list1[::-1])

    # 列表操作
    print(list1 + list2)  # 列表叠加
    print(list1 * 2)  # 列表重复
    print(1 in list1)  # 判断某个元素是否在列表内
    list1.append(6)  # 追加元素
    list1.extend(list2)  # 拓展列表
    list1.insert(8, 100)  # 指定位置插入
    list1.remove(1)  # 移除元素(按值)，只会删除第一个
    list1.pop(0)  # 移除元素(按照下标),无参数则移除末尾一个元素
    list1.clear()  # 移除所有元素
    list1.index(5)  # 返回指定值的下标
    list1.count(2)  # 返回指定值在列表中出现的次数
    list1.sort()  # 排序列表(正序)为永久性排序
    list1.sort(reverse=True)  # 逆序
    list1.reverse()  # 排序列表(逆序)
    len(list1)  # 获取列表的长度
    list3 = list1.copy()  # 复制列表
    del list1[0:4]  # 使用 del 语句可以从一个列表中依索引而不是值来删除一个元素
    for i, j in enumerate(['a', 'b', 'c']):     # 把一个列表变成索引-元素对
        print(i, j)


# 二.字符串(不可变)
def _string():
    """
    格式化    含义
      %c	 格式化字符及其ASCII码
      %s	 格式化字符串
      %d	 格式化整数
      %u	 格式化无符号整型
      %o	 格式化无符号八进制数
      %x	 格式化无符号十六进制数
      %X	 格式化无符号十六进制数（大写）
      %f	 格式化浮点数字，可指定小数点后的精度
      %e	 用科学计数法格式化浮点数
      %E	 作用同%e，用科学计数法格式化浮点数
      %g	 %f和%e的简写
      %G	 %F 和 %E 的简写
      %p	 用十六进制数格式化变量的地址
    """
    # 字符串格式化函数
    price = 52
    txt = 'The price is {} dollars'
    txt1 = 'The price is {:.2f} dollars'
    txt2 = 'The price is {0} dollars'
    txt3 = 'The price is {name1} dollars'
    print(txt3.format(name1=66))

    # 字符串的格式化 参考https://www.runoob.com/python/att-string-format.html
    """
        %[(name)][flags][width].[precision]typecodename)][flags][width].[precision]typecode
            flags:可以有 +，-，' '或 0。+表示右对齐。-表示左对齐。' '为一个空格，表示在正数的左侧填充一个空格，从而与负数对齐。0表示使用0填充。
            width:表示显示宽度
            precision:表示小数点后精度
            typecodename:类型码
                %s    字符串 (采用str()的显示)s    字符串 (采用str()的显示)
                %r    字符串 (采用repr()的显示)%r    字符串 (采用repr()的显示)
                %c    单个字符%c    单个字符
                %b    二进制整数%b    二进制整数
                %d    十进制整数%d    十进制整数
                %i    十进制整数%i    十进制整数
                %o    八进制整数%o    八进制整数
                %x    十六进制整数%x    十六进制整数
                %e    指数 (基底写为e)%e    指数 (基底写为e)
                %E    指数 (基底写为E)%E    指数 (基底写为E)
                %f    浮点数%f    浮点数
                %F    浮点数，与上相同%g    指数(e)或浮点数 (根据显示长度)%F    浮点数，与上相同%g    指数(e)或浮点数 (根据显示长度)
                %G    指数(E)或浮点数 (根据显示长度)%G    指数(E)或浮点数 (根据显示长度)
                %%    字符"%"%%    字符"%"
        {:x<4d}:数字补x (填充右边, 宽度为4)
        {:x>4d}:数字补x (填充左边, 宽度为4)
        ^, <, > 分别是居中、左对齐、右对齐，后面带宽度， : 号后面带填充的字符，只能是一个字符，不指定则默认是用空格填充。
        + 表示在正数前显示 +，负数前显示 -；  （空格）表示在正数前加空格
    """

    # 字符串的常见操作
    print('a' in "abc")  # 成员运算符
    print('a' not in "abc")  # 成员运算符
    print(r"\n\n\n\n")  # 原始字符串
    print(str(1.234))  # 转换为字符串
    print(str.capitalize('qwer'))  # 首字母大写, 其他字母变成小写
    print('abcdaaaa'.count('a'))  # 统计指定子串的出现次数            str.count(sub, start, end)
    print('abcdefg'.find('d'))  # 寻找子串首次出现位置(从左向右)    str.find(sub, start, end)
    print('abcdefg'.rfind('d'))  # 寻找子串首次出现位置(从右向左)    str.rfind(sub, start, end)
    print("abcdefgaabbccd".replace("a", "b", 2))  # 把string中的str1替换成str2,如果num指定，则替换不超过num 次.
    print("aaa".center(10))  # 原字符串居中,并用空格填充至width长度的字符串
    print("aBcD".swapcase())  # 大小写反转
    print("sandoainfnwqoid".startswith("sa", __start=0, __end=10))  # 判断字符串是否以某个字串结尾
    print("sandoainfnwqoid".endswith("id", __start=1, __end=10))  # 判断字符串是否以某个字串结尾
    print(str.isalnum('aaaaa asf'))  # 判断字符串是否全是字母或数字
    print(str.isalpha('asnasfn1312'))  # 判断字符串是否全是字母
    print(str.isdigit('13241emeqw'))  # 判断字符串是否全是数字
    print('(\n\r\f\t\v)'.isspace())  # 判断字符串是否只包含(\n、\r、\f、\t、\v)
    print('abc'.islower())  # 判断指定字符串是否全是小写字母
    print('abc'.isupper())  # 判断指定字符串是否全是大写字母
    print(','.join(['p', 'y', 't', 'h', 'o', 'n']))  # str.join(iterable)以指定字符串作为分隔符，将iterable中所有的元素(必须是字符串)合并为一个新的字符串。
    print('abc'.lower())  # 指定字符串转换为小写
    print('abc'.upper())  # 指定字符串转换为大写
    print('abc'.split(sep=',', maxsplit=-1))  # sep参数指定分隔符，maxsplit参数为最大分割次数，-1代表无限次
    print(str.strip("  hello,world  "))  # 去掉字符串首尾空格
    print(' asf '.rstrip())  # 去除字符串末尾空白
    print(' asad '.lstrip())  # 去除字符串开头空白


# 三.元组(适用于存储在程序运行期间可能变化的数据集)(不可变)
def _tuple():
    """
        元组的相关操作
    """
    a = ()  # 创建一个空元组
    a1 = (1,)  # 创建只有一个元素的元组
    a2 = (1)  # 等价于a2 = 1
    a3 = 1, 2, 3  # 默认为元组
    b = 4, 5, 6
    c = a3 + b  # 元组虽然不可以修改值，但可以拼接两个元组
    del c  # 删除元组


# 四.集合(无序不重复元素的集,基本功能包括关系测试和消除重复元素。)(可变)
def _set():
    basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
    print(basket)  # 自动删除重复的元素
    print('apple' in basket)  # 检测成员
    a = set('abracadabra')
    b = set('alacazam')
    print(a - b)  # 在a中不在b中
    print(a | b)  # 在a中或在b中
    print(a & b)  # 同时在a和b中
    print(a ^ b)  # 不同时在a和b中


# 五.字典(无序的键值对,键必须唯一)(可变)
def _dict():
    """
        d = {key1 : value1, key2 : value2 }
        键必须唯一切是不可变类型,如果出现了两个键一样,则保留后面那个键对应的值
        值不必须唯一
    :return:
    """
    # 字典的创建
    book = dict()  # 创建空字典
    mapping = zip(("第十二章-彻底掌握Python的字典类型",), (210,))
    book1 = dict(mapping)  # 使用mapping对象来创建字典
    book2 = {
        "name": "StudyPython",
        "price": 1,
    }
    book3 = {zip(["a", "b", "c"], [1, 2, 3])}  # 使用列表来创建字典
    alien0 = {'color': 'blue', 'point': 5, 'xposition': 0, 'yposition': 25}

    # 字典的遍历
    for key, value in alien0.items():  # 遍历字典
        print(f'{key}: {str(value)}')
    for key in alien0:  # 遍历字典中的键
        print(key)
    for value in alien0.values():  # 遍历字典中的值
        print(value)

    # 字典的操作
    print(alien0["name"])  # 访问字典
    alien0.copy()  # 返回字典的一个浅复制
    sorted(alien0)  # 字典临时排序
    dict.fromkeys(["a", 'b', 'c'], 3)
    alien0.get("name")  # 得到指定键的值
    alien0.items()  # 以列表的方式返回可遍历的(key,val)元组
    alien0.keys()  # 以列表返回一部字典所有的键
    alien0.values()  # 以列表返回一部字典所有的值
    alien0.pop("name")  # 删除字典给定键 key 所对应的值，返回值为被删除的值
    alien0.popitem()  # 返回并删除字典中的最后一对键和值。
    alien0.update({"size": 100})  # 把字典dict2的键/值对更新到dict里
    alien0["sex"] = "female"  # 修改字典
    del alien0['color']  # 删除字典的某个键是指定值的条目
    alien0.clear()  # 清空字典中的所有条目
    # del alien0  # 删除字典


# 六.Number数字类型(不可变)
def _number():
    """
        四种类型    int/float/bool/complex
                   10  10.0  True  1+2j
        类型转换
        int(x [,base ])         将x转换为一个指定了进制的整数
        long(x [,base ])        将x转换为一个长整数
        float(x)                将x转换到一个浮点数
        complex(real [,imag ])  创建一个复数
        str(x )                 将对象 x 转换为字符串
        repr(x )                将对象 x 转换为表达式字符串
        eval(str )              用来计算在字符串中的有效Python表达式,并返回一个对象
        tuple(s )               将序列 s 转换为一个元组
        list(s )                将序列 s 转换为一个列表
        chr(x )                 将一个整数转换为一个字符
        unichr(x )              将一个整数转换为Unicode字符
        ord(x )                 将一个字符转换为它的整数值
        hex(x )                 将一个整数转换为一个十六进制字符串
        oct(x )                 将一个整数转换为一个八进制字符串
        bin(x )                 将一个整数转换为一个二进制字符串
    :return:
    """
