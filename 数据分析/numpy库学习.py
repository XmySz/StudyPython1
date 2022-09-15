'''
        numpy支持大量高维度数组与矩阵运算。NumPy的主要对象是多维数组 Ndarray。
        NumPy 数组比 Python 列表更快、更紧凑。数组占用内存少，使用方便。
        NumPy 使用更少的内存来存储数据，它提供了一种指定数据类型的机制。这允许进一步优化代码。
       在NumPy中维度叫做轴Axes，轴的个数叫做秩Rank。
        NumPy 官网 http://www.numpy.org/
        NumPy 源代码：https://github.com/numpy/numpy
        SciPy 官网：https://www.scipy.org/
        SciPy 源代码：https://github.com/scipy/scipy
        Matplotlib 官网：https://matplotlib.org/
        Matplotlib 源代码：https://github.com/matplotlib/matplotlib
'''
import numpy as np


# numpy数组的常见属性
def attribute_np():
    """
        numpy数组的常用属性:
            ndarray.ndim	    秩，即轴的数量或维度的数量
            ndarray.shape	    数组的形状（元组），对于矩阵，n行m列
            ndarray.size	    数组元素的总个数，相当于 .shape中 n*m 的值
            ndarray.dtype	    ndarray对象的元素类型
            ndarray.itemsize	ndarray对象中每个元素的大小，以字节为单位
            ndarray.flags	    ndarray对象的内存信息
            ndarray.real	    ndarray元素的实部
            ndarray.imag	    ndarray元素的虚部
            ndarray.data	    包含实际数组元素的缓冲区，由于一般通过数组的索引获取元素，所以通常不需要使用这个属性。
    :return:
    """
    a = np.array([[1, 2, 3], [4, 5, 6]])
    print(a.ndim)
    print(a.shape)
    print(a.size)
    print(a.dtype)
    print(a.itemsize)
    print(a.flags)
    print(a.data)


# 创建numpy数组的常见方式
def create_np():
    """
        创建numpy数组的几种常见方式
        常见参数:
            object	数组或嵌套的数列
            dtype	数组元素的数据类型，可选
            copy	对象是否需要复制，可选
            order	创建数组的样式，C为行方向，F为列方向，A为任意方向（默认）
            subok	默认返回一个与基类类型一致的数组
            ndmin	指定生成数组的最小维度
    """
    x = np.empty(3, 2)  # 创建一个指定形状数据类型的未初始化的数组
    x1 = np.zeros(3, 2)  # 以0填充新数组
    x2 = np.ones((3, 2), dtype=int)  # 以1填充新数组
    x3 = np.full((2, 2), 4)  # 以指定数填充数组
    x4 = np.random.random((1, 3))  # 创建随机数组
    x5 = np.random.randn(2, 3)  # 创建标准正态分布数组
    x6 = np.random.randint(10, 20, (3, 4))  # 创建随机分布的整数型数组
    x11 = np.random.normal(1, 2, (1000, 1)) # 创建平均值1,方差2的指定形状的数组
    x7 = np.arange(10)  # 从数值范围创建数组 arange(start,stop,step,dtype)
    x8 = np.asarray([1, 2, 3, 4])  # 从已有数组创建数组
    x9 = list(range(100))  # 使用迭代器创建数组
    it = iter(x2)
    x10 = np.fromiter(it, dtype=float)
    # 创建一维等差数列数组    np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    #                       endpoint参数为是否显示stop，默认不显示，restep参数为是否显示间隔
    x11 = np.linspace(10, 20, 5, endpoint=True)

    # 创建一维等比数列数组    np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
    #                       base参数为对数log的底数，默认为10
    x12 = np.logspace(0, 9, 10, base=2)


# 操作numpy数组
def operate_np():
    """
        对数组的常见操作
    :return:
    """
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.arange(1, 4).reshape(1, 3)
    print(np.array([1, 2, 3, 4, 5], ndmin=2))  # 改变数组维度方式1
    print(x.reshape(3, 2))  # 改变数组维度方式2,reshape返回的数组和原数组共享同一段内存
    print(x.resize((3, 2)))  # 改变数组的维度方式3,返回一份浅拷贝
    x.shape = (3, 2)  # 改变数组的维度方式4
    print(x.reshape(-1))  # 展平数组方式1
    print(x.ravel())  # 展平数组方式2
    print(np.sum(x))  # 所有元素之和
    print(np.sum(x, axis=0))  # 指定沿某个轴
    print(np.sum(x, axis=1))  # 指定沿某个轴
    print(np.mean(x, axis=1))  # 求平均值,也可以指定某个轴
    print(np.tile(x, (1, 2)))  # 把原数组整体变为所给的元组形状
    print(np.argsort(x))  # 将元素按照从小到大排序,返回对应位置的下标,可以指定轴
    print(np.concatenate((x, y),axis=0))    # 矩阵的连接
    print(np.inner(np.array([1,2,3,4]), np.array([1,2,3,4])))    # 向量的内积
    print(x.dot(np.arange(6).reshape(2, 3)))  # 矩阵乘法
    print(x.T)  # 矩阵的转置方式1
    print(np.transpose(x))  # 矩阵的转置方式2
    print(np.linalg.inv(np.arange(4).reshape(2, 2)))  # 矩阵的逆
    print(np.append([1, 2, 3], [4, 5, 6]))  # 数组的末尾追加值append(数组，值，轴)，始终返回一个一维数组
    print(np.insert([1, 2, 3], 2, [5, 6]))  # insert(数组，在其之前插入值的索引，要插入的值，轴（如未提供，则数据被展开）)函数在给定索引之前，沿给定轴在输入数组中插入值
    print(np.delete([1, 2, 3, 4],
                    [1, 2]))  # delete（原数组，要删除的子数组，轴）函数返回从输入数组中删除指定子数组的新数组。与insert()函数的情况一样，如果未提供轴参数，则输入数组将展开。
    print(np.unique([1, 1, 2, 3, 4]))  # unique(输入数组(不是一维的话就展开)) 函数用于去除数组中的重复元素。
    # 使用nditer()迭代数组可以实现迭代每个标量元素。
    for i in np.nditer(np.array([[1, 2, 3, 4], [5, 6, 7, 8]])):
        print(i)


# 索引和切片numpy数组
def index_and_slice():
    """
        索引和切片操作
    :return:
    """
    # 切片操作
    x = np.arange(10)
    y = np.arange(16).reshape(4, 4)
    print(slice(1, 9, 2))
    print(x[1:9:2])
    print(x[1:])
    print(x[:-1])
    print(y[..., 1])  # 切片还可以包括省略号 …，来使选择元组的长度与数组的维度相同。
    print(y[2:, ...])   # ,被用来分割维度

    # 索引操作
    a = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
    print(a[1,2])   # 取某个确定值
    print(a[-2:,1:3])   # 取某个范围值
    a[np.arange(3),1]+=10   # 对a的某部分值进行运算
    print(a > 10)   # 获取数组中大于或小于某个确定值的数值
    print(a[a>10])


# 字符串操作函数
def str_np():
    """
        numpy中常见的操作字符串的函数
    :return:
    """
    # 连接两个字符串
    np.char.add('hello', 'world')

    # 多重连接
    np.char.multiply('runoob', 4)

    # 首字母大写
    np.char.capitalize('runoob')

    # 每个单词的首字母都大写
    np.char.title('i like you ')

    # 每个字母都转化为小写
    np.char.lower('AJFKS')

    # 每个字母都转化为大写
    np.char.upper('owehrw')

    # 通过指定分隔符对字符串进行分割，并返回数组
    np.char.split('www.runoob.com', sep='.')

    # 以换行符作为分隔符来分割字符串，并返回数组
    np.char.splitlines('i\nlike runoob?')

    # 移除开头或结尾的特定字符
    np.char.strip(['arunooba', 'admin', 'java'], 'a')

    # 通过指定分隔符来连接数组中的元素或字符串
    np.char.join(':', 'runoob')
    np.char.join([':', '-'], ['runoob', 'google'])

    # 使用新字符串替换字符串中的所有子字符串。
    np.char.replace('i like runoob', 'oo', 'cc')


# numpy数学函数
def math_np():
    # numpy位运算函数
    '''
        bitwise_and	对数组元素执行位与操作
        bitwise_or	对数组元素执行位或操作
        invert	    按位取反
        left_shift	向左移动二进制表示的位
        right_shift	向右移动二进制表示的位
    '''
    print(np.bitwise_and(13, 17))
    print(np.bitwise_or(13, 17))
    print(np.invert(13))
    print(np.left_shift(10, 2))
    print(np.right_shift(40, 2))
    a = np.array([0, 30, 45, 60, 90])
    # 不同角度的正弦值
    np.sin(a * np.pi / 180)
    # 不同角度的余弦值
    np.cos(a * np.pi / 180)
    # 不同角度的正切值
    np.tan(a * np.pi / 180)
    # 返回指定数字的四舍五入值 第一个参数为数组，第二个参数为精度
    np.around(a)
    # 返回≤指定表达式的最大整数
    np.floor(a)
    # 返回≥指定表达式的最小整数
    np.ceil(a)
    # 数组a和数组b的加减乘除
    # np.add(a, b)
    # np.subtract(a, b)
    # np.multiply(a, b)
    # np.divide(a, b)
    # 返回数组逐元素的倒数
    np.reciprocal(a)
    # 将第一个输入数组中的元素作为底数，计算它与第二个输入数组中相应元素的幂
    np.power(a, 2)

    # NumPy统计函数
    # 数组沿指定轴的最大最小值
    np.amin(a, axis=0)
    np.amax(a, axis=0)

    # 计算数组中最大值和最小值的差
    # np.ptp(a, b)

    # 计算数组元素的中位数
    np.median(a)

    # 计算数组的算术平均数
    np.mean(a)

    # 计算数组的加权平均值
    # np.average(a, [1, 2, 3, 4])

    # 数组的标准差        std = sqrt(mean((x - x.mean())**2))
    np.std(a)

    # 数组的方差          方差= mean((x - x.mean())** 2)
    np.var(a)
