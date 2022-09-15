import csv
import time
import matplotlib.pyplot as plt
from urllib.request import urlopen

from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D  # 3D包
import matplotlib.dates as mdates  # 转换日期
import numpy as np

# 设置中文显示
rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


"""
    单个图像的构成：
        axis:单个数轴
        axes:指坐标系
        figure:指整个画面,可以包含多个坐标系
        artist:画面中所有可以看到的东西都是一个artist对象
"""

# # 画图方式1
# fig = plt.figure()


def oop_grammar():
    """
        面向对象的精确语法
    :return:
    """
    seasons = [1, 2, 3, 4]
    stock1 = [4, 8, 2, 6]
    stock2 = [10, 12, 5, 1]
    # fig = plt.figure()  # 初始化,返回一个figure画面
    # ax = fig.add_subplot() # 添加一个Axes
    # ax = plt.subplot()  # 初始化并且添加一个Axes坐标系
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))  # axes:坐标系构成的numpy数组
    axes[0, 0].bar(seasons, stock1)
    axes[0, 1].plot(seasons, stock2, "b^--")
    ax = axes[1, 0]
    ax.scatter(seasons, np.array(stock2)-np.array(stock1), s=[10, 20, 50, 100], c=['r', 'b', 'c', 'y'])
    axes[1, 1].remove()
    axes[0, 0].set_title("股票1")
    axes[0, 1].set_title("股票2")
    ax.set_title("股票2-股票1")
    fig.suptitle("股票分析图")   # 整个figure的标题
    fig.supylabel("股价") # 整个figure的y轴标签
    fig.supxlabel("季度") # 整个figure的x轴标签
    plt.tight_layout()  # 内容紧凑
    plt.show()


def legend_title_label_grid():
    """
        图例，标题和标签等公用信息
    """
    x = [1, 2, 3]
    y = [5, 7, 4]
    x2 = [1, 2, 3]
    y2 = [10, 14, 12]

    plt.plot(x, y, 'r.-', label='First Line')   # 第三个参数指定画图的样式
    plt.plot(x2, y2, 'go--', label='Second Line')

    plt.title("test")
    plt.xlim(0, 10)  # x轴范围
    plt.ylim(0, 16)  # y轴范围
    plt.xlabel('Plot Number')  # x轴标签
    plt.ylabel('Important var')  # y轴标签
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # x轴刻度
    plt.yticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])  # y轴刻度
    plt.legend()  # 根据label值添加标签,参数为loc位置
    plt.grid()  # 网格线
    plt.show()


def axe():
    """
        坐标轴相关
    :return:
    """
    ax = plt.gca()  # 获取当前坐标轴信息
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["top"].set_visible(False)  # 设置顶边界不可见
    ax.spines["right"].set_visible(False)  # 设置顶边界不可见
    plt.show()


def bar():
    """
        条形图
    :return:
    """
    plt.bar([1, 3, 5, 7, 9], [5, 2, 7, 8, 2], label="Example one")
    plt.bar([2, 4, 6, 8, 10], [8, 6, 2, 5, 6], label="Example two", color='g')
    plt.legend()
    plt.xlabel('bar number')
    plt.ylabel('bar height')
    plt.title('Epic Graph\nAnother Line! Whoa')
    plt.show()


def hist():
    '''
        直方图
    :return:
    '''
    population_ages = [22, 55, 62, 45, 21, 22, 34, 42, 42, 4, 99, 102, 110, 120, 121, 122, 130, 111, 115, 112, 80, 75,
                       65, 54, 44, 43, 42, 48]
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    plt.hist(population_ages, bins, histtype='bar', rwidth=0.8, label='hist graph')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interesting Graph\nCheck it out')
    plt.legend()
    plt.show()


def scatter():
    """
        散点图
    :return:
    """
    girls_grades = [89, 90, 70, 89, 100, 80, 90, 100, 80, 34]
    boys_grades = [30, 29, 49, 48, 100, 48, 38, 45, 20, 30]
    grades_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.scatter(grades_range, girls_grades, color='r', label="girls")
    plt.scatter(grades_range, boys_grades, color='b', label="boys")
    plt.xlabel('Grades Range')
    plt.ylabel('Grades Scored')
    plt.title('scatter plot')
    plt.legend()
    plt.show()


def pie():
    """
        饼图
    :return:
    """
    slices = [7, 2, 2, 13]
    activities = ['sleeping', 'eating', 'working', 'playing']
    cols = ['c', 'm', 'r', 'b']
    plt.pie(slices,
            labels=activities,
            colors=cols,
            startangle=90,
            shadow=True,
            explode=(0, 0.1, 0, 0),
            autopct='%1.1f%%')
    plt.title('Interesting Graph\nCheck it out')
    plt.show()


def plot():
    """
        折线图
    :return:
    """
    # 准备绘制数据
    x = ["Mon", "Tues", "Wed", "Thur", "Fri", "Sat", "Sun"]
    y = [20, 40, 35, 55, 42, 80, 50]
    # "g" 表示红色，marksize用来设置'D'菱形的大小
    plt.plot(x, y, "g", marker='D', markersize=5, label="周活")
    # 绘制坐标轴标签
    plt.xlabel("登录时间")
    plt.ylabel("用户活跃度")
    plt.title("C语言中文网活跃度")
    # 显示图例
    plt.legend(loc="lower right")
    # 调用 text()在图像上绘制注释文本
    # x1、y1表示文本所处坐标位置，ha参数控制水平对齐方式, va控制垂直对齐方式，str(y1)表示要绘制的文本
    for x1, y1 in zip(x, y):
        plt.text(x1, y1, str(y1), ha='center', va='bottom', fontsize=10)
    # 保存图片
    plt.savefig("1.jpg")
    plt.show()


def read_from_file():
    """
        从文件中读入数据
    :return:
    """
    x = []
    y = []
    with open('data/example.txt', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')  # csv读取器自动按行分割文件，然后使用我们选择的分隔符分割文件中的数据。
        for row in plots:
            x.append(int(row[0]))
            y.append(int(row[1]))
    plt.plot(x, y, label='Loaded from file!')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interesting Graph\nCheck it out')
    plt.legend(loc="upper right")
    plt.show()


def read_from_network():
    """
        从网络中获取数据
    :return:
    """
    stock_price_url = 'http://chartapi.finance.yahoo.com/instrument/1.0/' + 'stock' + '/chartdata;type=quote;range=10y/csv'

    source_code = urlopen(stock_price_url).read().decode()
    stock_data = []
    split_source = source_code.split('\n')
    for line in split_source:
        split_line = line.split(',')
        if len(split_line) == 6 and 'values' not in line:
            stock_data.append(line)


def _3d():
    '''
        3D图
    :return:
    '''
    fig = plt.figure(figsize=(16, 12))  # 定义图像窗口
    # ax = fig.add_subplot(111, projection='3d')    # 添加3D坐标轴方式1
    ax = Axes3D(fig)  # 添加3D坐标轴方式2
    # X, Y value
    theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    X = np.sin(theta)
    Y = np.cos(theta)

    # 折线图
    def _plot():
        """
            在三维坐标系可以用plot函数绘制三维的线条，还可以绘制平面曲线。
        ax3d.plot(x,y,z)：绘制三维曲线。
        zdir参数绘制平面图
            ax3d.plot(x,y,zdir=‘z’)：在z=0的xy平面绘制曲线
            ax3d.plot(x,y,2,zdir=‘z’)：在z=2的xy平面绘制曲线
            ax3d.plot(y,z,zdir=‘x’)：在x=0的yz平面绘制曲线。zdir也可以为’y’
            ax3d.plot(y,z,2,zdir=‘x’)：在x=2的yz平面绘制曲线

        :return:
        """
        Z = np.linspace(-2, 2, 100)
        ax.plot(X, Y, Z)
        ax.plot(X, Y, zdir='z', label="z=0")
        ax.plot(X, Y, 2, zdir='z', label="z=2")
        ax.plot(Y, Z, zdir='x', label="x=0")
        ax.plot(Y, Z, 3, zdir='x', label="x=3")
        plt.legend()
        plt.show()

    # 散点图
    def _scatter():
        """
            在三维坐标系可以用scatter函数绘制三维的散点图，还可以绘制平面散点图。
                ax3d.scatter(x,y,zdir=‘z’)：在z=0的xy平面绘制散点图
                ax3d.scatter(x,y,2,zdir=‘z’)：在z=2的xy平面绘制散点图
                ax3d.scatter(y,z,zdir=‘x’)：在x=0的yz平面绘制散点图。zdir也可以为’y’
                ax3d.scatter(y,z,2,zdir=‘x’)：在x=2的yz平面绘制散点图
        :return:
        """
        x = np.random.randn(50)
        y = np.random.randn(50)
        z = np.random.randn(50)
        s = np.random.randn(50) * 100

        ax.scatter(x, y, z)  # 绘制3d散点图
        ax.scatter(x, y, z, marker="o")  # 设置不同的点样式
        ax.scatter(x, y, z, s=s, c=s)  # 绘制3d散点图
        ax.scatter(x, y, -3, zdir='z', c='r')  # 3d坐标系绘制平面散点
        ax.scatter(x, y, zdir='z', c='r')  # 3d坐标系绘制平面散点
        ax.scatter(x, z, 2, zdir='y')
        ax.scatter(y, z, 2, zdir='z')

        plt.show()

    # 曲面图
    def plot_surface():
        """
        x,y,z均为二维数组，根据数据绘制曲面
        :return:
        """
        x, y = np.mgrid[-3:3:0.2, -3:3:0.2]
        z = x * np.exp(-x ** 2 - y ** 2)

        # ax.plot_surface(x,y,z)
        # ax.plot_surface(x,y,z,rstride=2,cstride=2)# 两条线合并为一条线
        # ax.plot_surface(x,y,z,rcount=16,ccount=18)#设置最大显示线条数
        ax.plot_surface(x,y,z,cmap="YlOrRd")
        # ax.plot_surface(x, y, z, cmap="YlOrRd")

        plt.show()

    # 3D空间里的二维平面图
    def _plan():
        """

        :return:
        """
        # 二元函数定义域平面
        x = np.linspace(0, 9, 9)
        y = np.linspace(0, 9, 9)
        X, Y = np.meshgrid(x, y)

        ax.plot_surface(X, Y, Z=X*0 + 4.5, color = 'g')
        ax.plot_surface(X, Y, Z=X**2*1 + Y*1 + 4.5, color = 'r')

        # 设置坐标轴标题和刻度
        ax.set(xlabel='X',
               ylabel='Y',
               zlabel='Z',
               )
        plt.show()

    _plan()

    #
    # X, Y = np.meshgrid(X, Y)  # x-y 平面的网格
    # R = np.sqrt(X ** 2 + Y ** 2)
    # # height value
    # Z = np.sin(R)
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow')) # 颜色为彩虹色rainbow,rstrude/cstride代表跨度
    # plt.show()


