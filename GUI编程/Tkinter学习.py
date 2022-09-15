"""
    Tkinter是Python的标准GUI库。Python使用Tkinter可以快速的创建GUI应用程序。
由于Tkinter是内置到python的安装包中、只要安装好Python之后就能import Tkinter库、而且IDLE也是用Tkinter编写而成、对于简单的图形界面Tkinter还是能应付自如。
"""

from tkinter import *

# 窗口主体框架
window = Tk()  # 1.定义window窗口
window.title("my window")  # 2.定义window窗口的一些属性
window.geometry("500x500")  # 设置窗口大小和位置
window.config(background="black")  # 设置背景颜色
# window.withdraw()  # 将Tkinter.Tk()实例隐藏
window.update()  # 设置实时更新窗口(!!!!!!!!!!!!!)　
# 3.定义窗口内容
window.mainloop()  # 4.让窗口活起来

# 窗口布局方法
"""
    1.pack():pack()小部件用于组织块中的小部件。使用pack()方法添加到python应用程序的位置小部件可以通过使用方法调用中指定的各种选项来控制。
    2.grid():grid()几何管理器以表格形式组织小部件。我们可以将行和列指定为方法调用中的选项。我们还可以指定小部件的列跨度（宽度）或行跨度（高度）。
    3.place():place()几何管理器将小部件组织到特定的x和y坐标。
"""


def _pack():
    """
        可选参数:
            expand:如果expand设置为true，则小部件会扩展以填充任何空间。
            fill:默认情况下，填充设置为NONE。但是，我们可以将其设置为X或Y以确定小部件是否包含任何额外空间。
            side:它表示窗口上要放置小部件的父级的哪一侧。
    :return:
    """
    parent = Tk()
    redbutton = Button(parent, text="Red", fg="red")
    redbutton.pack(side=LEFT)
    greenbutton = Button(parent, text="Black", fg="black")
    greenbutton.pack(side=RIGHT)
    bluebutton = Button(parent, text="Blue", fg="blue")
    bluebutton.pack(side=TOP)
    blackbutton = Button(parent, text="Green", fg="red")
    blackbutton.pack(side=BOTTOM)
    parent.mainloop()


def _grid():
    """
        可选参数:
            Column:要放置小部件的列号,最左边的列用0表示。
            Columnspan:小部件的宽度,它表示将列展开到的列数。
            ipadx, ipady:它表示在小部件边框内填充小部件的像素数。
            padx, pady:它表示在小部件边框之外填充小部件的像素数。
            row:要放置小部件的行号。最上面一行用 0 表示
            rowspan:小部件的高度，即小部件展开到的行数。
            sticky:如果单元格大于小部件，则使用粘性来指定小部件在单元格内的位置。它可能是表示小部件位置的粘性字母的串联。它可能是 N、E、W、S、NE、NW、NS、EW、ES。
    :return:
    """
    parent = Tk()
    Label(parent, text="Name").grid(row=0, column=0)
    Entry(parent).grid(row=0, column=1)
    Label(parent, text="Password").grid(row=1, column=0)
    Entry(parent).grid(row=1, column=1)
    Button(parent, text="Submit").grid(row=4, column=0)
    parent.mainloop()


def _place():
    """
        可选参数:
            Anchor:它表示小部件在容器中的确切位置。默认值（方向）为 NW（左上角）
            bordermode:边框类型的默认值为INSIDE，表示忽略父边框的内部。另一个选项是 OUTSIDE。
            height,width:它是指以像素为单位的高度和宽度。
            relheight,relwidth:表示为0.0到1.0之间的浮点数，表示父级的高度和宽度的百分比。
            relx,rely: 它表示为0.0和1.0之间的浮点数，即水平和垂直方向的偏移量。
            x,y:它指的是像素中的水平和垂直偏移量。
    :return:
    """
    top = Tk()
    top.geometry("400x250")
    Label(top, text="Name").place(x=30, y=50)
    Label(top, text="Email").place(x=30, y=90)
    Label(top, text="Password").place(x=30, y=130)
    Entry(top).place(x=80, y=50)
    Entry(top).place(x=80, y=90)
    Entry(top).place(x=95, y=130)
    top.mainloop()
