from tkinter import *
from tkinter import messagebox

"""
    详细参考见：https://blog.csdn.net/qq_41854911/article/details/122697073
    十五种标准控件：
        Button:按钮控件，在程序中显示按钮,点击按钮出发一些事件
        Canvas:画布控件，显示图形元素如线条和文本
        Checkbutton:多选框控件，在程序中提供多选框
        Entry:输入框控件，用于显示简单的文本内容
        Frame:框架控件，在屏幕上显示一个矩形区域，多用来作容器
        Label:标签控件，可以显示文本和位图
        Listbox:列表框控件，显示一个字符串列表给用户
        Menubutton:菜单按钮，显示菜单项
        Menu:菜单控件，显示菜单栏，下拉菜单和弹出菜单
        Message:消息控件，用来显示多行文本
        Radiobutton:单选按钮控件，用来显示一个单选的按钮状态
        Scale:范围控件，显示一个数值刻度，为输出限定范围的数字区间
        Scrollbar:滚动条控件，当内容超过可视化区域时使用，如列表框
        Text:文本控件，用来显示多行文本
        Toplevel:容器控件，用来提供一个单独的对话框
        Spinbox:输入控件，可以指定输入范围值
        PanedWindow:窗口布局管理插件
        LabelFrame:一个简单的容器控件，常用于复杂的窗口布局
        tkMessageBox:用来显示你应用程序的消息框
    
    控件的共有参数：
        anchor：定义控件或文字信息在窗口内的位置
        bg：定义控件的背景颜色   
        fg：定义控件的前景颜色
        bitmap：定义显示在控件内的位图文件
        borderwidth：定义控件的边框宽度
        command：执行事件函数
        cursor：当鼠标指针移动到控件上时，定义鼠标指针的类型
        font：三元组（字体，大小，样式）
        height/width：定义控件的高和宽，单位为字符
        justify：定义多行文字的排列方式
        padx/pady:定义控件内的文字或图片与控件边框之间的水平/垂直距离
        relief：定义控件的边框样式
        text：定义控件的标题文字
        state：控制控件是否处于可用状态
"""

top = Tk()  # 顶层窗口
top.geometry('500x500')  # 调整顶部窗口大小


def _button():
    """
        button控件示例,按钮上可以放上文本或图像,按钮可用于监听用户行为,能够与一个Python函数关联,当按钮被按下时,自动调用该函数。
    """
    top = Tk()
    top.geometry("200x100")

    def fun():
        messagebox.showinfo("Hello", "Red Button clicked")
    b1 = Button(top, text="Red", command=fun, activeforeground="red", activebackground="pink", pady=10)
    b2 = Button(top, text="Blue", activeforeground="blue", activebackground="pink", pady=10)
    b3 = Button(top, text="Green", activeforeground="green", activebackground="pink", pady=10)
    b4 = Button(top, text="Yellow", activeforeground="yellow", activebackground="pink", pady=10)
    b1.pack(side=LEFT)
    b2.pack(side=RIGHT)
    b3.pack(side=TOP)
    b4.pack(side=BOTTOM)

    top.mainloop()


def _label():
    """
        label控件示例
    """
    hello = Label(top, text="label")
    hello.pack()


def _scale():
    # scale控件示例
    scale = Scale(top, from_=10, to=80, orient=HORIZONTAL)
    scale.pack(fill=X, expand=1)


def _frame():
    """

    """

    def say_hi():
        print("hello~")

    frame1 = Frame(top, bd=4)
    frame2 = Frame(top)
    top.title("tkinter frame")

    label = Label(frame1, text="Label", justify=LEFT)
    label.pack(side=LEFT)

    hi_there = Button(frame2, text="say hi~", command=say_hi)
    hi_there.pack()

    frame1.pack(padx=1, pady=1)
    frame2.pack(padx=10, pady=10)


def _canvas():
    """
        画布小部件用于将结构化图形添加到 python 应用程序。
        它用于将图形和绘图绘制到 python 应用程序。
    """
    top = Tk()
    top.geometry("200x200")
    # creating a simple canvas
    c = Canvas(top, bg="pink", height="200", width=200)
    arc = c.create_arc((5, 10, 150, 200), start=0, extent=150, fill="white")
    c.pack()
    top.mainloop()


def _entry():
    """
        输入控件，用来显示简单的文本内容
        常见方法：
            delete(first,last=None):删除文本框里直接位置值
            get():获取文本框里的值

    """
    top = Tk()
    top.geometry("400x250")
    name = Label(top, text="Name").place(x=30, y=50)
    email = Label(top, text="Email").place(x=30, y=90)
    password = Label(top, text="Password").place(x=30, y=130)
    sbmitbtn = Button(top, text="Submit", activebackground="pink", activeforeground="blue").place(x=30, y=170)
    e1 = Entry(top).place(x=80, y=50)
    e2 = Entry(top).place(x=80, y=90)
    e3 = Entry(top).place(x=95, y=130)
    top.mainloop()


def _checkbutton():
    """
        Checkbutton用于跟踪提供给应用程序的用户选择。
    """
    top = Tk()
    top.geometry("200x200")
    checkvar1 = IntVar()
    checkvar2 = IntVar()
    checkvar3 = IntVar()
    chkbtn1 = Checkbutton(top, text="C", variable=checkvar1, onvalue=1, offvalue=0, height=2, width=10)
    chkbtn2 = Checkbutton(top, text="C++", variable=checkvar2, onvalue=1, offvalue=0, height=2, width=10)
    chkbtn3 = Checkbutton(top, text="Java", variable=checkvar3, onvalue=1, offvalue=0, height=2, width=10)
    chkbtn1.pack()
    chkbtn2.pack()
    chkbtn3.pack()
    top.mainloop()



